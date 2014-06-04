import os, sys, joblib, random
from multiprocessing import cpu_count
from collections import defaultdict
import numpy as np
from dtw import DTW
from mfcc_and_gammatones import FBANKS_RATE
from itertools import izip
from random import shuffle

OLD_SCHEME = False

def find_words(folder):
    """ Recursively traverses the given folder and returns a dictionary with
    {'word': [(filename, start, end)]} with start and end in seconds.
    """

    words = defaultdict(lambda: [])
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-4:] != '.wrd':
                continue
            fullfname = d + '/' + fname
            fr = open(fullfname)
            for line in fr:
                [s, e, p] = line.rstrip('\n').split()
                s = float(s) / 16000 # in sec TODO wavfile open 
                e = float(e) / 16000 # in sec TODO + take sampling rate
                words[p].append((fullfname, s, e))
            fr.close()
    return words


def do_dtw(word, x, y):
    dtw = DTW(x, y, return_alignment=1)
    # word, x, y, cost_dtw, dtw_x_to_y_mapping, dtw_y_to_x_mapping
    return word, x, y, dtw[0], dtw[-1][1], dtw[-1][2]


def do_dtw_pair(p1, p2):
    dtw = DTW(p1[1], p2[1], return_alignment=1)
    # word, x, y, cost_dtw, dtw_x_to_y_mapping, dtw_y_to_x_mapping
    return p1[0], p1[1], p2[1], dtw[0], dtw[-1][1], dtw[-1][2]


def extract_features(tup, before_after=3):
    word, fname, s, e = tup
    sf = s * FBANKS_RATE
    ef = e * FBANKS_RATE
    fb = None
    with open(fname.split('.')[0] + "_fbanks.npy", 'rb') as f:
        fb = np.load(f)
    if fb == None:
        print >> sys.stderr, "problem with file", fname
        return
    before = max(0, sf - before_after)
    after = min(ef + before_after, fb.shape[0])
    return word, fb[before:after]


def pair_same_and_diff_words(words_timings, min_len_word_char=5): 
    """ Returns a pair (same, diff) of list of pairs of words 
    ('word', 'fname', 'start', 'end') that are matched in talker %.
    """
    # generate all pairs
    same_talker_same_word = []
    same_talker_diff_word = []
    diff_talker_same_word = []
    diff_talker_diff_word = []
    for i, (word1, l1) in enumerate(words_timings.iteritems()):
        if len(word1) < min_len_word_char:
            continue
        for ii, (fname1, s1, e1) in enumerate(l1):
            for j, (word2, l2) in enumerate(words_timings.iteritems()):
                if i > j:  # symmetric
                    continue
                if len(word2) < min_len_word_char:
                    continue
                for jj, (fname2, s2, e2) in enumerate(l2):
                    if i == j and ii > jj:
                        continue
                    talker1 = fname1.split('/')[-2]
                    talker2 = fname2.split('/')[-2]
                    pair = ((word1, fname1, s1, e1), (word2, fname2, s2, e2))
                    if talker1 == talker2:
                        if i == j:
                            same_talker_same_word.append(pair)
                        else:
                            same_talker_diff_word.append(pair)
                    else:
                        if i == j:
                            diff_talker_same_word.append(pair)
                        else:
                            diff_talker_diff_word.append(pair)
        print "same talker same word len:", len(same_talker_same_word)
        print "same talker diff word len:", len(same_talker_diff_word)
        print "diff talker same word len:", len(diff_talker_same_word)
        print "diff talker diff word len:", len(diff_talker_diff_word)
    # sample in pairs uniformly (without replacement)
    same_words = []
    for pair_same_talker, pair_diff_talker in izip(same_talker_same_word, diff_talker_same_word):
        same_words.append(pair_same_talker)
        same_words.append(pair_diff_talker)
    diff_words = []
    shuffle(same_talker_diff_word)
    shuffle(diff_talker_diff_word)
    for pair_same_talker, pair_diff_talker, _ in izip(same_talker_diff_word, diff_talker_diff_word, xrange(len(same_words)/2)):
        diff_words.append(pair_same_talker)
        diff_words.append(pair_diff_talker)
    return same_words, diff_words


def pair_word_features(words_timings, min_len_word_char=3, before_after=3,
        omit_words=['the']):
    """ Extract features (filterbanks by default) for all words.

    Parameters:
      - words_timings: (dict) dictionary of words in the dataset and the
                       files and timings at which they appear in these files.
      - min_len_word_char: (int) minimum length for the words to consider
                           (in characters).
      - omit_words: ([str]) (list of strings), words to omit / not align.
    """
    words_feats = defaultdict(lambda: [])
    for word, l in words_timings.iteritems():
        if len(word) < min_len_word_char or word in omit_words: 
            continue
        for fname, s, e in l:
            sf = s * FBANKS_RATE
            ef = e * FBANKS_RATE
            fb = None
            with open(fname.split('.')[0] + "_fbanks.npy") as f:
                fb = np.load(f)
            if fb == None:
                print >> sys.stderr, "problem with file", fname
                continue 
            before = max(0, sf - before_after)
            after = min(ef + before_after, fb.shape[0])
            #new_word_start = TODO
            #new_word_end = TODO
            words_feats[word].append(fb[before:after])
    return words_feats


def match_words(words_feats, serial=False):
    """ Matches same words, extracts their filterbanks, performs DTW, returns
    a list of tuples:
    [(word_label, fbanks1, fbanks2, DTW_cost, DTW_alignment)]

    Parameters:
      - words_feats: a dictionary of word->fbanks 
                     (the output of pair_word_features(words_timing_dict))
      - before_after: (int) number of frames to take before and after (if
        possible) the start and the end of the word.
      - serial: (bool) good ol' Python on one core if True, joblibed otherwise
    """
    
    #print d
    print "features rate (number of features vector per second)", FBANKS_RATE
    res = []
    if serial:
        for word, l in words_feats.iteritems():
            print word
            for i, x in enumerate(l):
                for j, y in enumerate(l):
                    if i >= j:  # that's symmetric!
                        continue
                    res.append(do_dtw(word, x, y))
    else:
        res = joblib.Parallel(n_jobs=cpu_count()-1)(joblib.delayed(do_dtw)
                (word, l[i], y)
                    for word, l in words_feats.iteritems()
                        for i, x in enumerate(l)
                            for j, y in enumerate(l)
                                if i < j)
    return res


def sample_words(words_feats, n_words):
    """ Randomly samples words and include them as negative examples.

    [(fbanks1, fbanks2)]
    """
    res = []
    n = 0
    skeys = sorted(words_feats.keys())
    lkeys = len(skeys) - 1
    while n < n_words:
        w1 = random.randint(0, lkeys)
        w2 = random.randint(0, lkeys)
        if w1 == w2:
            continue
        fb1 = 0
        if len(words_feats[w1]) > 1:
            fb1 = random.randint(0, len(words_feats[w1]) - 1)
        fb2 = 0
        if len(words_feats[w2]) > 1:
            fb2 = random.randint(0, len(words_feats[w2]) - 1)
        s1 = words_feats[skeys[w1]][fb1]
        s2 = words_feats[skeys[w2]][fb2]
        res.append((s1[:min(len(s1), len(s2))], s2[:min(len(s1), len(s2))]))
        n += 1
    return res


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        folder = sys.argv[1].rstrip('/')
    print "working on folder:", folder
    output_name = "dtw_words_eq"
    if folder != ".":
        output_name += "_" + folder.split('/')[-1]

    if OLD_SCHEME:
        words_feats = pair_word_features(find_words(folder), min_len_word_char=5)
        print "number of words in all (not pairs!):", len(words_feats)

        matched_words = match_words(words_feats)
        print "number of word pairs:", len(matched_words)
        joblib.dump(matched_words, output_name + ".joblib",
                compress=5, cache_size=512)
        # compress doesn't work for too big datasets!

        output_name = "neg" + output_name[3:]
        joblib.dump(sample_words(words_feats, len(matched_words)),
                output_name + ".joblib", compress=5, cache_size=512)
    else:
        words_timings = find_words(folder)
        print "number of words in all (not pairs!):", len(words_timings)
        same, diff = pair_same_and_diff_words(words_timings)
        same_pairs = map(extract_features, same)  # TODO joblib.Parallel
        diff_pairs = map(extract_features, diff)  # TODO joblib.Parallel
        same_words = joblib.Parallel(n_jobs=cpu_count()-1)(joblib.delayed(do_dtw_pairs)
                (sp[0], sp[1]) for sp in same_pairs)
        diff_words = [(dp[0][1][:min(len(dp[0][1]), len(dp[1][1]))],
            dp[1][1][:min(len(dp[0][1]), len(dp[1][1]))]) for dp in diff_pairs]
        
        print "number of word pairs:", len(same_words)
        joblib.dump(same_words, output_name + ".joblib",
                compress=5, cache_size=512)
        # compress doesn't work for too big datasets!

        output_name = "neg" + output_name[3:]
        joblib.dump(diff_words, output_name + ".joblib",
                compress=5, cache_size=512)

