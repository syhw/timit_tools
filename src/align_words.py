import os, sys, joblib, random
from multiprocessing import cpu_count
from collections import defaultdict
import numpy as np
from dtw import DTW
from mfcc_and_gammatones import FBANKS_RATE

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
    output_name = "dtw_words"
    if folder != ".":
        output_name += "_" + folder.split('/')[-1]

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
