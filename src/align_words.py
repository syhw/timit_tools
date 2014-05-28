import os, sys, joblib
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
    return word, x, y, dtw[0], dtw[-1][0]


def match_words(d, min_len_word_char=4, omit_words=['the'], before_after=3,
        serial=False):
    """ Matches same words, extracts their filterbanks, performs DTW, returns
    a list of tuples:
    [(word_label, fbanks1, fbanks2, DTW_cost, DTW_alignment)]

    Parameters:
      - d: a dictionary of word->files (output of find_words(folder))
      - min_len_word_char: (int) minimum length for the words to consider
        (in characters).
      - omit_words: ([str]) (list of strings), words to omit / not align.
      - before_after: (int) number of frames to take before and after (if
        possible) the start and the end of the word.
      - serial: (bool) good ol' Python on one core if True, joblibed otherwise
    """
    
    #print d
    print "features rate (number of features vector per second)", FBANKS_RATE
    words_feats = defaultdict(lambda: [])
    for word, l in d.iteritems():
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
        res = joblib.Parallel(n_jobs=20)(joblib.delayed(do_dtw)(word, l[i], y)
                    for word, l in words_feats.iteritems()
                        for i, x in enumerate(l)
                            for j, y in enumerate(l)
                                if i < j)
    return res


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        folder = sys.argv[1].rstrip('/')
    print "working on folder:", folder
    output_name = "dtw_words"
    if folder != ".":
        output_name += "_" + folder.split('/')[-1]
    joblib.dump(match_words(find_words(folder)), output_name + ".joblib",
            compress=4, cache_size=512)
    # compress doesn't work for too big datasets!
