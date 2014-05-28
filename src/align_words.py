import os, sys, cPickle
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


def match_words(d, min_len_word_char=4, before_after=3):
    """ Matches same words, extracts their filterbanks, performs DTW, returns
    a list of tuples:
    [(fbanks1, fbanks2, word1_start, word2_start, DTW_alignment)]

    Parameters:
      - d: a dictionary of word->files (output of find_words(folder))
      - min_len_word_char: (int) minimum length for the words to consider
        (in characters)
      - before_after: (int) number of frames to take before and after (if
        possible) the start and the end of the word.
    """
    
    print d
    words_feats = defaultdict(lambda: [])
    for word, l in d.iteritems():
        if len(word) < min_len_word_char: 
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
            before = min(0, sf - before_after)
            after = max(ef + before_after, fb.shape[0])
            #new_word_start = TODO
            #new_word_end = TODO
            words_feats[word].append(fb[before:after])
    res = []
    for word, l in words_feats.iteritems():
        print word
        for i, x in enumerate(l):
            for j, y in enumerate(l):
                if i == j:
                    continue
                res.append((x, y, DTW(x, y, return_alignment=1)[-1]))
    return res


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    print "", folder
    with open("dtw_words.pickle", "wb") as wf:
        cPickle.dump(match_words(find_words(folder)), wf)
