import os, copy, sys
try:
    import htkmfc
except:
    print >> sys.stderr, "ERROR: You don't have htkmfc"
    sys.exit(-1)
import numpy as np
import scipy.stats.stats as sss


def normalize(folder):
    corpus = {}
    full = np.ndarray((0,39))

    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-11:] != '.mfc_unnorm':
                continue
            fullfname = d + '/'+fname
            t = htkmfc.open(fullfname)
            corpus[fullfname[:-11]+'_mfc.npy'] = copy.deepcopy(t.getall())
            full = np.append(full, t.getall(), axis=0)

    mean = np.mean(full)
    stddev = sss.tstd(full)
    if stddev == 0:
        print >> sys.stderr, "*** null stddev, no *.mfc_unnorm file ??? ***"
        sys.exit(-1)

    for key,val in corpus.iteritems():
        corpus[key] = (val - mean) / stddev

    # verification:
    ### full = np.ndarray((0,39))
    ### for key,val in corpus.iteritems():
    ###     full = np.append(full, val, axis=0)
    ### print "verification of 0-mean 1-stddev"
    ### print "mean (big numeric errors, beware)"
    ### print np.mean(full)
    ### print "stddev"
    ### print sss.tvar(full)
    # /verification

    for key,val in corpus.iteritems():
        print "Dealt with:", key
        np.save(key, val)


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    print "Normalizing with all the *.mfc_unnorm in", folder
    print "WARNING: only the first 39 MFCC coefficients will be taken into account"
    normalize(folder)
