import sys
import numpy as np
from numpy.testing import assert_allclose
import htkmfc
from brian import Hz, log10
from brian.hears import loadsound, erbspace, Gammatone, ApproximateGammatone

"""
We try to reproduce the input of "Deep Belief Networks for phone recognition,
Mohamed et al., NIPS 2009" and other Hinton's group papers.
"""

# TODO load the following parameters from wav_config
SAMPLING_RATE = 16000 # Hz
MFCC_TIMESTEP = 10 # 10 ms
N_MFCC_COEFFS = 39 # as in Mohamed et al. / Dahl et al. (Hinton group) papers
N_FILTERBANK_COEFFS = 40 # as in Acoustic Modeling using Deep Belief Networks
                         # Mohamed et al. TODO

N_GAMMATONES = 50 # c.f. http://www.briansimulator.org/docs/hears.html
                  # and http://www.briansimulator.org/docs/examples-hears_approximate_gammatone.html#example-hears-approximate-gammatone
center_frequencies = erbspace(100*Hz, 1000*Hz, N_GAMMATONES)

TEST = True # test numpy serialization
 
usage = """
    python timit_to_numpy.py MLF_FILENAME.mlf
output files are MLF_FILENAME_xdata.npy, MLF_FILENAME_xgamma.npy, 
and MLF_FILENAME_ylabels.npy
    """


def compute_speed_and_accel(x):
    tmp_diff = np.pad(np.diff(x, axis=0), ((0, 1), (0, 0)), 
            'constant', constant_values=(0.0, 0.0))
    tmp_accel = np.pad(np.diff(tmp_diff, axis=0), ((0, 1), (0, 0)), 
            'constant', constant_values=(0.0, 0.0))
    return np.concatenate((x, tmp_diff, tmp_accel), axis=1)


def subsample_average(arr, n):
    # <=> subsample_apply_f(arr, n, np.mean)
    end = int(n) * int(arr.shape[0]/n)
    return np.mean(arr[:end].reshape(-1, arr.shape[1], n), axis=2)


def subsample_apply_f(arr, n, f):
    end = int(n) * int(arr.shape[0]/n)
    return np.apply_along_axis(f, 2, arr[:end].reshape(-1, arr.shape[1], n))


def extract_from_mlf(mlf):
    x = np.ndarray((0, N_MFCC_COEFFS), dtype='float32')
    x_gamma = np.ndarray((0, N_GAMMATONES*3), dtype='float32')
    y = []
    
    with open(mlf) as f:
        tmp_len_x = 0 # verify sizes
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 1:
                continue
            if line[0] == '"': # TODO remove SA
                if tmp_len_x != 0:
                    print "the file above this one was mismatching x and y lengths", line, "by", tmp_len_x
                # load HTK's MFCC
                t = htkmfc.open(line.strip('"')[:-3] + 'mfc') # .lab -> .mfc
                # load the wav sound (with Brian)
                sound = loadsound(line.strip('"')[:-3] + 'wav') # .lab -> .wav
                # compute the gammatones (see Brian's doc)
                bw = 10**(0.037+0.785*log10(center_frequencies))
                gammatone = ApproximateGammatone(sound, center_frequencies, 
                                                 bw, order=3)
                g = gammatone.process()
                # subsample the gammatones at the same rate than the MFCC's
                # (just for practicality so that they are aligned...)
                n_samples = g.shape[0]*1./(t.getall().shape[0] + 1) # TODO check "+1"
                ### # do the harmonic mean (nth root of the product of the terms)
                ### g_sub = subsample_apply_f(g, n_samples, lambda z: np.power(np.prod(z), 1./n_samples))
                g_sub = subsample_apply_f(g, n_samples, lambda z: np.sqrt(np.sum(np.square(z))))
                # compute the delta and delta of the subsampled gammatones
                gamma_speed_accel = compute_speed_and_accel(g_sub)
                # append
                tmp = gamma_speed_accel[:t.getall().shape[0]] # TODO check
                if tmp.shape[0] != t.getall().shape[0]: # TODO remove
                    print line
                    print tmp.shape
                    print t.getall().shape
                    print n_samples
                    print g.shape
                    print "exiting because of the mismatch"
                    sys.exit(-1)
                x_gamma = np.append(x_gamma, tmp, axis=0)
                x = np.append(x, t.getall(), axis=0)
                tmp_len_x = t.getall().shape[0]
            elif line[0].isdigit():
                start, end, state = line.split()[:3]
                start = (int(start)+1)/(MFCC_TIMESTEP * 10000) # htk
                end = (int(end)+1)/(MFCC_TIMESTEP * 10000) # htk
                for i in xrange(start, end):
                    tmp_len_x -= 1
                    y.append(state)
                
    assert(len(y) == x.shape[0])
    rootname = mlf[:-4] 
    np.save(rootname + '_xdata.npy', x)
    np.save(rootname + '_xgamma.npy', x_gamma)
    yy = np.array(y)
    np.save(rootname + '_ylabels.npy', yy)

    print "length x:", len(x), " length y:", len(y)
    print "shape x:", x.shape, "shape yy:", yy.shape 

    if TEST:
        tx = np.load(rootname + '_xdata.npy')
        tx_gamma = np.load(rootname + '_xgamma.npy')
        ty = np.load(rootname + '_ylabels.npy')
        if np.all(tx==x) and np.all(ty==yy):
            assert_allclose(tx_gamma, x_gamma, err_msg="x_gamma and its serialized version are not allclose")
            print "SUCCESS: serialized and current in-memory arrays are equal"
            sys.exit(0)
        else:
            print "ERROR: serialized and current in-memory arrays differ!"
            print "x (MFCC):", np.all(tx==x)
            print "y (labels):", np.all(ty==yy)
            sys.exit(-1)


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) < 2:
        print usage
        sys.exit(0)
    mlf = sys.argv[1]
    print "Producing a (x, y) dataset file for:", mlf
    print "WARNING: only the first 39 MFCC coefficients will be taken into account"
    extract_from_mlf(mlf)
