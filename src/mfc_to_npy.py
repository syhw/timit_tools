import os, sys
import numpy as np
import htkmfc

SAMPLING_RATE = 16000 # Hz
MFCC_TIMESTEP = 10 # 10 ms
N_MFCC_COEFFS = 39
NFRAMES_DBN = 11

def padding_before(nframes, x):
    pad_size = (nframes - 1) / 2
    return np.pad(x, ((pad_size, 0), (0, 0)),
            'constant', constant_values=(0.0, 0.0))


def padding_after(nframes, x):
    pad_size = (nframes - 1) / 2
    return np.pad(x, ((0, pad_size), (0, 0)),
            'constant', constant_values=(0.0, 0.0))


def padding(nframes, x):
    ba = (nframes - 1) / 2 # before // after
    x_f = np.zeros((x.shape[0], nframes * x.shape[1]), dtype='float32')
    print x.shape
    print x_f.shape
    for i in xrange(x.shape[0]):
        x_f[i] = np.pad(x[max(0, i - ba):i + ba + 1].flatten(),
                (max(0, (ba - i) * x.shape[1]), 
                    max(0, ((i+ba+1) - x.shape[0]) * x.shape[1])),
                'constant', constant_values=(0,0))
    return x_f


def concat_all(folder):
    l = []
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-4:] != '.mfc':
                continue
            fullfname = d + '/' + fname
            print fullfname
            t = htkmfc.open(fullfname)
            l.append(t.getall())
    stats = np.concatenate(l)
    mean = np.mean(stats, 0)
    stddev = np.std(stats, 0)
    for i, e in enumerate(l):
        l[i] = padding(NFRAMES_DBN, (e - mean) / stddev)
    a = np.concatenate(l) 
    np.save(folder + '/' + 'x_all_mfcc.npy', a)


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    print "Concatenating all the *.mfc into a big *.npy file", folder
    concat_all(folder.rstrip('/'))

