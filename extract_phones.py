import os, sys
import numpy as np

"""
We try to reproduce the input of "Deep Belief Networks for phone recognition,
Mohamed et al., NIPS 2009" and other Hinton's group papers.
"""

SAMPLING_RATE = 16000 # Hz
MFCC_TIMESTEP = 10 # 10 ms
N_FRAMES = 11 # as in Mohamed et al. / Dahl et al. (Hinton group) papers
N_MFCC_COEFFS = 39 # as in Mohamed et al. / Dahl et al. (Hinton group) papers

usage = """
      python extract_phones.py [folder]
    if folder is ommited, will use .
    output files are [folder]_xdata.npy and [folder]_ylabels.npy
    """

def start_end(start_str, end_str, len_mfc):
    """ Centering of the MFCC that we will take on the middle of the phone
        "First we did a forced-alignment using HVite tool from HTK (with -a -f options as far as i remember) to produce state level alignment. Then we moved a window over the input and use the state label of the middle input frame as the label of the whole window." Abdel-rahman Mohamed """
    start = 1000 * float(start_str) / SAMPLING_RATE # now in milli seconds
    end = 1000 * float(end_str) / SAMPLING_RATE # now in milli seconds
    duration = end - start
    middle = start + duration/2
    start = int(middle) - N_FRAMES * MFCC_TIMESTEP / 2
    end = int(middle) + N_FRAMES * MFCC_TIMESTEP / 2
    assert(11 <= (end-start) / MFCC_TIMESTEP < 12)
    start = max(0, int(start/MFCC_TIMESTEP)) # now in mfcc indice
    end = min(len_mfc + 1, int(end/MFCC_TIMESTEP)) # now in mfcc indice
    return (start, end)


def pad(a):
    """ Pad for silence, overflow in-between phones is already dealt with """
    if len(a) < N_MFCC_COEFFS * N_FRAMES:
        # we transform [1,2,3] in [0,...x...,0,1,2,3,0,...y...,0] with x <= y
        ret = np.zeros(N_MFCC_COEFFS * N_FRAMES)
        pad_size = (N_MFCC_COEFFS - len(a))/2 
        ret[pad_size:pad_size+len(a)] = a
        return ret
    return a


def extract(folder):
    x = []
    y = []

    name = folder.rstrip('/').strip('/')
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-8:] != '_mfc.npy':
                continue
            mfcfname = d+'/'+fname
            tmp_mfc = np.fromfile(mfcfname)
            phonefname = mfcfname[:-8]+'.phn'
            phonef = open(phonefname)
            for line in phonef:
                start_s, end_s, phone = line.rstrip('\n').split()
                start, end = start_end(start_s, end_s, len(tmp_mfc))
                x.append(pad(tmp_mfc[start:end].flatten()))
                y.append(phone)
            phonef.close()
            print "Dealt with:", fname

    xx = np.array(x)
    yy = np.array(y)
    print "length x:", len(x), " length y:", len(y)
    print "shape xx:", xx.shape, "shape yy:", yy.shape 

    np.save(name+'_xdata', xx)
    np.save(name+'_ylabels', yy)


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h':
            print usage
            sys.exit(0)
        folder = sys.argv[1]
    print "Producing a (x, y) dataset file for folder:", folder
    print "WARNING: only the first 39 MFCC coefficients will be taken into account"
    extract(folder)
