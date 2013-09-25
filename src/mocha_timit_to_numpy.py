import sys
import numpy as np
import htkmfc

"""
We try to reproduce the input of "Deep Belief Networks for phone recognition,
Mohamed et al., NIPS 2009" and other Hinton's group papers.
"""

# TODO load the following parameters from wav_config
SAMPLING_RATE = 16000 # Hz
MFCC_TIMESTEP = 10 # 10 ms
N_MFCC_COEFFS = 39 
N_EMA_COEFFS = 20 * 3 # number of articulatory coordinates

TEST = True # test numpy serialization

usage = """
    python timit_to_numpy.py MLF_FILENAME.mlf
output files are MLF_FILENAME_xdata.npy and MLF_FILENAME_ylabels.npy
    """

def from_mfcc_ema_to_mfcc_arti_tuple(x_mfc, x_ema):
    """ Takes MFCC and EMA data and output concatenated MFCC 
    and position/speed/acceleration of the articulators """
    # THE EMA FILE STARTS AFTER "!ENTER" and "breath" # TODO check again
    tmp_ema = np.pad(x_ema, 
            ((x_mfc.shape[0] - x_ema.shape[0], 0), (0, 0)), 
            'constant', constant_values=(0.0, 0.0)) # should we pad with 0? TODO
    tmp_diff = np.pad(np.diff(tmp_ema, axis=0),
            ((0, 1), (0, 0)),
            'constant', constant_values=(0.0, 0.0))
    tmp_accel = np.pad(np.diff(tmp_diff, axis=0),
            ((0, 1), (0, 0)),
            'constant', constant_values=(0.0, 0.0))
    return x_mfc, np.concatenate((tmp_ema, tmp_diff, tmp_accel), axis=1)

def extract_from_mlf(mlf):
    x = np.ndarray((0, N_MFCC_COEFFS + N_EMA_COEFFS), dtype='float32')
    y = []
    
    with open(mlf) as f:
        tmp_len_x = 0 # verify sizes
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 1:
                continue
            if line[0] == '"': 
                if tmp_len_x != 0:
                    print "the file above this one was mismatching x and y lengths", line
                t = htkmfc.open(line.strip('"')[:-3] + 'mfc') # .lab -> .mfc
                mfc_file = t.getall()
                with open(line.strip('"')[:-4] + '_ema.npy') as ema_f: # .lab -> _ema.npy
                    ema_file = np.load(ema_f)[:,2:]
                x_file = np.concatenate(from_mfcc_ema_to_mfcc_arti_tuple(
                    mfc_file, ema_file), axis=1)
                x = np.append(x, x_file, axis=0)
                tmp_len_x = mfc_file.shape[0]
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
    yy = np.array(y)
    np.save(rootname + '_ylabels.npy', yy)

    print "length x:", len(x), " length y:", len(y)
    print "shape x:", x.shape, "shape yy:", yy.shape 

    if TEST:
        tx = np.load(rootname + '_xdata.npy')
        ty = np.load(rootname + '_ylabels.npy')
        if np.all(tx==x) and np.all(ty==yy):
            print "SUCCESS: serialized and current in-memory arrays are equal"
            sys.exit(0)
        else:
            print "ERROR: serialized and current in-memory arrays differ!"
            sys.exit(-1)


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) < 2:
        print usage
        sys.exit(0)
    mlf = sys.argv[1]
    print "Producing a (x, y) dataset file for:", mlf
    print "WARNING: only the first 39 MFCC coefficients will be taken into account"
    print "WARNING: and then 20 EMA coeffs and their speed and acceleration"
    extract_from_mlf(mlf)
