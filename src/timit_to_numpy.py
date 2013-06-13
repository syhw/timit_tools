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
N_MFCC_COEFFS = 39 # as in Mohamed et al. / Dahl et al. (Hinton group) papers

TEST = True # test numpy serialization

usage = """
    python timit_to_numpy.py MLF_FILENAME.mlf
output files are MLF_FILENAME_xdata.npy and MLF_FILENAME_ylabels.npy
    """


def extract_from_mlf(mlf):
    x = np.ndarray((0,39), dtype='float32')
    y = []
    
    with open(mlf) as f:
        tmp_len_x = 0 # verify sizes
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 1:
                continue
            if line[0] == '"': # TODO remove SA
                if tmp_len_x != 0:
                    print "the file above this one was mismatching x and y lengths", line
                t = htkmfc.open(line.strip('"')[:-3] + 'mfc') # .lab -> .mfc
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
    extract_from_mlf(mlf)
