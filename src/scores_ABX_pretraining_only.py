import numpy as np
import htkmfc
import sys, cPickle, functools, os
import scipy.io
sys.path.append(os.getcwd())
sys.path.append('DBN')

from batch_viterbi import compute_likelihoods_dbn
from batch_viterbi import padding

INSERTION_PENALTY = 2.5 # penalty of inserting a new phone (in the Viterbi)
SCALE_FACTOR = 1.0 # importance of the LM w.r.t. the acoustics
VERBOSE = True
epsilon = 1E-5 # degree of precision for floating (0.0-1.0 probas) operations
epsilon_log = 1E-30 # to add for logs
#APPEND_NAME = '_dbn.mat'
APPEND_NAME = '_pretraining.mat'
#APPEND_NAME = '_hmm_dbn.mat'
DEBUG = True # adds asserts...


class InnerLoop(object): # to circumvent pickling pbms w/ multiprocessing.map
    def __init__(self, start_end, depth_1_likelihoods=None, 
            depth_2_likelihoods=None, depth_3_likelihoods=None, depth_4_likelihoods=None):
        self.start_end = start_end
        self.depth_1_likelihoods = depth_1_likelihoods
        self.depth_2_likelihoods = depth_2_likelihoods
        self.depth_3_likelihoods = depth_3_likelihoods
        self.depth_4_likelihoods = depth_4_likelihoods
    def __call__(self, mfcc_file):
        print "doing", mfcc_file
        start, end = self.start_end[mfcc_file]
        if VERBOSE:
            print mfcc_file
            print start, end
        if DEBUG:
            assert(not (self.depth_1_likelihoods[start:end] == np.NaN).any())
            assert(not (self.depth_2_likelihoods[start:end] == np.NaN).any())
            assert(not (self.depth_3_likelihoods[start:end] == np.NaN).any())
            #assert(not (self.depth_4_likelihoods[start:end] == np.NaN).any())
        self.write_file(mfcc_file, start, end)
    def write_file(self, mfcc_file, start, end):
        print ">>> written", mfcc_file
        scipy.io.savemat(mfcc_file[:-4] + APPEND_NAME, mdict={
            'depth_1_likelihoods': self.depth_1_likelihoods[start:end],
            'depth_2_likelihoods': self.depth_2_likelihoods[start:end],
            'depth_3_likelihoods': self.depth_3_likelihoods[start:end]})
            #'depth_4_likelihoods': self.depth_4_likelihoods[start:end]})


if __name__ == "__main__":
    usage = "python scores_ABX.py directory input_dbn"
    if len(sys.argv) < 3:
        print usage
        sys.exit(-1)

    from DBN_Gaussian_timit import DBN # not Gaussian if no GRBM
    with open(sys.argv[2]) as idbnf:
        dbn = cPickle.load(idbnf)
    depth_1_computer = functools.partial(compute_likelihoods_dbn, dbn, depth=1)
    depth_2_computer = functools.partial(compute_likelihoods_dbn, dbn, depth=2)
    depth_3_computer = functools.partial(compute_likelihoods_dbn, dbn, depth=3)
    #depth_4_computer = functools.partial(compute_likelihoods_dbn, dbn, depth=4)

    list_of_mfcc_files = []
    for d, ds, fs in os.walk(sys.argv[1]):
        for fname in fs:
            if fname[-4:] != '.mfc':
                continue
            fullname = d.rstrip('/') + '/' + fname
            list_of_mfcc_files.append(fullname)

    input_n_frames = dbn.rbm_layers[0].n_visible / 39 # TODO generalize
    print "this is a DBN with", input_n_frames, "frames on the input layer"
    print "concatenating MFCC files" 
    all_mfcc = np.ndarray((0, dbn.rbm_layers[0].n_visible), dtype='float32')
    map_file_to_start_end = {}
    mfcc_file_name = 'tmp_allen_mfcc_' + str(int(input_n_frames)) + '.npy'
    map_mfcc_file_name = 'tmp_allen_map_file_to_start_end_' + str(int(input_n_frames)) + '.pickle'
    try:
        print "loading concat MFCC from pickled file"
        with open(mfcc_file_name) as concat_mfcc:
            all_mfcc = np.load(concat_mfcc)
        with open(map_mfcc_file_name) as map_mfcc:
            map_file_to_start_end = cPickle.load(map_mfcc)
    except:
        for ind, mfcc_file in enumerate(list_of_mfcc_files):
            start = all_mfcc.shape[0]
            x = htkmfc.open(mfcc_file).getall()
            if input_n_frames > 1:
                x = padding(input_n_frames, x)
            all_mfcc = np.append(all_mfcc, x, axis=0)
            map_file_to_start_end[mfcc_file] = (start, all_mfcc.shape[0])
            print "did", mfcc_file, "ind", ind
        with open(mfcc_file_name, 'w') as concat_mfcc:
            np.save(concat_mfcc, all_mfcc)
        with open(map_mfcc_file_name, 'w') as map_mfcc:
            cPickle.dump(map_file_to_start_end, map_mfcc)

    depth_1_likelihoods = depth_1_computer(all_mfcc)
    depth_2_likelihoods = depth_2_computer(all_mfcc)
    depth_3_likelihoods = depth_3_computer(all_mfcc) 
    #depth_4_likelihoods = depth_4_computer(all_mfcc) 
    print "computed all likelihoods"

    il = InnerLoop(map_file_to_start_end,
            depth_1_likelihoods=depth_1_likelihoods,
            depth_2_likelihoods=depth_2_likelihoods,
            depth_3_likelihoods=depth_3_likelihoods)
            #depth_4_likelihoods=depth_4_likelihoods)
    map(il, list_of_mfcc_files)

