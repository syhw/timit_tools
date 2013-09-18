import numpy as np
import htkmfc
import sys, cPickle, functools, os
from multiprocessing import Pool, cpu_count
import scipy.io
sys.path.append(os.getcwd())
sys.path.append('DBN')

from batch_mocha_viterbi import precompute_det_inv, phones_mapping, parse_hmm
from batch_mocha_viterbi import compute_likelihoods, compute_likelihoods_dbn
from batch_mocha_viterbi import Phone, viterbi, initialize_transitions
from batch_mocha_viterbi import penalty_scale, padding

INSERTION_PENALTY = 2.5 # penalty of inserting a new phone (in the Viterbi)
SCALE_FACTOR = 1.0 # importance of the LM w.r.t. the acoustics
VERBOSE = True
epsilon = 1E-5 # degree of precision for floating (0.0-1.0 probas) operations
epsilon_log = 1E-30 # to add for logs
APPEND_NAME = '_dbn_mocha.mat'
from batch_mocha_viterbi import N_BATCHES_DATASET

usage = "python scores_ABX.py directory input_hmm [input_dbn dbn_dict]"

class InnerLoop(object): # to circumvent pickling pbms w/ multiprocessing.map
    def __init__(self, likelihoods, map_states_to_phones, transitions,
            using_bigram=False):
        self.likelihoods = likelihoods
        self.map_states_to_phones = map_states_to_phones
        self.transitions = transitions
        self.using_bigram = using_bigram
    def __call__(self, mfcc_file):
        start, end = self.likelihoods[1][mfcc_file]
        if VERBOSE:
            print mfcc_file
            print start, end
        _, posteriorgrams = viterbi(self.likelihoods[0][start:end],
                                   self.transitions, 
                                   self.map_states_to_phones,
                                   using_bigram=self.using_bigram)
        self.write_file(mfcc_file, start, end, posteriorgrams)
    def write_file(self, mfcc_file, start, end, posteriorgrams):
        print "written", mfcc_file
        scipy.io.savemat(mfcc_file[:-4] + APPEND_NAME, mdict={
            'likelihoods': self.likelihoods[0][start:end],
            'posteriors': posteriorgrams})


def reconstruct_articulatory_features_likelihoods(dbn, mat, normalize=True, 
                                                  unit=False,
                                                  pca_whiten_mfcc=False,
                                                  pca_whiten_arti=False):
    n_mfcc = 39
    if pca_whiten_mfcc:
        ### PCA whitening, beware it's sklearn's and thus stays in PCA space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_whiten_mfcc, whiten=True)
        pca.fit(mat[:, :n_mfcc])
        n_mfcc = pca.n_components
        # and thus here we still never saw test data
        mat = np.concatenate([pca.transform(mat[:, :n_mfcc]),
                                 mat[:, n_mfcc:]], axis=1)
    n_arti = 60
    if pca_whiten_arti:
        ### PCA whitening, beware it's sklearn's and thus stays in PCA space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_whiten_arti, whiten=True)
        pca.fit(mat[:, n_mfcc:])
        n_arti = pca.n_components
        # and thus here we still never saw test data
        mat = np.concatenate([mat[:, :n_mfcc],
                                 pca.transform(mat[:, n_mfcc:])], axis=1)
    if normalize:
        # if the first layer of the DBN is a Gaussian RBM, we need to normalize mat
        mat = (mat - np.mean(mat, 0)) / np.std(mat, 0)
    elif unit:
        # if the first layer of the DBN is a binary RBM, send mat in [0-1] range
        mat = (mat - np.min(mat, 0)) / np.max(mat, 0)
    import theano.tensor as T
    ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype="float32")
    from theano import shared#, scan
    batch_size = mat.shape[0] / N_BATCHES_DATASET
    out_ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype="float32")
    for ind in xrange(0, mat.shape[0]+1, batch_size):
        output = shared(mat[ind:ind+batch_size])
        [pre, out_mfcc] = dbn.rbm_layers[0].propup(output[:, :dbn.rbm_layers[0].n_visible])
        # TODO use (in and out) samples instead of means
        [_, _, _, pre, input, in_sample] = dbn.rbm_layers[2].gibbs_vhv(T.concatenate([out_mfcc, np.zeros(out_mfcc.eval().shape, dtype='float32')], axis=1)) # out_mfcc.shape==out_arti.shape, so we use it by proxy
        #zeroing out the articulatory features, that's not comparable to training MFCC only
        [_, _, _, pre, out_arti, out_sample] = dbn.rbm_layers[1].gibbs_hvh(input[:, 960:])
        [pre, output] = dbn.rbm_layers[2].propup(T.concatenate([out_mfcc, out_arti], axis=1))
        for layer_ind in xrange(3, dbn.n_layers):
            [pre, output] = dbn.rbm_layers[layer_ind].propup(output)
        ret = T.nnet.softmax(T.dot(output, dbn.logLayer.W) + dbn.logLayer.b)
        out_ret[ind:ind+batch_size] = T.log(ret).eval()
    return out_ret


if len(sys.argv) != 3 and len(sys.argv) != 5:
    print usage
    sys.exit(-1)

with open(sys.argv[2]) as ihmmf:
    n_states, transitions, gmms = parse_hmm(ihmmf)

gmms_ = precompute_det_inv(gmms)
map_states_to_phones = phones_mapping(gmms)
likelihoods_computer = functools.partial(compute_likelihoods, gmms_)

dbn = None
if len(sys.argv) == 5:
    from DBN_Gaussian_mocha_timit import DBN
    with open(sys.argv[3]) as idbnf:
        dbn = cPickle.load(idbnf)
    with open(sys.argv[4]) as idbndtf:
        dbn_to_int_to_state_tuple = cPickle.load(idbndtf)
    dbn_phones_to_states = dbn_to_int_to_state_tuple[0]
    likelihoods_computer = functools.partial(reconstruct_articulatory_features_likelihoods, dbn)

# TODO bigrams
transitions = initialize_transitions(transitions)
#print transitions
transitions = penalty_scale(transitions, insertion_penalty=INSERTION_PENALTY,
        scale_factor=SCALE_FACTOR)

dummy = np.ndarray((2,2)) # to force only 1 compile of Viterbi's C
viterbi(dummy, [None, dummy], {}) # also for this compile's debug purposes

list_of_mfcc_files = []
for d, ds, fs in os.walk(sys.argv[1]):
    for fname in fs:
        if fname[-4:] != '.mfc':
            continue
        fullname = d.rstrip('/') + '/' + fname
        list_of_mfcc_files.append(fullname)
#print list_of_mfcc_files

if dbn != None:
    input_n_frames_mfcc = dbn.rbm_layers[0].n_visible / 39 # TODO generalize
    input_n_frames_arti = dbn.rbm_layers[1].n_visible / 59 # 60 # TODO generalize
    print "this is a DBN with", input_n_frames_mfcc, "MFCC frames on the input layer"
    print "and", input_n_frames_arti, "articulatory frames on the other input layer"
    print "concatenating MFCC and articulatory files" 
    all_mfcc = np.ndarray((0, dbn.rbm_layers[0].n_visible), dtype='float32')
    map_file_to_start_end = {}
    mfcc_file_name = 'tmp_allen_mocha_.npy'
    map_mfcc_file_name = 'tmp_allen_mocha_map_file_to_start_end_.pickle'
    try:
        print "loading concat MFCC and articulatory from pickled file"
        with open(mfcc_file_name) as concat:
            all_mfcc = np.load(concat)
        with open(map_mfcc_file_name) as map_mfcc:
            map_file_to_start_end = cPickle.load(map_mfcc)
    except:
        for ind, mfcc_file in enumerate(list_of_mfcc_files):
            start = all_mfcc.shape[0]
            x = htkmfc.open(mfcc_file).getall()
            if input_n_frames_mfcc > 1:
                x = padding(input_n_frames_mfcc, x)
            all_mfcc = np.append(all_mfcc, x, axis=0)
            map_file_to_start_end[mfcc_file] = (start, all_mfcc.shape[0])
            print "did", mfcc_file, "ind", ind
        with open(mfcc_file_name, 'w') as concat:
            np.save(concat, all_mfcc)
        with open(map_mfcc_file_name, 'w') as map_mfcc:
            cPickle.dump(map_file_to_start_end, map_mfcc)

    tmp_likelihoods = likelihoods_computer(all_mfcc)
    columns_remapping = [dbn_phones_to_states[map_states_to_phones[i]] for i in xrange(tmp_likelihoods.shape[1])]
    likelihoods = (tmp_likelihoods[:, columns_remapping],
        map_file_to_start_end)
else:
    all_mfcc = np.ndarray((0, 39), dtype='float32')
    map_file_to_start_end = {}
    mfcc_file_name = 'tmp_allen_mfcc_.npy'
    map_mfcc_file_name = 'tmp_allen_map_file_to_start_end_.pickle'
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
            all_mfcc = np.append(all_mfcc, x, axis=0)
            map_file_to_start_end[mfcc_file] = (start, all_mfcc.shape[0])
            print "did", mfcc_file, "ind", ind
        with open(mfcc_file_name, 'w') as concat_mfcc:
            np.save(concat_mfcc, all_mfcc)
        with open(map_mfcc_file_name, 'w') as map_mfcc:
            cPickle.dump(map_file_to_start_end, map_mfcc)
    likelihoods = (likelihoods_computer(all_mfcc), map_file_to_start_end)

il = InnerLoop(likelihoods, map_states_to_phones, transitions)
p = Pool(cpu_count())
p.map(il, list_of_mfcc_files)

