import numpy as np
import functools
import sys 
import cPickle
import htkmfc
from multiprocessing import Pool, cpu_count
import os
sys.path.append(os.getcwd())

from mocha_timit_to_numpy import from_mfcc_ema_to_mfcc_arti_tuple
from batch_viterbi import InnerLoop, viterbi, initialize_transitions, usage
from batch_viterbi import compute_likelihoods, clean, phones_mapping
from batch_viterbi import penalty_scale, padding, precompute_det_inv
from batch_viterbi import parse_lm, parse_wdnet, parse_lm_matrix, parse_hmm


VERBOSE = False
UNIGRAMS_ONLY = False # says if we use only unigrams when we have _our_ bigrams
MATRIX_BIGRAM = True # is the bigram file format a matrix? (ARPA-MIT if False)
THRESHOLD_BIGRAMS = -10.0 # log10 min proba for a bigram to not be backed-off
SCALE_FACTOR = 1.0 # importance of the LM w.r.t. the acoustics
INSERTION_PENALTY = 2.5 # penalty of inserting a new phone (in the Viterbi)
epsilon = 1E-5 # degree of precision for floating (0.0-1.0 probas) operations
epsilon_log = 1E-80 # to add for logs
N_BATCHES_DATASET = 4 # number of batches in which we divide the dataset 
                      # (to fit in the GPU memory, only 2Gb at home)


def compute_likelihoods_dbn(dbn, mat, normalize=True, unit=False):
    """ compute the log-likelihoods of each states i according to the Deep 
    Belief Network (stacked RBMs) in dbn, for each line of mat (input data) """
    # first normalize or put in the unit ([0-1]) interval
    # TODO do that only if we did not do that at the full scale of the corpus
    if normalize:
        # if the first layer of the DBN is a Gaussian RBM, we need to normalize mat
        mat = (mat - np.mean(mat, 0)) / np.std(mat, 0)
    elif unit:
        # if the first layer of the DBN is a binary RBM, send mat in [0-1] range
        mat = (mat - np.min(mat, 0)) / np.max(mat, 0)

    import theano.tensor as T
    ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype="float32")
    from theano import shared#, scan
    # propagating through the deep belief net
    batch_size = mat.shape[0] / N_BATCHES_DATASET
    out_ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype="float32")
    for ind in xrange(0, mat.shape[0]+1, batch_size):
        output = shared(mat[ind:ind+batch_size])
        [pre, out_mfcc] = dbn.rbm_layers[0].propup(output[:, :dbn.rbm_layers[0].n_visible])
        [pre, out_arti] = dbn.rbm_layers[1].propup(output[:, dbn.rbm_layers[0].n_visible:])
        [pre, output] = dbn.rbm_layers[2].propup(T.concatenate([out_mfcc, out_arti], axis=1))
        #[pre, output] = dbn.rbm_layers[2].propup(T.concatenate([out_mfcc, np.zeros(out_arti.eval().shape, dtype='float32')], axis=1)) # zeroing out the articulatory features, just to try, that's not comparable to training MFCC onle
        for layer_ind in xrange(3, dbn.n_layers):
            [pre, output] = dbn.rbm_layers[layer_ind].propup(output)
        ret = T.nnet.softmax(T.dot(output, dbn.logLayer.W) + dbn.logLayer.b)
        out_ret[ind:ind+batch_size] = T.log(ret).eval()
    return out_ret


def process(ofname, iscpfname, ihmmfname, 
        ilmfname=None, iwdnetfname=None, unibifname=None, 
        idbnfname=None, idbndictstuple=None):

    with open(ihmmfname) as ihmmf:
        n_states, transitions, gmms = parse_hmm(ihmmf)

    gmms_ = precompute_det_inv(gmms)
    map_states_to_phones = phones_mapping(gmms)
    likelihoods_computer = functools.partial(compute_likelihoods, gmms_)
    gmm_likelihoods_computer = functools.partial(compute_likelihoods, gmms_) #TODO REMOVE

    dbn = None
    dbn_to_int_to_state_tuple = None
    if idbnfname != None:
        with open(idbnfname) as idbnf:
            dbn = cPickle.load(idbnf)
        with open(idbndictstuple) as idbndtf:
            dbn_to_int_to_state_tuple = cPickle.load(idbndtf)
        dbn_phones_to_states = dbn_to_int_to_state_tuple[0]
        likelihoods_computer = functools.partial(compute_likelihoods_dbn, dbn)
        # like that = for GRBM first layer (normalize=True, unit=False)
        # TODO correct the normalize/unit to work on full test dataset

    if iwdnetfname != None:
        with open(iwdnetfname) as iwdnf:
            transitions = parse_wdnet(transitions, iwdnf) # parse wordnet
    elif ilmfname != None:
        with open(ilmfname) as ilmf:
            if MATRIX_BIGRAM:
                transitions = parse_lm_matrix(transitions, ilmf) # parse bigram LM in matrix format in ilmf
            else:
                transitions = parse_lm(transitions, ilmf) # parse bigram LM in ARPA-MIT in ilmf
    elif unibifname != None: # our own unigram and bigram counts,
                             # c.f. src/produce_LM.py
        with open(unibifname) as ubf:
            transitions = initialize_transitions(transitions, ubf, 
                    unigrams_only=UNIGRAMS_ONLY)
    else:
        # uniform transitions between phones
        transitions = initialize_transitions(transitions)
    transitions = penalty_scale(transitions, 
            insertion_penalty=INSERTION_PENALTY, scale_factor=SCALE_FACTOR)


    dummy = np.ndarray((2,2)) # to force only 1 compile of Viterbi's C
    viterbi(dummy, [None, dummy], {}) # also for this compile's debug purposes
    
    if dbn != None:
        input_n_frames_mfcc = dbn.rbm_layers[0].n_visible / 39 # TODO generalize
        print "this is a DBN with", input_n_frames_mfcc, "MFCC frames"
        input_n_frames_arti = dbn.rbm_layers[1].n_visible / 59 # 60 # TODO generalize
        print "this is a DBN with", input_n_frames_arti, "articulatory frames"
        input_file_name = 'tmp_input_mocha.npy'
        map_input_file_name = 'tmp_map_file_to_start_end_mocha.pickle'
        try: # TODO remove?
            print "loading concat MFCC from pickled file"
            with open(input_file_name) as concat:
                all_input = np.load(concat)
            with open(map_input_file_name) as map_input:
                map_file_to_start_end = cPickle.load(map_input)
        except:
            print "concatenating MFCC and articulatory files" # TODO parallelize + use np.concatenate
            all_input = np.ndarray((0, dbn.rbm_layers[0].n_visible + dbn.rbm_layers[1].n_visible), dtype='float32')
            map_file_to_start_end = {}
            with open(iscpfname) as iscpf:
                for line in iscpf:
                    cline = clean(line)
                    start = all_input.shape[0]
                    # get the 1 framed signals
                    x_mfcc = htkmfc.open(cline).getall()
                    with open(cline[:-4] + '_ema.npy') as ema:
                        x_arti = np.load(ema)[:, 2:]
                    # compute deltas and deltas deltas for articulatory features
                    _, x_arti = from_mfcc_ema_to_mfcc_arti_tuple(x_mfcc, x_arti)
                    # add the adjacent frames
                    if input_n_frames_mfcc > 1:
                        x_mfcc = padding(input_n_frames_mfcc, x_mfcc)
                    if input_n_frames_arti > 1:
                        x_arti = padding(input_n_frames_arti, x_arti)
                    # do feature transformations if any
                    # TODO with mocha_timit_params.json params
                    # concatenate
                    x_mfcc_arti = np.concatenate((x_mfcc, x_arti), axis=1)
                    all_input = np.append(all_input, x_mfcc_arti, axis=0)
                    map_file_to_start_end[cline] = (start, all_input.shape[0])
            with open(input_file_name, 'w') as concat:
                np.save(concat, all_input)
            with open(map_input_file_name, 'w') as map_input:
                cPickle.dump(map_file_to_start_end, map_input)
    else: # GMM
        all_mfcc = np.ndarray((0, 39), dtype='float32') # TODO generalize

    print "computing likelihoods"
    if dbn != None: # TODO clean
        tmp_likelihoods = likelihoods_computer(all_input)
        #mean_dbns = np.mean(tmp_likelihoods, 0)
        #tmp_likelihoods *= (mean_gmms / mean_dbns)
        print tmp_likelihoods
        print tmp_likelihoods.shape
        columns_remapping = [dbn_phones_to_states[map_states_to_phones[i]] for i in xrange(tmp_likelihoods.shape[1])]
        print columns_remapping
        likelihoods = (tmp_likelihoods[:, columns_remapping],
            map_file_to_start_end)
        print likelihoods[0]
        print likelihoods[0].shape
    else:
        likelihoods = (likelihoods_computer(all_mfcc), map_file_to_start_end)

    print "computing viterbi paths"
    list_mlf_string = []
    with open(iscpfname) as iscpf:
        il = InnerLoop(likelihoods,
                map_states_to_phones, transitions,
                using_bigram=(ilmfname != None 
                    or iwdnetfname != None 
                    or unibifname != None))
        p = Pool(cpu_count())
        list_mlf_string = p.map(il, iscpf)
    with open(ofname, 'w') as of:
        of.write('#!MLF!#\n')
        for line in list_mlf_string:
            of.write(line)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        if '--help' in sys.argv:
            print usage
            sys.exit(0)
        args = dict(enumerate(sys.argv))
        options = filter(lambda (ind, x): '--' in x[0:2], enumerate(sys.argv))
        input_unibi_fname = None # my bigram LM
        input_lm_fname = None # HStats bigram LMs (either matrix of ARPA-MIT)
        input_wdnet_fname = None # HTK's wdnet (with bigram probas)
        dbn_fname = None # DBN cPickle
        dbn_dicts_fname = None # DBN to_int and to_states dicts tuple
        if len(options): # we have options
            for ind, option in options:
                args.pop(ind)
                if option == '--verbose':
                    VERBOSE = True
                if option == '--p':
                    INSERTION_PENALTY = float(args[ind+1])
                    args.pop(ind+1)
                if option == '--s':
                    SCALE_FACTOR = float(args[ind+1])
                    args.pop(ind+1)
                if option == '--ub':
                    input_unibi_fname = args[ind+1]
                    args.pop(ind+1)
                    print "initialize the transitions between phones with the discounted bigram lm", input_unibi_fname 
                if option == '--b':
                    input_lm_fname = args[ind+1]
                    args.pop(ind+1)
                    print "initialize the transitions between phones with the bigram lm", input_lm_fname
                if option == '--w':
                    input_wdnet_fname = args[ind+1]
                    args.pop(ind+1)
                    print "initialize the transitions between phones with the wordnet", input_wdnet_fname
                    print "WILL IGNORE LANGUAGE MODELS!"
                if option == '--d':
                    if not (ind+2) in args:
                        print >> sys.stderr, "We need the DBN and the states/phones mapping"
                        print >> sys.stderr, usage
                        sys.exit(-1)
                    try:
                        from DBN_Gaussian_timit import DBN # not Gaussian if no GRBM
                    except:
                        print >> sys.stderr, "experimental: TO BE LAUNCHED FROM THE 'DBN/' DIR"
                        sys.exit(-1)
                    dbn_fname = args[ind+1]
                    args.pop(ind+1)
                    print "will use the following DBN to estimate states likelihoods", dbn_fname
                    dbn_dicts_fname = args[ind+2]
                    args.pop(ind+2)
                    print "and the following to_int / to_state dicts tuple", dbn_dicts_fname
        else:
            print "initialize the transitions between phones uniformly"
        output_fname = args.values()[1]
        input_scp_fname = args.values()[2]
        input_hmm_fname = args.values()[3]
        process(output_fname, input_scp_fname, 
                input_hmm_fname, input_lm_fname, 
                input_wdnet_fname, input_unibi_fname,
                dbn_fname, dbn_dicts_fname)
    else:
        print usage
        sys.exit(-1)
