import numpy as np
from numpy import linalg
import functools
import sys, math
#import cPickle
from collections import defaultdict, deque
import htkmfc
import itertools
#from utils import memoized
#import multiprocessing
#pool = multiprocessing.Pool()

usage = """
python viterbi.py OUTPUT[.mlf] INPUT_SCP INPUT_HMM [INPUT_LM] [options: --help]
"""

THRESHOLD_BIGRAMS = -10.0 # log10 min proba for a bigram to not be backed-off
epsilon = 1E-10 # degree of precision for floating (0.0-1.0 probas) operations

class Phone:
    def __init__(self, phn_id, phn):
        self.phn_id = phn_id
        self.phn = phn
        self.to_ind = []

    def update(self, indice):
        self.to_ind.append(indice)

    def __repr__(self):
        return self.phn + ": " + str(self.phn_id) + '\n' + str(self.to_ind)


def clean(s):
    return s.strip().rstrip('\n')


def eval_gauss_mixt(v, gmixt):
    """ UNTESTED """ # TODO re-test since change
    assert(len(gmixt[0]) == gmixt[1].shape[0] == gmixt[2].shape[0])
    def eval_gauss_comp(mix_comp): # closure
        pi_k, mu_k, sigma2_k_inv = mix_comp
        return pi_k * math.exp(-0.5 * np.dot((v - mu_k).T, 
                    np.dot(sigma2_k_inv, v - mu_k)))
    return reduce(lambda x, y: x + y, map(eval_gauss_comp, 
        itertools.izip(gmixt[0], gmixt[1], gmixt[2])))


def precompute_det_inv(gmms):
    # /!\ iteration order is important, this gives us:
    ret = []
    for _, gm in gmms.iteritems():
        for gm_st in gm:
            pi_k = []
            mu_k = []
            inv_sqrt_det_sigma2 = []
            inv_sigma2 = []
            for component in gm_st:
                pi_k.append(component[0])
                mu_k.append(component[1])
                sigma2_k = component[2]
                inv_sqrt_det_sigma2.append((2 * np.pi * linalg.det(np.diag(sigma2_k))) ** (-0.5))
                inv_sigma2.append(linalg.inv(np.diag(sigma2_k)))
            ret.append((np.array(pi_k) * np.array(inv_sqrt_det_sigma2), 
                    np.array(mu_k).T, 
                    np.array(inv_sigma2).T))
    return ret


#@profile
def compute_likelihoods(n_states, mat, gmms_):
    # HUGE TODO: optimize that whole function
    #import scipy.io
    #scipy.io.savemat('sa1.mat', mdict={'arr': mat})
    #scipy.io.savemat('gmms.mat', mdict={'arr': gmms_})
    ret = np.ndarray((mat.shape[0], n_states))
    ret[:] = 0.0
    for state_id, mixture in enumerate(gmms_):
        pis, mus, inv_sigmas = mixture
        # N_mixtures = len(pis) = mus.shape[1] = inv_sigmas.shape[2]
        # N_features = mus.shape[0] = inv_sigmas.shape[0|1]
        assert(pis.shape[0] == mus.shape[1])
        assert(pis.shape[0] == inv_sigmas.shape[2])
        x_minus_mus = np.ndarray((mat.shape[0], mus.shape[0], mus.shape[1]))
        x_minus_mus.T[:,] = mat.T
        x_minus_mus -= mus
        #components = np.einsum('ik...,...km->i...', x_minus_mus[:,:,0], 
        #        np.einsum('ik...,jk...', inv_sigmas, x_minus_mus))
        tmp = np.einsum('ik...,jk...', inv_sigmas, x_minus_mus)
        components = np.einsum('ik...,...km->i...', x_minus_mus[:,:,0], 
                tmp)
        import code
        code.interact(local=locals())
        ret[:, state_id] = np.dot(components, pis)
    print ret
    print ret[0]
    return ret


def viterbi(posteriors, transitions):
    # TODO
    pass


def phones_mapping(gmms):
    map_states_to_phones = {}
    i = 0
    for phn, gm in gmms.iteritems():
        st_id = 2
        for gm_st in gm:
            map_states_to_phones[i] = phn + "[" + str(st_id) + "]"
            i += 1
            st_id += 1
    return map_states_to_phones


def string_mlf(map_states_to_phones, input_lab, states):
    s = []
    for i, line in enumerate(open(input_lab)): # TODO change/remove input_lab
        s.append(' '.join(line.split()[:-1] + [map_states_to_phones[states[i][0]], str(states[i][1])])) # TODO correct timings with forced alignment
    s.append('')
    return '\n'.join(s)


def online_viterbi(n_states, mat, gmms_, transitions):
    # transform to from e.g. (sigma2_invs) 39*39*17 to 17*39*39
    g = map(lambda (pis, mus, sigma2_invs): (pis, mus.T, sigma2_invs.T), gmms_) 
    t = np.ndarray((mat.shape[0], n_states))
    t[:] = -100000.0 # log
    t[0] = map(math.log, map(functools.partial(eval_gauss_mixt, mat[0]), g)) # log
    backpointers = np.ndarray((mat.shape[0]-1, n_states), dtype=int)
    backpointers[:] = -1
    nonnulls = [jj for jj, val in enumerate(t[0]) if val > -100000.0] # log
    for i in xrange(1, mat.shape[0]):
        for j in xrange(n_states):
            max_ = -100000.0 # log
            max_ind = -2
            for k in nonnulls:
                if transitions[1][k][j] == 0.0:
                    continue
                tmp_prob = (t[i-1][k] + math.log(transitions[1][k][j]) # log
                        + math.log(eval_gauss_mixt(mat[i], g[j])))     # log
                if tmp_prob > max_:
                    max_ = tmp_prob
                    max_ind = k
            t[i][j] = max_ # log
            backpointers[i-1][j] = max_ind
        nonnulls = [jj for jj, val in enumerate(t[i]) if val > -100000.0] # log
        if len(nonnulls) == 0:
            print >> sys.stderr, ">>>>>>>>> NONNULLS IS EMPTY", i, mat.shape[0]
    states = deque([(t[mat.shape[0]-1].argmax(), t[mat.shape[0]-1].max())])
    for i in xrange(mat.shape[0] - 2, -1, -1):
        states.appendleft((backpointers[i][states[0][0]], t[i][backpointers[i][states[0][0]]]))
        #states.appendleft((t[i].argmax(), t[i].max()))
    return states
    #return t, backpointers


def parse_lm(trans, f):
    """ parse ARPA MIT-LL backed-off bigrams in f """
    p_1grams = {}
    b_1grams = {}
    p_2grams = defaultdict(lambda: {}) # p_2grams[A][B] = unnormalized P(A|B)
    # parse the file to fill the above dicts
    parsing1grams = False
    parsing2grams = False
    for line in f:
        if clean(line) == "":
            continue
        if "1-grams" in line:
            parsing1grams = True
        elif "2-grams" in line:
            parsing1grams = False
            parsing2grams = True
        elif "end" == line[1:4]:
            break
        elif parsing1grams: 
            l = clean(line).split()
            p_1grams[l[1]] = float(l[0]) # log10 prob
            if len(l) > 2:
                b_1grams[l[1]] = float(l[2]) # log10 prob
            else:
                b_1grams[l[1]] = -100.0 # guess that's low enough
        elif parsing2grams:
            l = clean(line).split()
            if len(l) != 3:
                print >> sys.stderr, "bad language model file format"
                sys.exit(-1)
            p_2grams[l[1]][l[2]] = float(l[0]) # log10 prob, already discounted
    # do the backed-off probs for p_2grams[phn1][phn2] = P(phn2|phn1)
    for phn1, d in p_2grams.iteritems():
        s = 0.0
        for phn2, log_prob in d.iteritems():
            # j follows i, p(j)*b(i)
            if log_prob < p_1grams[phn2] + b_1grams[phn1] \
                    or log_prob < THRESHOLD_BIGRAMS:
                p_2grams[phn1][phn2] = p_1grams[phn2] + b_1grams[phn1]
            s += 10 ** p_2grams[phn1][phn2]
        s = math.log10(s)
        for phn2, log_prob in d.iteritems():
            p_2grams[phn1][phn2] = log_prob - s
    # edit the trans[1] matrix with the backed-off probs,
    # could do in the above "backed-off probs" loop 
    # I but prefer to keep it separated
    for phn1, d in p_2grams.iteritems():
        phone1 = trans[0][phn1]
        buffer_prob = 1.0 - trans[1][phone1.to_ind[len(phone1.to_ind) - 1]].sum(0)
        assert(buffer_prob != 0.0) # you would never go out of this phone (/!\ !EXIT)
        for phn2, log_prob in d.iteritems():
            # transition from phn1 to phn2
            phone2 = trans[0][phn2]
            trans[1][phone1.to_ind[len(phone1.to_ind) - 1]][phone2.to_ind[0]] = buffer_prob * (10 ** log_prob)
        assert(1.0 - epsilon < trans[1][phone1.to_ind[len(phone1.to_ind) - 1]].sum(0) < 1.0 + epsilon) # make sure we normalized our probs
    return trans


def parse_hmm(f):
    """ parse HTK HMMdefs (chapter 7 of the HTK book) in f """
    l = f.readlines()
    n_phones = 0
    n_states_tot = 0
    for line in l:
        if '~h' in line:
            n_phones += 1
        elif '<NUMSTATES>' in line:
            n_states_tot += int(line.strip().split()[1]) - 2 
            # we remove init/end states: eg. 5 means 3 states once connected
    transitions = ({}, np.ndarray((n_states_tot, n_states_tot), 
        dtype='float64'))
    # transitions = ( t[phn] = Phone,
    #                               | phn1_s1, phn1_s2, phn1_s3, phn2_s1|
    #                     ----------|-----------------------------------|
    #                     | phn1_s1 | proba  , proba  , proba  , proba  |
    #                     | phn1_s2 | proba  , proba  , proba  , proba  |
    #                     | phn1_s3 | proba  , proba  , proba  , proba  |
    #                     | phn2_s1 | proba  , proba  , proba  , proba  |
    #                     -----------------------------------------------  )
    #             with proba_2 marking the transition from phn1_s1 to phn_s2
    gmms = {}
    #                 <---  mix. comp.  --->
    # gmms[phn] = [ [ [pi_k, mu_k, sigma2_k] , ...] , ...]
    #               <----------  state  ---------->
    # gmms[phn] is a list of states, which are a list of Gaussian mixtures 
    # components, which are a list of weight (float) followed by means (vec) 
    # and covar (vec, circular (i.e. diagonal covar matrix) covar)
    phn = ""
    phn_id = -1
    current_states_numbers = 0
    for i, line in enumerate(l):
        if '~h' in line:
            phn = clean(line).split()[1].strip('"')
            phn_id += 1
            gmms[phn] = []
        elif '<STATE>' in line:
            gmms[phn].append([])
        elif '<MIXTURE>' in line:
            gmms[phn][-1].append([float(clean(line).split()[2])])
        elif '<MEAN>' in line or '<VARIANCE>' in line:
            if not len(gmms[phn][-1]):
                gmms[phn][-1].append([1.0])
            gmms[phn][-1][-1].append(np.array(map(float, 
                clean(l[i+1]).split()), dtype='float64'))
        elif '<TRANSP>' in line:
            n_st = int(clean(line).split()[1]) - 2  # we also remove init/end
            transitions[0][phn] = Phone(phn_id, phn)
            for j in xrange(n_st):
                transitions[0][phn].update(current_states_numbers + j)
                transitions[1][current_states_numbers + j] = \
                    [0.0 for tmp_k in xrange(current_states_numbers)] + \
                    map(float, clean(l[i + j + 2]).split()[1:-1]) + \
                    [0.0 for tmp_k in xrange(n_states_tot
                        - current_states_numbers - n_st)]
            current_states_numbers += n_st
    assert(n_states_tot == current_states_numbers)
    #print gmms["!EXIT"][0][0][0] # pi_k of state 0 and mixture comp. 0
    #print gmms["!EXIT"][0][0][1] # mu_k
    #print gmms["!EXIT"][0][0][2] # sigma2_k
    #print gmms["eng"][0][0][0] # pi_k of state 0 and mixture comp. 0
    #print gmms["eng"][0][0][1] # mu_k
    #print gmms["eng"][0][0][2] # sigma2_k
    #print transitions[0].keys() # phones
    #print transitions[0]["!EXIT"] # !EXIT phn_id = 61
    #print transitions[1] # all the transitions
    #print transitions[1][transitions[0]['aa'].to_ind[2]]
    return n_states_tot, transitions, gmms


def process(ofname, iscpfname, ihmmfname, ilmfname):
    ihmmf = open(ihmmfname)
    ilmf = None
    n_states, transitions, gmms = parse_hmm(ihmmf)
    ihmmf.close()
    if ilmfname != None:
        ilmf = open(input_lm_fname)
        transitions = parse_lm(transitions, ilmf)
        ilmf.close()
    iscpf = open(iscpfname)
    
    gmms_ = precompute_det_inv(gmms)
    map_states_to_phones = phones_mapping(gmms)
    #gmms_ = [gm_st for _, gm in gmms.iteritems() for gm_st in gm]
    list_mlf_string = []
    for line_number, line in enumerate(iscpf): # TODO parallelize (pool.map for instance)
        cline = clean(line)
        print cline
        posteriors = compute_likelihoods(n_states,
                htkmfc.open(cline).getall(), gmms_)
        #viterbi(posteriors, transitions)
        #s = '"' + cline[:-3] + 'lab"\n' + \
        #        string_mlf(map_states_to_phones,
        #                cline[:-3] + 'lab',
        #                online_viterbi(n_states, htkmfc.open(cline).getall(), 
        #                gmms_, transitions)) + \
        #        '.\n'
        #list_mlf_string.append(s)
        if line_number > 0: # TODO remove
            break
    iscpf.close()
    with open(ofname, 'w') as of:
        of.write('#!MLF!#\n')
        for line in list_mlf_string:
            of.write(line)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        if '--help' in sys.argv:
            print usage
            sys.exit(0)
        #if '--debug' in sys.argv:
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        output_fname = l[1]
        input_scp_fname = l[2]
        input_hmm_fname = l[3]
        input_lm_fname = None
        if len(l) > 4:
            input_lm_fname = l[4]
        process(output_fname, input_scp_fname, 
                input_hmm_fname, input_lm_fname)
    else:
        print usage
        sys.exit(-1)
