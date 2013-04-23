import numpy as np
import sys, math
import cPickle
from collections import defaultdict

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


def viterbi(posteriors, transitions):
    pass


def compute_likelihoods(f, gmms):
    pass


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
    return transitions, gmms


def process(ofname, iscpfname, ihmmfname, ilmfname):
    with open(ofname, 'w') as of:
        ihmmf = open(ihmmfname)
        ilmf = None
        transitions, gmms = parse_hmm(ihmmf)
        ihmmf.close()
        if ilmfname != None:
            ilmf = open(input_lm_fname)
            transitions = parse_lm(transitions, ilmf)
            ilmf.close()
        iscpf = open(iscpfname)
        posteriors = compute_likelihoods(iscpf, gmms)
        iscpf.close()
        of = viterbi(posteriors, transitions)


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
