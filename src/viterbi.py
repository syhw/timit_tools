import numpy as np
import sys
import cPickle

usage = """
python viterbi.py OUTPUT[.mlf] INPUT_SCP INPUT_HMM [INPUT_LM] [options: --help]
"""

def clean(s):
    return s.strip().rstrip('\n')


def viterbi(posteriors, transitions):
    pass


def compute_likelihoods(f, gmms):
    pass


def parse_lm(f):
    pass


def parse_hmm(f):
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
    # transitions = ( t[phn] = (nb_states, id),
    #                               | phn1_s1, phn1_s2, phn1_s3, phn2_s1|
    #                     ----------|-----------------------------------|
    #                     | phn1_s1 | proba  , proba  , proba  , proba  |
    #                     | phn1_s2 | proba  , proba  , proba  , proba  |
    #                     | phn1_s3 | proba  , proba  , proba  , proba  |
    #                     | phn2_s1 | proba  , proba  , proba  , proba  |
    #                     -----------------------------------------------  )
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
            transitions[0][phn] = (n_st, phn_id)
            for j in xrange(n_st):
                sys.stdout.flush()
                transitions[1][phn_id + j] = \
                    [0.0 for tmp_k in xrange(current_states_numbers)] + \
                    map(float, clean(l[i + j + 2]).split()[1:-1]) + \
                    [0.0 for tmp_k in xrange(n_states_tot
                        - current_states_numbers - n_st)]
            current_states_numbers += n_st
            # TODO do not forget to have transition[1] lines that sum to 1
    #print gmms["!EXIT"][0][0][0] # pi_k of state 0 and mixture comp. 0
    #print gmms["!EXIT"][0][0][1] # mu_k
    #print gmms["!EXIT"][0][0][2] # sigma2_k
    #print gmms["eng"][0][0][0] # pi_k of state 0 and mixture comp. 0
    #print gmms["eng"][0][0][1] # mu_k
    #print gmms["eng"][0][0][2] # sigma2_k
    #print transitions[0].keys() # phones
    #print transitions[0]["!EXIT"] # (n_st, phn_id) -> !EXIT phn_id = 61
    #print transitions[1] # all the transitions
    #print transitions[1][61:64] # last 3 states, i.e. the last phone (!EXIT)
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
