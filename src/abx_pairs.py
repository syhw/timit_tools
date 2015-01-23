""" Takes an MLF as input and gives triphones in the *.items ABX format as output.

python src/abx_pairs.py aligned.mlf [foldings.json]

e.g.:
python src/abx_pairs.py /fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split/aligned_train.mlf timit_foldings.json
"""

# currently tested only for TIMIT
import sys, json

def find_triphones(mlf, foldings={}, triphones_mode=True):
    ret = []
    current_file = None
    with open(mlf) as f:
        skipfile = False
        for line in f:
            if '.lab"' in line:
                current_file = line.rstrip('\n').strip('"')
                current_file_fbank = current_file.split('.')[0] + '_fbanks.npy'
                try:
                    rf = open(current_file_fbank)
                    skipfile = False
                    rf.close()
                except IOError:
                    # skip this file if we don't have the fbanks
                    skipfile = True
                current_file_fbank = current_file.split('.')[0]
                current_file_fbank = current_file_fbank.split('/')[-2] + "_" +\
                    current_file_fbank.split('/')[-1]
                t_minus_2 = None
                t_minus_1 = None
                p_minus_2 = None
                p_minus_1 = None
                current_talker = current_file.split('/')[-2]
            if skipfile:
                continue
            if line[0].isdigit():
                # assume a line of START END PHONE_STATE LOG_LIKELIHOOD [PHONE]
                # e.g. 1100000 1200000 d[2] -67.333511 d
                # or   1200000 1400000 d[3] -151.204285
                tmp = line.rstrip('\n').split()
                if len(tmp) == 5:  # that means we have the final [PHONE]
                    t = (str(float(tmp[0])/10000000), str(float(tmp[1])/10000000))
                    p = tmp[-1]
                    if p == "!ENTER" or p == "!EXIT":  # we omit ENTER and EXIT
                        continue
                    if p in foldings:
                        p = foldings[p]
                    if t_minus_2 is not None and p_minus_2 is not None:
                        if triphones_mode:
                            ret.append([current_file_fbank, t_minus_2[0], t[1], 
                                p_minus_1, p_minus_2 + "-" + p, current_talker])
                        else:
                            ret.append([current_file_fbank, t_minus_1[0], t_minus_1[1], 
                                p_minus_1, p_minus_2 + "-" + p, current_talker])
                    p_minus_2 = p_minus_1
                    p_minus_1 = p
                    t_minus_2 = t_minus_1
                    t_minus_1 = t
    return ret


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print __doc__
    print >> sys.stderr, "working on the MLF:", sys.argv[1]
    print >> sys.stderr, "!!! works only if the fbanks feature files exist!"
    foldings = {}
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            foldings = json.load(f)
    l = find_triphones(sys.argv[1], foldings, False)
    print >> sys.stderr, "filename onset offset phone context(left-right) talker"
    print "#file onset offset #phone context talker"
    print "\n".join(map(lambda x: " ".join(x), l))
