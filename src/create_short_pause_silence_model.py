import sys

usage = """
    python create_short_pause_silence_model.py in_hmmdef out_hmmdef out_labels

    Adds the "sp" phone to out_hmmdef from in_hmmdef "sil" middle state and 
    also adds "sp" to the labels in out_labels.
"""

def add_sp(in_hmmdef, out_hmmdef, out_labels):
    found_sil = False
    skip = False
    out = open(out_hmmdef, 'w')
    out_l = open(out_labels, 'w')
    for line in open(in_hmmdef):
        if "~h" in line:
            out_l.write(line.split()[-1].rstrip('\n').strip('"') + '\n')
        out.write(line)
    for line in open(in_hmmdef):
        if found_sil and "<STATE>" in line:
            if not "<STATE> 3" in line:
                skip = True
            else:
                skip = False
        if found_sil and "<TRANSP>" in line:
            out.write("<TRANSP> 3\n")
            out.write(" 0.000000e+00 1.000000e+00 0.000000e+00\n")
            out.write(" 0.000000e+00 0.500000e+00 0.500000e+00\n") # TODO check
            out.write(" 0.000000e+00 0.000000e+00 0.000000e+00\n")
        if found_sil and not skip:
            if "<NUMSTATES>" in line:
                tmp = line.split()
                tmp[-1] = "3\n"
                out.write(' '.join(tmp))
            elif "<STATE> 3" in line:
                tmp = line.split()
                tmp[-1] = "2\n"
                out.write(' '.join(tmp))
            else:
                out.write(line)
        if '~h "sil"' in line:
            found_sil = True
            out.write('~h "sp"\n')
            out_l.write("sp\n")
        if found_sil and "<ENDHMM>" in line:
            found_sil = False
            out.write(line) # we should be skipping at this point
    out.close()
    out_l.close()
    print 'Written', out_labels, 'adding "sp"'
    print "Written", out_hmmdef, "with a short-pause model"


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print >> sys.stderr, usage
        sys.exit(-1)
    add_sp(sys.argv[1], sys.argv[2], sys.argv[3])

