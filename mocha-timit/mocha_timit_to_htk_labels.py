import os, sys

def convert(folder):
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-4:] != '.phn':
                continue
            fullfname = d + '/' + fname
            fr = open(fullfname)
            fw = open(fullfname[:-4] + '.lab', 'w')
            for line in fr:
                [s, e, p] = line.rstrip('\n').split()
                s_ns = str(int(float(s) * 1000000000))
                e_ns = str(int(float(e) * 1000000000))
                fw.write(s_ns + ' ' + e_ns + ' ' + p + '\n')
            fr.close()
            fw.close()
            print "Converted", fullfname


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    print "Converting the *.phn (in frames) in *.lab (in nanosecs) in", folder
    convert(folder)
