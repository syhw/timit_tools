import os, sys

def convert(folder): 
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-7:] != '.phones' or fname[0] == '.': # no hidden files
                continue
            fullfname = d + '/' + fname
            # TODO SPLIT .phones INTO SEVERAL .lab
            fr = open(fullfname)
            fw = open(fullfname[:-7] + '.lab', 'w')
            start = 0.0
            for line in fr:
                if line[0].isalpha() or line[0] == '#': # header
                    continue
                tmp = line.strip(' \n').split(';')[0] # no comments
                if not len(tmp):
                    continue
                if len(tmp.split()) < 3:
                    e = tmp.split()[0]
                    start = float(e)
                else:
                    [e, _, p] = tmp.split()[:3] # removes all phones but the first one!
                    s_ns = str(int(float(start) * 1000000000))
                    e_ns = str(int(float(e) * 1000000000))
                    fw.write(s_ns + ' ' + e_ns + ' ' + p + '\n')
                    start = float(e)
            fr.close()
            fw.close()
            print "Converted", fullfname


if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    print "Converting the *.phones (in seconds) in *.lab (in nanosecs) in", folder
    convert(folder) 
