import os, sys

"""
License: WTFPL http://www.wtfpl.net
Copyright: Gabriel Synnaeve 2013
"""

doc = """
Usage:
    python substitute_phones.py [$folder_path] [--sentences]

Substitutes phones found in .lab files (in-place) by using the foldings dict.

The optional --sentences argument will replace starting and ending pauses 
respectively by <s> and </s>.
"""

foldings = {'ux': 'uw', 
            'axr': 'er', 
            'em': 'm',
            'nx': 'n',
            'eng': 'ng',
            'hv': 'hh',
            'pcl': 'sil',
            'tcl': 'sil',
            'kcl': 'sil',
            'qcl': 'sil',
            'bcl': 'sil',
            'dcl': 'sil',
            'gcl': 'sil',
            'h#': 'sil',
            '#h': 'sil',
            'pau': 'sil',
            'epi': 'sil',
            'axh': 'ax',
            'el': 'l',
            'en': 'n',
            'sh': 'zh',
            'ao': 'aa',
            'ih': 'ix',
            'ah': 'ax',
            'q': 'sil'} # <- they removed 'Q' (glottal stop), is it ok to sil?

def process(folder, sentences=False):
    c_before = {}
    c_after = {}
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-4:] != '.lab':
                continue
            fullname = d.rstrip('/') + '/' + fname
            phones_before = []
            phones_after = []
            os.rename(fullname, fullname+'~')
            fr = open(fullname+'~', 'r')
            fw = open(fullname, 'w')
            saw_pause = 0
            for line in fr:
                phones_before.append(line.split()[2])
                tmpline = line
                if sentences:
                    if not saw_pause:
                        tmpline = tmpline.replace('h#', '<s>')
                    else:
                        tmpline = tmpline.replace('h#', '</s>')
                tmpline = tmpline.replace('-', '')
                tmp = tmpline.split()
                for k, v in foldings.iteritems():
                    if tmp[2] == k:
                        tmp[2] = v
                        tmpline = ' '.join(tmp) + '\n'
                fw.write(tmpline)
                phones_after.append(tmpline.split()[2])
                if 'h#' in line:
                    saw_pause += 1
            if saw_pause > 2:
                print "this file has more than 2 pauses", fname
            fw.close()
            os.remove(fullname+'~')
	    for phn in phones_before:
		c_before[phn] = c_before.get(phn, 0) + 1
            for phn in phones_after:
		c_after[phn] = c_after.get(phn, 0) + 1
            print "dealt with", fullname 
    print "Counts before substitution", c_before
    print "Counts after substitution", c_after


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print doc
            sys.exit(0)
        sentences = False
        if '--sentences' in sys.argv:
            sentences = True
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        foldername = '.'
        if len(l) > 1:
            foldername = l[1]
        process(foldername, sentences)
    else:
        process('.') # default
