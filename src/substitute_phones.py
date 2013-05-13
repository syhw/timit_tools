import os, sys, cPickle

"""
License: WTFPL http://www.wtfpl.net
Copyright: Gabriel Synnaeve 2013
"""

doc = """
Usage:
    python substitute_phones.py [$folder_path] [--sentences]

Substitutes phones found in .lab files (in-place) by using the foldings dict.

The optional --sentences argument will replace starting and ending pauses 
respectively by !ENTER and !EXIT. 
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
# http://troylee2008.blogspot.fr/2011/05/asr-complete-matlab-script-for-timit.html classifies 'q' as 'pau' (i.e. pause/silence) too



def process(folder, 
        sentences=False, # should we apply !ENTER/!EXIT for start/end?
        substitute=True, # should we substitute phones with foldings dict?
        startend_sil=False): # should we substitute start and end w/ sil
    if not substitute:
        foldings = {}
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
            text_buffer = []
            for line in fr:
                phones_before.append(line.split()[-1]) # phone last elt of line
                tmpline = line
                tmpline = tmpline.replace('-', '')
                tmp = tmpline.split()
                for k, v in foldings.iteritems():
                    if tmp[-1] == k:
                        tmp[-1] = v
                        tmpline = ' '.join(tmp)
                text_buffer.append(tmpline.split())
            first_phone = text_buffer[0][-1]
            last_phone = text_buffer[-1][-1]
            if sentences:
                if first_phone == 'h#' or first_phone == 'sil':
                    text_buffer[0] = text_buffer[0][:-1] + ['!ENTER']
                if last_phone == 'h#' or last_phone == 'sil':
                    text_buffer[-1] = text_buffer[-1][:-1] + ['!EXIT']
            if startend_sil:
                text_buffer[0] = text_buffer[0][:-1] + ['sil']
                text_buffer[-1] = text_buffer[-1][:-1] + ['sil']
            for buffer_line in text_buffer:
                phones_after.append(buffer_line[-1])
                fw.write(' '.join(buffer_line) + '\n')
            fw.close()
            os.remove(fullname+'~')
            for tmp_phn in phones_before:
                c_before[tmp_phn] = c_before.get(tmp_phn, 0) + 1
            for tmp_phn in phones_after:
                c_after[tmp_phn] = c_after.get(tmp_phn, 0) + 1
            print "dealt with", fullname 
    print "Counts before substitution", c_before
    print "Counts after substitution", c_after
    with open(folder.rstrip('/') + '/unigrams.pickle', 'w') as unidump:
        cPickle.dump(c_after, unidump)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print doc
            sys.exit(0)
        sentences = False
        substitute = True
        startend_sil = False
        if '--sentences' in sys.argv:
            sentences = True
        if '--startendsil' in sys.argv:
            startend_sil = True
        if '--nosubst' in sys.argv:
            substitute = False
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        foldername = '.'
        if len(l) > 1:
            foldername = l[1]
        process(foldername, sentences, substitute, startend_sil)
    else:
        process('.') # default
