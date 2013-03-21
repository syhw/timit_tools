import os, sys, shutil
from collections import Counter

"""
License: WTFPL http://www.wtfpl.net
Copyright: Gabriel Synnaeve 2013
"""

doc = """
Usage:
    python create_phonesMLF_and_labels.py [$folder_path]

Will create "${folder}.mlf", "${folder}.scp", "labels" files in $folder_path.

If you run it only on the training folder, all the phones that you will
encounter in the test should be present in training so that the "labels" 
corresponds.
"""

def process(folder):
    folder = folder.rstrip('/')
    c = Counter()
    mfc_list_fname = folder + '/' + folder.split('/')[-1] + '.scp'
    master_label_fname = folder + '/' + folder.split('/')[-1] + '.mlf'
    labels_fname = folder + '/labels'
    mfc_list_file = open(mfc_list_fname, 'w')
    master_label_file = open(master_label_fname, 'w')
    labels_file = open(labels_fname, 'w')
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            fullname = d.rstrip('/') + '/' + fname
            if fname[-4:] == '.mfc': # TODO .mfc_unnorm also
                mfc_list_file.write(fullname + '\n')
            if fname[-4:] != '.lab':
                continue
            master_label_file.write('"' + fullname + '"\n')
            phones = []
            for line in open(fullname):
                master_label_file.write(line)
                phones.append(line.split()[2])
            c.update(phones)
            master_label_file.write('.\n')
            print "dealt with", fullname 
    mfc_list_file.close()
    print "written SCP file", mfc_list_fname
    master_label_file.close()
    print "written MLF file", master_label_fname
    labels_file = open(labels_fname, 'w')
    for label in c.iterkeys():
        labels_file.write(label + '\n')
    labels_file.close()
    print "written labels dict", labels_fname
    print "phones counts:", c
    print "number of phones:", len(c)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print doc
            sys.exit(0)
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        foldername = '.'
        if len(l) > 1:
            foldername = l[1]
        process(foldername)
    else:
        process('.') # default
