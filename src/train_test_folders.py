import shutil, random, os, sys

# mkdir train && mkdir test

if len(sys.argv) < 2:
    print "usage: python train_test_folders.py source_folder [splitted_folder]"
    sys.exit(-1)

folder = sys.argv[1]
whereto = ''
if len(sys.argv) > 2:
    whereto = sys.argv[2].rstrip('/') + '/'

for d, ds, fs in os.walk(folder):
    for fname in fs:
        if fname[-4:] != '.lab':
            continue
        if fname[0] == '.':
            continue
        if random.uniform(0,1) > 0.2:
            print fname, 'to train'
            shutil.copy(d + '/' + fname, whereto + 'train/')
            shutil.copy(d + '/' + fname.split('.')[0] + '.mfc', whereto + 'train/')
            try:
                shutil.copy(d + '/' + fname.split('.')[0] + '_ema.npy', whereto + 'train/')
            except:
                print "no", fname.split('.')[0] + "_ema.npy"
        else:
            print fname, 'to test'
            shutil.copy(d + '/' + fname, whereto + 'test/')
            shutil.copy(d + '/' + fname.split('.')[0] + '.mfc', whereto + 'test/')
            try:
                shutil.copy(d + '/' + fname.split('.')[0] + '_ema.npy', whereto + 'test/')
            except:
                print "no", fname.split('.')[0] + "_ema.npy"
