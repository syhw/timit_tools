import os, shutil, sys
from subprocess import call
try:
    import numpy 
except:
    print >> sys.stderr, "ERROR: You don't have numpy"
    sys.exit(-1)
try:
    import scipy
except:
    print >> sys.stderr, "ERROR: You don't have scipy"
    sys.exit(-1)
from scipy.io import wavfile
import cPickle

"""
License: WTFPL http://www.wtfpl.net
Copyright: Gabriel Synnaeve 2013
"""

doc = """
Usage:
    python mfcc_and_gammatones.py [$folder_path] [--debug] [--htk_mfcc] 

You may need:
    - GammaTones.py http://work.thaslwanter.at/CSS/Code/GammaTones.py
    - HCopy from HTK
    - wav_config (for HCopy, 25ms window, 10ms slide, 12 coefficiens, and 
        MFCC_0_D_A means we want the energy (0), first derivative (D) 
        and second derivative (A, acceleration).
    - this python file

For all file.wav wav files in the dataset, what this script does is eqvlt to:
    - mv file.wav file.rawaudio (because the wav in TIMIT is w/o headers)
    - sox file.rawaudio file.wav (to reconstruct the headers)
    - HCopy -A -D -T 1 -C wav_config file.wav file.mfc_unnorm
    - Gammatones, produces file_gamma_unnorm.npy
"""

def process(folder, 
        debug=False, htk_mfc=False, stereo_wav=False, gammatones=False):
    """ debug output? HCopy for MFCC? wav are stereo? produce gammatones? """
    fdname = folder.strip('/').rstrip('/') + '/'
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-4:] != '.wav':
                continue
            rawfname = fdname+d+'/'+fname[:-4]+'.rawaudio'
            wavfname = fdname+d+'/'+fname
            tempfname = fdname+d+'/'+fname[:-4]+'_temp.wav' # temp fname with .wav for sox
            mfccfname = fdname+d+'/'+fname[:-4]+'.mfc_unnorm'
            gammatonefname = fdname+d+'/'+fname[:-4]+'_gamma_unnorm.npy'
            shutil.move(wavfname, tempfname)
            call(['sox', tempfname, wavfname]) # w/o headers, sox uses extension
            shutil.move(tempfname, rawfname)
            if htk_mfc:
                call(['HCopy','-C', 'wav_config', wavfname, mfccfname])
            sr = 16000
            sr, sound = wavfile.read(wavfname)
            if stereo_wav and len(sound.shape) == 2: # in mono sound is a list
                sound = sound[:,1] # for stereo wav, arbitrarily take channel 1
            if gammatones:
                (forward, feedback, fc, ERB, B) = GammaTones.GammaToneMake(sr, 50, 200, 5000, 'moore')
                gamma_sound = GammaTones.GammaToneApply(sound, forward, feedback)
                cPickle.dump(gamma_sound, open(gammatonefname, 'w'), -1)
                if debug:
                    #print gamma_sound
                    import matplotlib.pyplot as mpl
                    mpl.figure(1)
                    ax = mpl.subplot(121)
                    GammaTones.BMMplot(gamma_sound, fc, sr, 
                            [0, 9, 19, 29, 39, 49])
                    mpl.figure(1)
                    ax = mpl.subplot(122)
                    GammaTones.BMMplot(gamma_sound[[0, 9, 19, 29, 39, 49],:],
                            fc, sr, '')
                    mpl.savefig('gamma.png')
                    mpl.close()
            print "dealt with file", wavfname


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print doc
            sys.exit(0)
        debug = False
        htk_mfcc = False
        stereo = False
        gammatones = False
        if '--debug' in sys.argv:
            debug = True
        if '--htk_mfcc' in sys.argv:
            htk_mfcc = True
        if '--stero' in sys.argv:
            stereo = True
        if '--gammatones' in sys.argv:
            gammatones = True
            try:
                import GammaTones
            except:
                print >> sys.stderr, "ERROR: You don't have GammaTones"
                sys.exit(-1)
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        foldername = '.'
        if len(l) > 1:
            foldername = l[1]
        process(foldername, debug, htk_mfcc, stereo, gammatones)
    else:
        process('.') # default
