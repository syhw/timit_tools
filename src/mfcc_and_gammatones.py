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
        [--gammatones] [--stereo]

You may need:
    - HCopy from HTK
    - wav_config (for HCopy, 25ms window, 10ms slide, 12 coefficiens, and 
        MFCC_0_D_A means we want the energy (0), first derivative (D) 
        and second derivative (A, acceleration).
    - Brian hears http://www.briansimulator.org/docs/hears.html
    - this python file

For all file.wav wav files in the dataset, what this script does is eqvlt to:
    - mv file.wav file.rawaudio (because the wav in TIMIT is w/o headers)
    - sox file.rawaudio file.wav (to reconstruct the headers)
    - HCopy -A -D -T 1 -C wav_config file.wav file.mfc_unnorm
    - outputing the gammatones in file_gamma.npy
    - outputing the spectrograms in file_specgram.npy
"""

SPECGRAM_WINDOW = 0.020 # 20ms
SPECGRAM_OVERLAP = 0.010 # 10ms
N_GAMMATONES_FILTERS = 1000 # 3000 ?

def process(folder, 
        debug=False, 
        htk_mfc=False, 
        forcemfcext=False,
        stereo_wav=False, 
        gammatones=False,
        spectrograms=False):
    """ debug output? HCopy for MFCC? wav are stereo? produce gammatones? """

    # first find if we produce normalized MFCC, otherwise note it in the ext
    # because we can then normalize on the whole corpus with another py script
    mfc_extension = '.mfc_unnorm'
    wcfg = open('wav_config', 'r')
    for line in wcfg:
        if "ENORMALISE" in line:
            mfc_extension = '.mfc'
    if forcemfcext:
        mfc_extension = '.mfc'
    print "MFC extension:", mfc_extension

    # run through all the folders and files in the path "folder"
    # and put a header to the waves, save the originals as .rawaudio
    # use HCopy to produce MFCC files according to "wav_config" file
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if fname[-4:] != '.wav':
                continue
            rawfname = d+'/'+fname[:-4]+'.rawaudio'
            wavfname = d+'/'+fname
            tempfname = d+'/'+fname[:-4]+'_temp.wav' # temp fname with .wav for sox
            mfccfname = d+'/'+fname[:-4]+mfc_extension
            shutil.move(wavfname, tempfname)
            call(['sox', tempfname, wavfname]) # w/o headers, sox uses extension
            shutil.move(tempfname, rawfname)
            if htk_mfc:
                call(['HCopy', '-C', 'wav_config', wavfname, mfccfname])
            sr = 16000
            sr, sound = wavfile.read(wavfname)
            if stereo_wav and len(sound.shape) == 2: # in mono sound is a list
                sound = sound[:,1] # for stereo wav, arbitrarily take channel 1
            if gammatones:
                from brian import Hz, kHz
                from brian.hears import loadsound, erbspace, Gammatone
                gammatonefname = d+'/'+fname[:-4]+'_gamma.npy'
                tmp_snd = loadsound(wavfname)
                cf = erbspace(20*Hz, 20*kHz, N_GAMMATONES_FILTERS)
                fb = Gammatone(tmp_snd, cf)
                with open(gammatonefname, 'w') as of:
                    numpy.save(of, fb.process())
            if spectrograms:
                from pylab import specgram
                Pxx, freqs, bins, im = specgram(sound, NFFT=int(sr * SPECGRAM_WINDOW), Fs=sr, noverlap=int(sr * SPECGRAM_OVERLAP))
                specgramfname = d+'/'+fname[:-4]+'_specgram.npy'
                with open(specgramfname, 'w') as of:
                    numpy.save(of, Pxx.T)
            print "dealt with file", wavfname


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print doc
            sys.exit(0)
        debug = False
        forcemfcext = False
        htk_mfcc = False
        stereo = False
        gammatones = False
        spectrograms = False
        if '--debug' in sys.argv:
            debug = True
        if '--forcemfcext' in sys.argv:
            forcemfcext = True
        if '--htk-mfcc' in sys.argv:
            htk_mfcc = True
        if '--stereo' in sys.argv:
            stereo = True
        if '--gammatones' in sys.argv:
            gammatones = True
        if '--spectrograms' in sys.argv:
            spectrograms = True
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        foldername = '.'
        if len(l) > 1:
            foldername = l[1]
        process(foldername, debug, htk_mfcc, forcemfcext, stereo, gammatones, spectrograms)
    else:
        process('.') # default
