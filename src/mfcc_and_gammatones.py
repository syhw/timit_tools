"""
Transforms wav files into speech features files.

License: WTFPL http://www.wtfpl.net
Copyright: Gabriel Synnaeve 2013
"""

import os, shutil, sys
from subprocess import call
try:
    from numpy import save as npsave
except ImportError:
    print >> sys.stderr, "ERROR: You don't have numpy"
    sys.exit(-1)
try:
    import scipy  # just to test
except ImportError:
    print >> sys.stderr, "ERROR: You don't have scipy"
    sys.exit(-1)
from scipy.io import wavfile


USAGE = """
Usage:
    python mfcc_and_gammatones.py [$folder_path] [--debug] [--htk_mfcc] 
        [--gammatones] [--spectrograms] [--filterbanks] [--stereo] [--no-sox]

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
    - outputing the log filterbanks in file_fbanks.npy
"""

SPECGRAM_WINDOW = 0.020 # 20ms
SPECGRAM_OVERLAP = 0.010 # 10ms
FBANKS_WINDOW = 0.025 # 25ms
FBANKS_RATE = 100 # 10ms
N_FBANKS = 40
N_GAMMATONES_FILTERS = 1000


def process(folder,
        debug=False,
        htk_mfc=False,
        forcemfcext=False,
        stereo_wav=False,
        gammatones=False,
        spectrograms=False,
        filterbanks=False,
        sox=True):
    """ applies to all *.wav in folder """

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
    if gammatones:
        try:
            from brian import Hz, kHz
            from brian.hears import loadsound, erbspace, Gammatone
        except ImportError:
            print >> sys.stderr, "You need Brian Hears"
            print >> sys.stderr, "http://www.briansimulator.org/docs/\
                    hears.html"
            sys.exit(-1)
    if spectrograms:
        try:
            from pylab import specgram
        except ImportError:
            print >> sys.stderr, "You need Pylab"
            sys.exit(-1)
    fbanks = None
    if filterbanks:
        try:
            sys.path.append('../spectral')
            from spectral import Mel
        except ImportError:
            print >> sys.stderr, "You need spectral (in the parent folder)"
            print >> sys.stderr, "https://github.com/mwv/spectral"
            sys.exit(-1)

    # run through all the folders and files in the path "folder"
    # and put a header to the waves, save the originals as .rawaudio
    # use HCopy to produce MFCC files according to "wav_config" file
    for bdir, _, files in os.walk(folder):
        for fname in files:
            if fname[-4:] != '.wav':
                continue
            rawfname = bdir+'/'+fname[:-4]+'.rawaudio'
            wavfname = bdir+'/'+fname
            tempfname = bdir+'/'+fname[:-4]+'_temp.wav'
            # temp fname with .wav for sox
            mfccfname = bdir+'/'+fname[:-4]+mfc_extension
            if sox:
                shutil.move(wavfname, tempfname)
                call(['sox', tempfname, wavfname])
                # w/o headers, sox uses extension
                shutil.move(tempfname, rawfname)
            if htk_mfc:
                call(['HCopy', '-C', 'wav_config', wavfname, mfccfname])
            srate = 16000
            srate, sound = wavfile.read(wavfname)
            if stereo_wav and len(sound.shape) == 2: # in mono sound is a list
                sound = sound[:, 0] + sound[:, 1]
                # for stereo wav, sum both channels
            if gammatones:
                gammatonefname = bdir+'/'+fname[:-4]+'_gamma.npy'
                tmp_snd = loadsound(wavfname)
                gamma_cf = erbspace(20*Hz, 20*kHz, N_GAMMATONES_FILTERS)
                gamma_fb = Gammatone(tmp_snd, gamma_cf)
                with open(gammatonefname, 'w') as o_f:
                    npsave(o_f, gamma_fb.process())
            if spectrograms:
                powerspec, _, _, _ = specgram(sound, NFFT=int(srate
                    * SPECGRAM_WINDOW), Fs=srate, noverlap=int(srate
                        * SPECGRAM_OVERLAP)) # TODO
                specgramfname = bdir+'/'+fname[:-4]+'_specgram.npy'
                with open(specgramfname, 'w') as o_f:
                    npsave(o_f, powerspec.T)
            if filterbanks:
                # convert to Mel filterbanks
                if fbanks == None: # assume parameters are fixed
                    fbanks = Mel(nfilt=N_FBANKS,    # nb of filters in mel bank
                                 alpha=0.97,             # pre-emphasis
                                 fs=srate,               # sampling rate
                                 frate=FBANKS_RATE,      # frame rate
                                 wlen=FBANKS_WINDOW,     # window length
                                 nfft=512,               # length of dft
                                 mel_deltas=False,       # speed
                                 mel_deltasdeltas=False  # acceleration
                                 )
                fbank = fbanks.transform(sound)[0]  # first dimension is for
                                                    # deltas & deltasdeltas
                fbanksfname = bdir+'/'+fname[:-4]+'_fbanks.npy'
                with open(fbanksfname, 'w') as o_f:
                    npsave(o_f, fbank)
            # TODO wavelets scattergrams / scalograms
            print "dealt with file", wavfname


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print USAGE
            sys.exit(0)
        printdebug = False
        doforcemfcext = False
        dohtk_mfcc = False
        isstereo = False
        dogammatones = False
        dospectrograms = False
        dofilterbanks = False
        dosox = True
        if '--debug' in sys.argv:
            printdebug = True
        if '--forcemfcext' in sys.argv:
            doforcemfcext = True
        if '--htk-mfcc' in sys.argv:
            dohtk_mfcc = True
        if '--stereo' in sys.argv:
            isstereo = True
        if '--gammatones' in sys.argv:
            dogammatones = True
        if '--spectrograms' in sys.argv:
            dospectrograms = True
        if '--filterbanks' in sys.argv:
            dofilterbanks = True
        if '--no-sox' in sys.argv:
            dosox = False
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        foldername = '.'
        if len(l) > 1:
            foldername = l[1]
        process(foldername, printdebug, dohtk_mfcc, doforcemfcext, isstereo,
                dogammatones, dospectrograms, dofilterbanks, dosox)
    else:
        process('.') # default
