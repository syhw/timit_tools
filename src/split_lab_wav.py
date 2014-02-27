import os, sys, math
import numpy as np
import scipy.io.wavfile as wav
###import wave, struct

ONE_S = 1000000000 # one second in nanoseconds
MFCC_FRAMERATE = 10000000 # 10ms MFCC framerate (in nanoseconds)
MFCC_FRAMES_PER_SECOND = ONE_S/MFCC_FRAMERATE
MIN_PHONES = 4
assert(MIN_PHONES >= 1)
N_ENTER_EXIT_FRAMES = 5 # number of !ENTER and !EXIT frames, currently this adds NOISE
                        # see TODO in wave_split()

usage = """python src/split_lab_mfc.py $folder $phone1 [$phone2] [$phone3]
-> try with IVER and VOCNOISE"""



def wave_split(start, end, wavfname, compt):
    # start and end in nanoseconds (float)
    #print "start, end", start, end

    (fr, sample) = wav.read(wavfname)
    #print fr
    #print sample.shape
    #print type(sample)
    ###wavefile = wave.open(wavfname, 'r')
    ###nf = wavefile.getnframes()
    ###nchannels = wavefile.getnchannels()
    ###fr = wavefile.getframerate()

    mult = fr * 1. / ONE_S # because start and end are in nanoseconds
    start_in_frames = int(start * mult)
    end_in_frames = int(math.ceil(end) * mult) + fr / (2 * MFCC_FRAMES_PER_SECOND) # adds half a MFCC frame

    interval = sample[start_in_frames:end_in_frames]
    if interval.shape[0] == 0:
        return -1
    max_volume = np.max(interval)
    ###wavefile.readframes(start_in_frames) # skipping to start
    ###interval = wavefile.readframes(end_in_frames - start_in_frames)
    ###values = struct.unpack_from("%dh" % (end_in_frames - start_in_frames) * nchannels, interval)
    ###max_volume = np.max(values)

    outputfilename = wavfname[:-4] + '_' + str(compt) + '.wav'

    to_write = interval
    n_frames_enter_exit = N_ENTER_EXIT_FRAMES * fr / MFCC_FRAMES_PER_SECOND
    ###wavefile_ext = wave.open(outputfilename, 'w')
    ###wavefile_ext.setparams((nchannels, wavefile.getsampwidth(), 
    ###    fr, end_in_frames + 2*n_frames_enter_exit, 
    ###    wavefile.getcomptype(), wavefile.getcompname()))


    if N_ENTER_EXIT_FRAMES:
        blank = 0 * interval[:2*n_frames_enter_exit] # TODO
        blank += 0.01 * max_volume * np.random.random(2*n_frames_enter_exit) # TODO
        to_write = np.append(blank, to_write)
        ###wavefile_ext.writeframes(struct.pack("%dh" % n_frames_enter_exit * nchannels, *blank))
    ###wavefile_ext.writeframes(interval)
    if N_ENTER_EXIT_FRAMES:
        blank = 0 * interval[:2*n_frames_enter_exit] # TODO
        blank += 0.01 * max_volume * np.random.random(2*n_frames_enter_exit) # TODO
        to_write = np.append(to_write, blank)
        ###wavefile_ext.writeframes(struct.pack("%dh" % n_frames_enter_exit * nchannels, *blank))

    wav.write(outputfilename, fr, to_write)
    ###wavefile_ext.close()
    ###return wavefile_ext
    return 0


def write_lab(buffer, labfname, compt):
    assert len(buffer) >= MIN_PHONES + 2 # at least MIN_PHONES phone in between
    offs = buffer[1][0] - N_ENTER_EXIT_FRAMES * MFCC_FRAMERATE # offset
    with open(labfname[:-4] + '_' + str(compt) + '.lab', 'w') as lab_w:
        if N_ENTER_EXIT_FRAMES:
            lab_w.write('0 ' + str(N_ENTER_EXIT_FRAMES * MFCC_FRAMERATE) + ' !ENTER\n')
        end = 0
        for s, e, p in buffer[1:-1]:
            end = e
            lab_w.write(str(s - offs) + ' ' + str(e - offs) + ' ' + p + '\n')
        if N_ENTER_EXIT_FRAMES:
            lab_w.write(str(end) + ' ' + str(end + N_ENTER_EXIT_FRAMES * MFCC_FRAMERATE) + ' !EXIT\n')


def split_in(folder, split_phones):
    sp = split_phones
    sp.append('{B_TRANS}')
    sp.append('{E_TRANS}')
    files_buffer = []
    for d, _, fs in os.walk(folder):
        for fname in fs:
            files_buffer.append((d, fname))
    print files_buffer
    for d, fname in files_buffer:
        if fname[-4:] != '.lab' or fname[0] == '.':
            continue
        lab_fname = d + '/' + fname
        mfc_fname = lab_fname[:-4] + '.mfc'
        wav_fname = lab_fname[:-4] + '.wav'
        raw_fname = lab_fname[:-4] + '.rawaudio'
        phones_fname = lab_fname[:-4] + '.phones'
        compt = 0
        with open(lab_fname) as lab_r:
            buffer = []
            for line in lab_r:
                s, e, p = line.strip(' \n').split()
                s, e = int(s), int(e)
                buffer.append((s, e, p))
                if p in split_phones or 'SIL' in p and (e-s) > 0.5*ONE_S:
                    if len(buffer) >= MIN_PHONES + 2:
                        start = buffer[1][0]
                        end = buffer[-2][1]
                        assert(end > start)
                        if wave_split(start, end, wav_fname, compt) != 0:
                            print "ERROR with:", start, end, wav_fname, compt
                            break
                        write_lab(buffer, lab_fname, compt)
                        compt += 1
                    buffer = [(int(s), int(e), p)]
        print "did", lab_fname
        # some cleaning now:
        os.remove(lab_fname)
        if os.path.isfile(mfc_fname):
            os.remove(mfc_fname)
        if os.path.isfile(wav_fname):
            os.remove(wav_fname)
        if os.path.isfile(raw_fname):
            os.remove(raw_fname)
        os.remove(phones_fname)


if __name__ == '__main__':
    folder = '.'
    split_phones = []
    if len(sys.argv) > 2:
        folder = sys.argv[1]
        split_phones = sys.argv[2:]
    else:
        print usage
        sys.exit(-1)
    print "Spliting *.lab and *.wav files on", split_phones, "in", folder
    split_in(folder, split_phones) 
