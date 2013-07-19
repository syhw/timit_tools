import sys, cPickle
import numpy as np
import scipy.signal

"""
	Electromagnetic Articulograph	*.ema
		16bit (stored as 4byte float) 500Hz binary files with
		EST headers. Edinburgh Speech Tools Trackfile format 
		consists of a variable length ascii header and a 4 byte 
		float representation per channel. The first channel is a 
		time value in seconds the second value is always 1 (used 
		to indicate if the sample is present or not) subsequent 5 
		values are coil 1-5 x-values followed by coil 1-5 y-values 
		followed by coil 6-10 x-values and finally coils 6-10 y-values. 

    Electromagnetic Articulograph 500Hz sample rate (Carstens 10 Channel)
        upper incisor
        lower incisor
        upper lip
        lower lip
        tongue tip
        tongue blade
        tongue dorsum
        velum
"""

# to apply it to each of the ema files:
# for file in `ls ~/postdoc/datasets/MOCHA_TIMIT/msak0/*.ema`; do python ema.py $file; done
# if there is a problem
# for file in `ls ~/postdoc/datasets/MOCHA_TIMIT/msak0/*.ema`; do if [ ! -f ${file%.*}_ema.npy ]; then echo ${file%.*}; fi; done 

PLOT = True # plot or not?
WINDOW_DUR = 25.0E-3 # 25ms window
SAMPLING_STEP = 10.0E-3 # 10ms speech
SCIPY_RESAMPLING = False # using scipy.signal.resample, otherwise just use the mean on the grouped values

def clean(s):
    return s.rstrip('\n').strip()


def smooth(x, window_len=3, window="hamming"):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


### parsing EMA files 

columns = {}
columns[0] = 'time'
columns[1] = 'present?'

if len(sys.argv) < 2:
    print >> sys.stderr, "usage: python ema.py file.ema"
    sys.exit(-1)
fname = sys.argv[1]
#fname = '/Users/gabrielsynnaeve/postdoc/datasets/MOCHA_TIMIT/msak0/msak0_001.ema'
with open(fname, 'rb') as f:
    f.readline() # EST_File Track
    datatype = clean(f.readline()).split()[1]
    nframes = int(clean(f.readline()).split()[1])
    f.readline() # ByteOrder
    nchannels = int(clean(f.readline()).split()[1])
    while not 'CommentChar' in f.readline():
        pass # EqualSpace, BreaksPresent, CommentChar
    f.readline() # empty line
    line = clean(f.readline())
    while not "EST_Header_End" in line:
        channel_number = int(line.split()[0].split('_')[1])
        channel_name = line.split()[1]
        columns[channel_number + 2] = channel_name
        line = clean(f.readline())
    string = f.read()
    data = np.fromstring(string, dtype='float32')
    data = np.reshape(data, (-1, len(columns)))

assert(nframes == data.shape[0])
assert(data[:,1].all()) # we got all measurements
assert((np.abs(np.diff(data[:,0]) - 2.0E-3) < 1.0E-6).all()) # 500 Hz ? 

### and now for the smoothing and downsampling

sample_dur = data[1,0] - data[0,0] # duration in seconds of a sample
window_dur = WINDOW_DUR
w_l = window_dur/sample_dur # window length in samples
smoothed = np.ndarray(data.shape, dtype=float)
#smoothed = np.ndarray((data.shape[0]+w_l-2, data.shape[1]), dtype=float)
for c in xrange(data.shape[1]):
    smoothed[:,c] = smooth(data[:,c], window_len=w_l)[w_l/2:-w_l/2+2] # TODO check for correctness
    #smoothed[:,c] = smooth(data[:,c], window_len=w_l)
print "data shape", data.shape
print "smoothed shape", smoothed.shape
downsampling = int(np.round(SAMPLING_STEP / (sample_dur)))
print "using", downsampling, "samples for 1 final sample"
if SCIPY_RESAMPLING:
    arr = np.ndarray((smoothed.shape[0]/downsampling, smoothed.shape[1]), dtype=float)
else:
    arr = np.ndarray(((smoothed.shape[0]+4)/downsampling, smoothed.shape[1]), dtype=float)
for c in xrange(smoothed.shape[1]):
    if SCIPY_RESAMPLING:
        arr[:,c] = scipy.signal.resample(smoothed[:,c], smoothed.shape[0]/downsampling)
    else:
        arr[:,c] = map(lambda x: x.mean(), [smoothed[i:i+downsampling,c] for i in xrange(0, smoothed.shape[0], 5)])
print "downsampled shape", arr.shape
print "total duration", data[-1,0], "s"

if PLOT:
    import pylab as pl
    n_plots = data.shape[1] - 2
    print "n plots:", n_plots
    x_y_n = int(np.ceil(np.sqrt(n_plots)))
    for i in xrange(n_plots):
        pl.subplot(x_y_n, x_y_n - 1, i+1) # TODO -1
        pl.plot(arr[:, i+2])
        pl.ylabel(columns[i+2])
    pl.subplots_adjust(wspace=0.5, hspace=0.5)
    pl.suptitle('Electromagnetic Articulograph for ' + sys.argv[1]) # verticalalignment='bottom', horizontalalignment='center')
    pl.show()

#with open('/' + '/'.join(sys.argv[1].split('/')[:-1]) + '/ema_columns.pickle', 'w') as f:
#    cPickle.dump(columns, f)

#with open(sys.argv[1].replace('.ema', '_ema.npy'), 'w') as f:
#    np.save(f, arr)

