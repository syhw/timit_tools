import sys
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
"""

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


columns = {}
columns[0] = 'time'
columns[1] = 'present?'

fname = sys.argv[1]
#fname = '/Users/gabrielsynnaeve/postdoc/datasets/MOCHA_TIMIT/msak0/msak0_001.ema'
with open(fname, 'rb') as f:
    f.readline() # EST_File Track
    datatype = clean(f.readline()).split()[1]
    nframes = int(clean(f.readline()).split()[1])
    f.readline() # ByteOrder
    nchannels = int(clean(f.readline()).split()[1])
    f.readline() # EqualSpace
    f.readline() # BreaksPresent
    f.readline() # CommentChar
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

#print data[:,0] # time in sec
#print data[:,1] # 1 if the sample is present
#print columns[2], data[:,2] # values of coil 1 (channel columns[2])
import pylab as pl
#pl.plot(data[:,2])
#pl.show()
sample_dur = data[1,0] - data[0,0] # duration in seconds of a sample
#pl.plot(smooth(data[:, 2], window_len=25.0E-3/sample_dur))
#pl.show()
downsampling = 10.0E-3 / (sample_dur) # 10ms speech
arr = np.ndarray((data.shape[0]/downsampling, data.shape[1]), dtype=float)
for c in xrange(data.shape[1]):
    arr[:,c] = scipy.signal.resample(data[:,c], data.shape[0]/downsampling)
#pl.plot(arr[:, 2])
#pl.show()

assert(nframes == data.shape[0])


