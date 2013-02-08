'''
Python-port of the clever alorithms from Slaney (1993), which were first
implemented in Matlab by Nick Clarke (2007)
The original ideas have been published in the Apple TR #35, "An Efficient
Implementation of the Patterson-Holdsworth Cochlear Filter Bank." (You can
find this article under "PattersonsEar.pdf" on the WWW).
All formulas are form there, with exeption of the modification of the 
center-frequency calculation.
'''

'''
Ver 1.1: fixed problem with integer rate values
Ver 1.2: main program moved into "main()"-function

ThH, June-2012
Ver 1.2
'''

import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as mpl

dbFlag = 1  # Set to "1" to see debug information

# ----------------------------------------------------------------------------
def gtmSetParameters(loFreq, method = 'moore'):
    ''' Simply a subroutine to set the parameters for GammaToneMake (below) '''
    
    #  stop errors when a very low fequency is used
    loFreq = max(loFreq, 75)
    
    # Based on the "method", set the analysis parameters
    if (method == 'lyon') or (method == 'stanley'):
        # Lyon + Stanley Parameters (1988)
        EarQ = 8
        minBW = 125
        order = 2
    elif method == 'greenwood':
        # Greenwood Parameters (1990) as (nearly) in DSAM
        EarQ = 7.23824
        minBW = 22.8509
        order = 1
    elif (method == 'moore') or (method == 'glasberg'):
        # Glasberg and Moore Parameters (1990)
        EarQ = 9.26449
        minBW = 24.7
        order = 1
    elif method == 'wierddsam':
        EarQ = 9.26
        minBW = 15.719
        order = 1
    else:
        print 'Invalid "method" - unp.sing "moore"'
        EarQ = 9.26449
        minBW = 24.7
        order = 1
    
    return (loFreq, EarQ, minBW, order)
    
    
def GammaToneMake(fs,numChannels,loFreq,hiFreq,method):
    '''
    GammaToneMake(fs,numChannels,loFreq,hiFreq,method)
    
    Input:
        fs ... sampling frequency [Hz]
        numChannels ... number of Channels
        loFreq ... lower frequency limit
        hiFreq ... upper frequency limit
        method ... method for finding the parameters
        
    Output:
        forward ... "b"-coefficients for the linear filter
        feedback ... "a"-coefficients for the linear filter
        cf ... center frequency
        ERB ... Equivalent Rectangular Bandwidth
        B ... Gammatone filter parameter in Roy Patterson's ear model
        
    Computes the filter coefficients for a bank of Gammatone filters. These
    filters were defined by Patterson and Holdworth for simulating
    the cochlea. The results are returned as arrays of filter
    coefficients. Each row of the filter arrays (forward and feedback)
    can be passed to the SciPy "lfilter" function.
    '''
    
    fs = float(fs)
    (loFreq, EarQ, minBW, order) = gtmSetParameters(loFreq, method)        
    T = 1/fs
    
    # to make sure that the subsequent calculations are in float
    loFreq = float(loFreq)
    hiFreq = float(hiFreq)
    
    ERBlo = ((loFreq/EarQ)**order + minBW**order) ** (1/order)
    ERBhi = ((hiFreq/EarQ)**order + minBW**order) ** (1/order)
    overlap = (ERBhi/ERBlo) ** (1./(numChannels-1))
    ERB = np.array([ERBlo * (overlap ** channel) for channel in range(numChannels)])
    
    cf = EarQ*((ERB**order - minBW**order)**(1/order))
    pi = np.pi
    B = 1.019 * 2 * pi * ERB    # in rad here. Note: some models require B in Hz (NC)
    
#    a = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T))) 
#    b = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T))) 
#    c = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T))) 
#    d = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T))) 
#    e = (-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*pi*T) +2*(1 + np.exp(4*1j*cf*pi*T))/np.exp(B*T))**4     
    
    gain = abs( \
     (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T))) \
    *(-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T))) \
    *(-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T))) \
    *(-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T))) \
    /(-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*pi*T) +2*(1 + np.exp(4*1j*cf*pi*T))/np.exp(B*T))**4 )
    
    feedback = np.zeros((len(cf),9))
    forward =  np.zeros((len(cf),5))
    
    forward[:,0] =    T**4 / gain
    forward[:,1] = -4*T**4 * np.cos(2*cf*pi*T) / np.exp(B*T)   / gain
    forward[:,2] = 6 *T**4 * np.cos(4*cf*pi*T) / np.exp(2*B*T) / gain
    forward[:,3] = -4*T**4 * np.cos(6*cf*pi*T) / np.exp(3*B*T) / gain
    forward[:,4] =    T**4 * np.cos(8*cf*pi*T) / np.exp(4*B*T) / gain
    
    feedback[:,0] = np.ones(len(cf))
    feedback[:,1] = -8 * np.cos(2*cf*pi*T) / np.exp(B*T)
    feedback[:,2] =  4 * (4 + 3*np.cos(4*cf*pi*T)) / np.exp(2*B*T)
    feedback[:,3] = -8 * (6*np.cos(2*cf*pi*T) + np.cos(6*cf*pi*T)) / np.exp(3*B*T)
    feedback[:,4] =  2 * (18 + 16*np.cos(4*cf*pi*T) + np.cos(8*cf*pi*T)) / np.exp(4*B*T)
    feedback[:,5] = -8 * (6*np.cos(2*cf*pi*T) + np.cos(6*cf*pi*T)) / np.exp(5*B*T)
    feedback[:,6] =  4 * (4 + 3*np.cos(4*cf*pi*T)) / np.exp(6*B*T)
    feedback[:,7] = -8 * np.cos(2*cf*pi*T) / np.exp(7*B*T)
    feedback[:,8] = np.exp(-8*B*T)
    
#    print 'Done'
    return (forward,feedback,cf,ERB,B)

# ----------------------------------------------------------------------------
def GammaToneApply(x,forward,feedback):
    '''
    This function filters the waveform x with the array of filters
    specified by the forward and feedback parameters. Each row
    of the forward and feedback parameters are the parameters
    to the SciPy function "lfilter".
    '''

    # Allocate the memory
    (rows, cols) = np.shape(feedback)
    y = np.zeros( (rows,len(x)) )
    
    # Filter the signal
    for ii in range(rows):
        y[ii,:] = ss.lfilter(forward[ii,:], feedback[ii,:], x)
    
    return y
    
# ----------------------------------------------------------------------------
def BMMplot(stimIn, fc, sr, fcLabel):
    '''
    This is a simple plotting routine to mimic the basilar-membrane-movement
    plot types seen frequently in the "Journal Of The
    Acoustical Society Of America" (JASA) articles among others, as well as
    software such as AIM and AMS.  This allows data from each channel to be
    viewed as stackedline graphs.
    '''
    
    # plot the different traces above each other
    stimOut = np.zeros(np.shape(stimIn))
    for n in range(stimIn.shape[0]):
        stimOut[n, :] = n
    stimOut = stimIn / np.max(abs(stimIn)) + stimOut
    
    # Set the time axis
    timeAx = (np.arange(stimIn.shape[1])+1) *1.e3/sr

    # Plot the data    
    mpl.plot(timeAx,stimOut.transpose(),'k')
    
    # Format the plot
    if not fcLabel == '':
        mpl.yticks(fcLabel, np.round(fc[fcLabel]))
    else:
        mpl.yticks([0], '')
    
    mpl.ylim(-1, stimIn.shape[0])      
    mpl.xlabel('Time [ms]')
    mpl.ylabel('Center Frequency [Hz]')
    
# ----------------------------------------------------------------------------    
def main():
    ''' Test function, with a click-train as input. '''
    
    if dbFlag > 0:
        print 'Let''s start!'
        (forward,feedback,fc,ERB,B) = GammaToneMake(44100, 21,400, 5000,'moore')    
        
    sr = 16e3   # sampling rate
    x = np.zeros((sr*25e-3))     # create a 25ms input
    x[[1, 100, 200, 300]] = 1    # make a click train
    
    # And now for the real thing: make the filterbank ...
    (forward,feedback,fc,ERB,B) = GammaToneMake(sr,50,200,3000,'moore')    
    
    # ... and filter into individual channels.
    y = GammaToneApply(x,forward,feedback)
    
    # Show the plots
    mpl.figure(1)
    ax = mpl.subplot(121)
    # Show all frequencies, and label a selection of centre frequencies
    BMMplot(y, fc, sr, [0, 9, 19, 29, 39, 49])
    
    mpl.figure(1)
    ax = mpl.subplot(122)
    # For better visibility, plot selected center-frequencies in a second plot.
    # Dont plot the centre frequencies on the ordinate.
    BMMplot(y[[0, 9, 19, 29, 39, 49],:], fc, sr, '')
    
    mpl.show()
    mpl.close()
    
    if dbFlag > 0:
        print 'Done!'

# ----------------------------------------------------------------------------    

if __name__ == '__main__':
    main()    
