# Filter

import numpy as np
from scipy import signal

# IIR filter
# Input: signal and filter coefficients/parameters
# Output: filtered signal
# Funtions with "sos" in their names use second-order sections (sos) for numerical stability.
# Funtions with "2" in their names use fordward-backward filtering to realize zero phase.

def iir(y, b, a, axis=0):
    return signal.lfilter(b, a, y, axis=axis)

def iirsos(y, sos, axis=0):
    return signal.sosfilt(sos, y, axis=axis)

def butter(y, sr, btype, order, freq, axis=0):
    # N (lp or hp) or 2*N (bp or bs) -order Butterworth filter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    b, a = ba_butter(sr, btype, order, freq)
    return iir(y, b, a, axis=axis)

def bessel(y, sr, btype, order, freq, axis=0):
    # N (lp or hp) or 2*N (bp or bs) -order Bessel filter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    b, a = ba_bessel(sr, btype, order, freq)
    return iir(y, b, a, axis=axis)

def buttersos(y, sr, btype, order, freq, axis=0):
    sos = sos_butter(sr, btype, order, freq)
    return iirsos(y, sos, axis=axis)

def besselsos(y, sr, btype, order, freq, axis=0):
    sos = sos_bessel(sr, btype, order, freq)
    return iirsos(y, sos, axis=axis)

def bq(y, sr, sos_bq_type, axis=0, **kwargs):
    # Single biquad filter.
    # The type of sos_bq_type is FUNCTION, e.g. sos_bq_type=sos_bq_allpass.
    # **kwargs need to be single values, e.g. freq=200, Q=0.7.
    # **kwargs may include: freq, Q, bw (bandwidth), slope, gain. But only freq is always needed.
    # Please refer to the different "sos_bq_type" functions in the "IIR filter coefficient" section.
    sos = sos_bq(sr, sos_bq_type, **kwargs)
    return iirsos(y, sos, axis=axis)

def bqcas(y, sr, sos_bq_type, axis=0, **kwargs):
    # Cascaded biquad filters of the same type.
    # The type of sos_bq_type is FUNCTION, e.g. sos_bq_type=sos_bq_allpass.
    # **kwargs need to be lists, e.g. freq=[200, 500, 1000], Q=[0.7, 1.0, 1.2].
    # The length of input lists (should be the same) is the number of sos.
    # Returned sos array shape is (number of sos, 6).
    sos = sos_bqcas(sr, sos_bq_type, **kwargs)
    return iirsos(y, sos, axis=axis)

def bqtile(y, sr, sos_bq_type, ntile=2, axis=0, **kwargs):
    # Tiled biquad filters of the same type and the same parameters.
    # **kwargs need to be single values, e.g. freq=200, Q=0.7.
    sos = sos_bq(sr, sos_bq_type, **kwargs)
    sos = tile_sos(sos, ntile)
    return iirsos(y, sos, axis=axis)

def iir2(y, b, a, axis=0):
    return signal.filtfilt(b, a, y, axis=axis)

def iirsos2(y, sos, axis=0):
    return signal.sosfiltfilt(sos, y, axis=axis)

def butter2(y, sr, btype, order, freq, axis=0):
    b, a = ba_butter(sr, btype, order, freq)
    return iir2(y, b, a, axis=axis)

def bessel2(y, sr, btype, order, freq, axis=0):
    b, a = ba_bessel(sr, btype, order, freq)
    return iir2(y, b, a, axis=axis)

def buttersos2(y, sr, btype, order, freq, axis=0):
    sos = sos_butter(sr, btype, order, freq)
    return iirsos2(y, sos, axis=axis)

def besselsos2(y, sr, btype, order, freq, axis=0):
    sos = sos_bessel(sr, btype, order, freq)
    return iirsos2(y, sos, axis=axis)

def bq2(y, sr, sos_bq_type, axis=0, **kwargs):
    sos = sos_bq(sr, sos_bq_type, **kwargs)
    return iirsos2(y, sos, axis=axis)

def bqcas2(y, sr, sos_bq_type, axis=0, **kwargs):
    sos = sos_bqcas(sr, sos_bq_type, **kwargs)
    return iirsos2(y, sos, axis=axis)

def bqtile2(y, sr, sos_bq_type, ntile=2, axis=0, **kwargs):
    sos = sos_bq(sr, sos_bq_type, **kwargs)
    sos = tile_sos(sos, ntile)
    return iirsos2(y, sos, axis=axis)

# IIR filter coefficient
# Input: filter parameters
# Output: filter coefficients
# Reference: Audio EQ Cookbook (W3C Working Group Note, 08 June 2021)

def cascade_sos(sos_list):
    # input sos list (or tuple): [sos1, sos2,..., sosn]
    return np.concatenate(sos_list, axis=0)

def tile_sos(sos, ntile):
    # Tile a sos from shape(nsos, 6) to shape(ntile*nsos, 6).
    return np.tile(sos, (ntile, 1))

def ba_butter(sr, btype, order, freq):
    # Get the b and a parameters of a butterworth IIR filter.
    # signal.butter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.butter(order, freq, btype=btype, fs=sr)

def ba_bessel(sr, btype, order, freq):
    # Get the b and a parameters of a bessel IIR filter.
    # signal.butter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.bessel(order, freq, btype=btype, fs=sr)

def sos_butter(sr, btype, order, freq):
    # Get the sos of a butterworth IIR filter.
    # signal.butter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.butter(order, freq, btype=btype, output='sos', fs=sr)

def sos_bessel(sr, btype, order, freq):
    # Get the sos of a bessel IIR filter.
    # signal.butter
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.bessel(order, freq, btype=btype, output='sos', fs=sr)

def sos_iir(sr, ftype, btype, order, freq, rp=None, rs=None):
    # Get the sos of certain types of IIR filter.
    # signal.iirfilter
    # ftype: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
    # btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    return signal.iirfilter(order, freq, rp=rp, rs=rs, btype=btype, \
                            ftype=ftype, output='sos', fs=sr)

def sos_bq(sr, sos_bq_type, **kwargs):
    # Get the sos coefficient array of a single biquad filter.
    return sos_bq_type(sr, **kwargs)

def sos_bqcas(sr, sos_bq_type, **kwargs):
    # Get the sos coefficient array of cascaded biquad filters, limited to the same biquad type.
    kwargs_key, kwargs_val = list(kwargs.keys()), list(kwargs.values())
    nsos = len(kwargs_val[0])
    sos = np.empty((0, 6))
    for i in range(nsos):
        subkwargs_val = list(sub_val[i] for sub_val in kwargs_val)
        subkwargs = dict(zip(kwargs_key, subkwargs_val))
        sos_ = sos_bq_type(sr, **subkwargs)
        sos = np.append(sos, sos_, axis=0)
    return sos

def sos_bq_allpass(sr, freq, Q):
    # biquad all pass filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([1-alpha, -2*cosw, 1+alpha])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])
        
def sos_bq_lowpass(sr, freq, Q):
    # biquad low pass filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([0.5*(1-cosw), 1-cosw, 0.5*(1-cosw)])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])

def sos_bq_highpass(sr, freq, Q):
    # biquad high pass filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = 0.5*sinw/Q
    norm = 1+alpha    
    b = np.array([0.5*(1+cosw), -1-cosw, 0.5*(1+cosw)])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])

def sos_bq_bandpass(sr, freq, bw):
    # biquad band pass filter with a constant 0 dB peak gain.
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = sinw*np.sinh(0.5*np.log(2)*bw*w/sinw)
    norm = 1+alpha    
    b = np.array([alpha, 0.0, -alpha])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])

def sos_bq_bandstop(sr, freq, bw):
    # biquad band stop (notch) filter
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = sinw*np.sinh(0.5*np.log(2)*bw*w/sinw)
    norm = 1+alpha    
    b = np.array([1.0, -2*cosw, 1.0])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha)/norm])
    return np.array([np.append(b, a)])
    
def sos_bq_peak(sr, freq, bw, gain):
    # biquad peaking (bell) EQ
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    alpha = sinw*np.sinh(0.5*np.log(2)*bw*w/sinw)
    A = np.power(10, gain/40)
    norm = 1+alpha/A    
    b = np.array([1+alpha*A, -2*cosw, 1-alpha*A])/norm
    a = np.array([1.0, -2*cosw/norm, (1-alpha/A)/norm])
    return np.array([np.append(b, a)])

def sos_bq_lowshelf(sr, freq, slope, gain):
    # biquad low shelf
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    A = np.power(10, gain/40)
    alpha = 0.5*sinw*np.sqrt((A+1/A)*(1/slope-1)+2)
    norm = (A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha    
    b0 = A*((A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = 2*A*((A-1) - (A+1)*cosw)
    b2 = A*((A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha)
    a1 = -2*((A-1) + (A+1)*cosw)
    a2 = (A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha
    b = np.array([b0, b1, b2])/norm
    a = np.array([1.0, a1/norm, a2/norm])
    return np.array([np.append(b, a)])

def sos_bq_highshelf(sr, freq, slope, gain):
    # biquad high shelf
    w = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w), np.sin(w)
    A = np.power(10, gain/40)
    alpha = 0.5*sinw*np.sqrt((A+1/A)*(1/slope-1)+2)
    norm = (A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha    
    b0 = A*((A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw)
    b2 = A*((A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha)
    a1 = 2*((A-1) - (A+1)*cosw)
    a2 = (A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha
    b = np.array([b0, b1, b2])/norm
    a = np.array([1.0, a1/norm, a2/norm])
    return np.array([np.append(b, a)])

# IIR filter frequency response
# Input: filter coefficients/parameters
# Output: frequency response i.e. frequency (Hz), magnitude (dB) and phase (rad) arrays.
# In case of zero division warning: (old_settings =) np.seterr(divide='ignore')

def fr_iir(sr, b, a):
    f, z = signal.freqz(b, a, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z))

def fr_iirsos(sr, sos):
    f, z = signal.sosfreqz(sos, fs=sr)
    return f, 20*np.log10(abs(z)), np.unwrap(np.angle(z))
