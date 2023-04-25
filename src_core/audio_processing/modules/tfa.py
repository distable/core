# Audio Time-Frequency Analysis

import numpy as np
from scipy import fft, signal

# discrete cosine transform
def dct_z(y, axis=-1, dct_type=2):
    return fft.dct(y, dct_type, axis=axis, norm='backward')

def dct_zf(y, sr, axis=-1, dct_type=2):
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return z, f

def dct_mf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, frequency
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def dct_mf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, frequency
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def dct_mp(y, axis=-1, dct_type=2):
    # returns: magnitude, sign
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.sign(z)
    return m, p

def dct_mpf(y, sr, axis=-1, dct_type=2):
    # returns: magnitude, sign, frequency
    z = fft.dct(y, dct_type, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.sign(z)
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def idct_z(z, axis=-1, dct_type=2):
    return fft.idct(z, dct_type, axis=axis, norm='backward')

def idct_mp(m, p, axis=-1, dct_type=2):
    return fft.idct(m*p, dct_type, axis=axis, norm='backward')

# real discrete Fourier transform
def rfft_z(y, axis=-1):
    return fft.rfft(y, axis=axis, norm='backward')

def rfft_zf(y, sr, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return z, f

def rfft_m(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.rfft(y, axis=axis, norm='backward'))

def rfft_mf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    m = np.abs(fft.rfft(y, axis=axis, norm='backward'))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, f

def rfft_mp(y, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    return m, p

def rfft_mpf(y, sr, axis=-1):
    z = fft.rfft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    f = fft.rfftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def irfft_z(z, time_size_is_even=True, axis=-1):
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2-1, axis=axis, norm='backward')

def irfft_mp(m, p, time_size_is_even=True, axis=-1):
    z = np.empty(m.shape, dtype=np.complex128)
    z.real, z.imag = m*np.cos(p), m*np.sin(p)
    if time_size_is_even:
        return fft.irfft(z, axis=axis, norm='backward')
    else:
        return fft.irfft(z, n=z.shape[axis]*2-1, axis=axis, norm='backward')

# discrete Fourier transform
def fft_z(y, axis=-1):
    return fft.fft(y, axis=axis, norm='backward')

def fft_zf(y, sr, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return z, f

def fft_m(y, axis=-1):
    # This returns frequency magnitudes (real positive numbers) instead of complex numbers.
    return np.abs(fft.fft(y, axis=axis, norm='backward'))

def fft_mf(y, sr, axis=-1):
    # This returns frequency magnitudes and the frequencies consistent with sr.
    m = np.abs(fft.fft(y, axis=axis, norm='backward'))
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, f

def fft_mp(y, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    return m, p

def fft_mpf(y, sr, axis=-1):
    z = fft.fft(y, axis=axis, norm='backward')
    m = np.abs(z)
    p = np.unwrap(np.angle(z))
    f = fft.fftfreq(y.shape[axis], d=1/sr)
    return m, p, f

def ifft_z(z, axis=-1):
    return fft.ifft(z, axis=axis, norm='backward')

def ifft_mp(m, p, axis=-1):
    z = np.empty(m.shape, dtype=np.complex128)
    z.real, z.imag = m*np.cos(p), m*np.sin(p)
    return fft.ifft(z, axis=axis, norm='backward')

# Hilbert transform
def hilbert_z(y, axis=-1):
    # This returns the analytic signal of y.
    return signal.hilbert(y, axis=axis)

def hilbert_ap(y, axis=-1):
    # This returns (am, pm), the instaneous amplitude and phase arrays.
    z = signal.hilbert(y, axis=axis)
    return np.abs(z), np.unwrap(np.angle(z))

def hilbert_af(y, sr, axis=-1):
    # This returns (am, fm), the instaneous amplitude and frequency arrays.
    z = signal.hilbert(y, axis=axis)
    am = np.abs(z)
    fm = 0.5*sr*np.diff(np.unwrap(np.angle(z)), axis=axis)/np.pi
    return am, fm

def hilbert_apf(y, sr, axis=-1):
    # This returns (am, pm, fm).
    z = signal.hilbert(y, axis=axis)
    am = np.abs(z)
    pm = np.unwrap(np.angle(z))
    fm = 0.5*sr*np.diff(pm, axis=axis)/np.pi
    return am, pm, fm

def ihilbert_z(z):
    return np.real(z)

def ihilbert_ap(am, pm):
    return am*np.cos(pm)

# Short-time Fourier transform
class stft_class():
    # STFT and ISTFT using python's class.
    def __init__(self, sr, T=0.025, overlap=0.75, fft_ratio=1.0, win='blackmanharris', fft_type='m, p', GLA_n_iter=100, GLA_random_phase_type='mono'):
        """
        Parameters:
        sr: int (Hz). Sample rate, ususally 44100 or 48000.
        T: float (seconds). Time length of a each window. For 48000kHz, T=0.01067 means n=512.
        overlap: float (ratio between 0 and 1). Overlap ratio between each two adjacent windows.
        fft_ratio: float (ratio >= 1). The fft ratio relative to T.
        win: str. Please refer to scipy's window functions. Window functions like kaiser will require a tuple input including additional parameters. e.g. ('kaiser', 14.0)
        fft_type: str ('m', 'm, p', 'z' or 'zr, zi'). Please refer to the illustration of the returns of self.forward(). If fft_type=='m', istft will use the Griffin-Lim algorithm (GLA).
        GLA_n_iter: int. The iteration times for GLA.
        GLA_random_phase_type: str ('mono' or 'stereo'). Whether the starting random phases for GLA are different between 2 stereo channels.
        """
        self.sr, self.nperseg, self.noverlap, self.nfft = sr, int(sr*T), int(sr*T*overlap), int(sr*T*fft_ratio)
        self.nhop = self.nperseg - self.noverlap
        self.win, self.fft_type = signal.windows.get_window(win, self.nperseg, fftbins=True), fft_type
        self.GLA_n_iter, self.GLA_random_phase_type = GLA_n_iter, GLA_random_phase_type

    def fw(self, au):
        """
        Short-Time Fourier Transform

        Parameters:
        au: ndarray (dtype = float between -1 and 1). Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. 

        Returns:
        f: 1d array. As signal.stft returns.
        t: 1d array. As signal.stft returns.
        m: if self.fft_type='m'. The magnitudes array of shape (f.size, t.size) or (f.size, t.size, au.shape[-1]). PLEASE NOTE that the istft will use phases of a white noise!
        m, p: if self.fft_type='m, p'. The magnitudes array and phases array of shapes (f.size, t.size) or (f.size, t.size, au.shape[-1]). The phase range is [-pi, pi].
        z: if self.fft_type='z'. The complex array of shape (f.size, t.size) or (f.size, t.size, au.shape[-1]).
        zr, zi: if self.fft_type='zr, zi'. The complex array' real array and imaginary array of shapes (f.size, t.size) or (f.size, t.size, au.shape[-1]).
        """
        f, t, z = signal.stft(au, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
        z = z.swapaxes(1, -1)
        print(f'au.shape = {au.shape}')
        print(f'f.shape = {f.shape}')
        print(f't.shape = {t.shape}')
        print(f'z.shape = {z.shape}')
        if self.fft_type == 'm':
            m = np.abs(z)
            print(f'm.shape = {m.shape}')
            return f, t, m
        elif self.fft_type == 'm, p':
            m, p = np.abs(z), np.unwrap(np.angle(z))
            print(f'm.shape = {m.shape}')
            print(f'p.shape = {p.shape}')
            return f, t, m, p
        elif self.fft_type == 'z':
            return f, t, z
        elif self.fft_type == 'zr, zi':
            return f, t, z.real, z.imag
        else:
            raise ValueError('Parameter self.fft_type has to be "m", "m, p", "z" or "zr, zi".')

    def bw(self, m=None, p=None, z=None, zr=None, zi=None, nsample=None):
        """
        Inverse Short-Time Fourier Transform

        Parameters:
        in_tup: an ndarray or a tuple containing 2 ndarrays corresponding to self.fft_type. Please refer to the illustration of the returns of self.forward().
        
        Returns:
        au_re: ndarray. Audio array after inverse short-time fourier transform.
        """
        if self.fft_type == 'm, p':
            assert m is not None, f'm is None'
            assert p is not None, f'p is None'
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
        elif self.fft_type == 'm':
            assert m is not None, f'm is None'
            assert nsample is not None, f'nsample is None'
            p = self.get_random_phase(nsample, m.ndim)
            z = np.empty(m.shape, dtype=np.complex128)
            z.real, z.imag = m*np.cos(p), m*np.sin(p)
            for i in range(0, self.GLA_n_iter):
                t, au_re = signal.istft(z, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, time_axis=1, freq_axis=0)
                f, t, z = signal.stft(au_re, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
                z = z.swapaxes(1, -1)
                p = np.angle(z)
                z.real, z.imag = m*np.cos(p), m*np.sin(p)   
        elif self.fft_type == 'z':
            assert z is not None, f'z is None'
        elif self.fft_type == 'zr, zi':
            assert zr is not None, f'zr is None'
            assert zi is not None, f'zi is None'
            z = np.empty(in_tup[0].shape, dtype=np.complex128)
            z.real, z.imag = zr, zi
        else:
            raise ValueError('Parameter self.fft_type has to be "m", "m, p", "z" or "zr, zi".')
        t, au_re = signal.istft(z, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, time_axis=1, freq_axis=0)
        print(f'au_re.shape = {au_re.shape}')
        return au_re

    def re(self, au):
        """
        Reconstruct an audio array using stft and then istft. Please refer to the illustration of the returns of self.forward().

        Parameters:
        au: ndarray (dtype = float between -1 and 1). Need to have 1 or 2 dimensions like normal single-channel or multi-channel audio. 
        """          
        if self.fft_type == 'm, p':
            f, t, m, p = self.fw(au)
            au_re = self.bw(m=m, p=p)
        elif self.fft_type == 'm':
            # Using the Griffin-Lim algorithm.
            f, t, m = self.fw(au)
            nsample = au.shape[0]
            print(f'nsample = {nsample}')
            au_re = self.bw(m=m, nsample=nsample)
        elif self.fft_type == 'z':
            f, t, z = self.fw(au)
            au_re = self.bw(z=z)
        elif self.fft_type == 'zr, zi':
            f, t, zr, zi = self.fw(au)
            au_re = self.bw(zr=zr, zi=zi)
        else:
            raise ValueError('Parameter self.fft_type has to be "m", "m, p", "z" or "zr, zi".')
        return au_re        

    def get_random_phase(self, nsample, m_ndim):
        if m_ndim == 3:
            if self.GLA_random_phase_type == 'mono':
                noise = 0.5*np.random.uniform(-1, 1, nsample)
                noise = np.stack((noise, noise), axis=-1)
            elif self.GLA_random_phase_type == 'stereo':
                noise = 0.5*np.random.uniform(-1, 1, (nsample, 2))
            else:
                raise ValueError('self.GLA_random_phase_type != "mono" or "stereo"')
        elif m_ndim == 2:
            noise = 0.5*np.random.uniform(-1, 1, nsample)
        else:
            raise ValueError('m_ndim != 2 or 3')
        f_noise, t_noise, z_noise = signal.stft(noise, fs=self.sr, window=self.win, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, axis=0)
        z_noise = z_noise.swapaxes(1, -1)
        p_noise = np.angle(z_noise*np.exp(0.5*np.pi*1.0j))
        print(f'p_noise.shape = {p_noise.shape}')
        return p_noise    

# generate test signal
def get_sinewave(sr, du=1.0, f=440, phase=0, A=0.3, stereo=False, ls=None, ts=None):
    """
    Generate a pure sine wave for testing.
    sr: positive int (Hz). Sample rate.
    du: positive float (seconds). Duration of sinewave.
    f: positive float (Hz). Frequency.
    phase: float (rad angle). Initial phase.
    A: positive float (amp). Maxinum amplitude.
    ls: positive float (seconds). Duration of leading silence.
    ts: positive float (seconds). Duration of trailing silence.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    t = np.arange(0, size)/sr
    y = A*np.sin(2*np.pi*f*t + phase)
    if ls:
        y = np.append(np.zeros(int(ls*sr)), y)
    if ts:
        y = np.append(y, np.zeros(int(ts*sr)))
    if stereo:
        return np.broadcast_to(y.reshape((size, 1)), (size, 2))
    else:
        return y
    
def get_uniform_noise(sr, du=1.0, A=0.3, ls=None, ts=None, stereo=False):
    """
    Generate a uniform white noise signal for testing.
    sr: positive int (Hz). Sample rate.
    du: positive float (seconds). Duration of sinewave.
    A: positive float (amp). Maxinum amplitude.
    ls: positive float (seconds). Duration of leading silence.
    ts: positive float (seconds). Duration of trailing silence.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    if stereo == False: # mono
        noise = A*np.random.uniform(-1, 1, size)
        if ls:
            noise = np.append(np.zeros(int(sr*ls)), noise)
        if ts:
            noise = np.append(noise, np.zeros(int(sr*ts)))
    else:
        noise = A*np.random.uniform(-1, 1, (size, 2))    
        if ls:
            noise = np.append(np.zeros((int(sr*ls), 2)), noise, axis=0)
        if ts:
            noise = np.append(noise, np.zeros((int(sr*ts), 2)), axis=0)
    return noise

def get_gaussian_noise(sr, du=1.0, A=0.3, limit=3.0, ls=None, ts=None, stereo=False):
    """
    Generate a gaussian white noise signal for testing.
    sr: positive int (Hz). Sample rate.
    du: positive float (seconds). Duration of sinewave.
    A: positive float (amp). Maxinum amplitude.
    limit: positive float. Values out of range(-limit*std, limit*std) will be set to -limit*std or limit*std.
    ls: positive float (seconds). Duration of leading silence.
    ts: positive float (seconds). Duration of trailing silence.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    if stereo == False: # mono
        noise = np.random.normal(0.0, 1.0, size)*A/limit
        noise[noise < -A] = -A
        noise[noise > A] = A
        if ls:
            noise = np.append(np.zeros(int(sr*ls)), noise)
        if ts:
            noise = np.append(noise, np.zeros(int(sr*ts)))
    else:
        noise = np.random.normal(0.0, 1.0, (size, 2))*A/limit
        noise[noise < -A] = -A
        noise[noise > A] = A
        if ls:
            noise = np.append(np.zeros((int(sr*ls), 2)), noise, axis=0)
        if ts:
            noise = np.append(noise, np.zeros((int(sr*ts), 2)), axis=0)
    return noise

def get_silence(sr, du=1.0, stereo=False):
    """
    Generate a silence signal for testing.
    sr: int (Hz). Sample rate.
    du: float (seconds). Duration of sinewave.
    stereo: bool. If true, return a 2d array. If false, return a 1d array.
    """
    size = int(sr*du)
    if stereo == False:
        return np.zeros(size)
    else:
        return np.zeros((size, 2))

# pitch detection
def get_pitch_given(au, sr, du=None, given_freq=440, given_cent=100, cent_step=1):
    """
    Detect the pitch of audio (specifically piano single note) given a pitch, cent band and cent step, using discrete time fourier transform in limited frequency range.
    The computation will be quite slow since it does not use FFT, but it's much more accurate than signal.stft in terms of frequency resolution. 
    I've ensured the cpu and memory pressure won't be high by using for-loop.
    
    Parameters:
    au: ndarray (float between -1 and 1). The input audio.
    sr: int. Sample rate of audio.
    du: None or float (seconds). The duration of audio to be analyzed. If set to None, it will be the maxinum integar seconds available.
    given_freq: float (Hz). The central frequency around which pitch will be detected.
    given_cent: positive float (cent). Half of the cent band around the given frequency for pitch detection.
    cent_step: float (cent). The distance between Fourier transform's frequencies measured in cents, i.e. the resolution of frequencies.
    """
    if au.ndim == 1:
        pass
    elif au.ndim == 2:
        au = np.average(au, axis=-1)
    else:
        raise ValueError('The input audio array has no dimension, or over 2 dimensions which means it may be a framed audio.')
    if du is None:
        size = au.size
    else:
        size = int(sr*du)
        au = au[0: size]
    t = np.arange(0, size)/sr
    F = given_freq*np.exp2(np.arange(-given_cent, given_cent+1, cent_step)/1200)
    M = np.empty(0)
    for f in F:
        m = np.abs(np.average(au*np.exp(-2*np.pi*f*t*1.0j)))
        M = np.append(M, m)
    pitch = F[np.argmax(M)]
    print(f'{round(pitch, 2)}Hz is the detected pitch given {round(given_freq, 2)}Hz, {round(given_cent, 2)} cent band and {np.round(cent_step, 2)} cent step.')
    return pitch

# Frame audio
def frame_audio(au, sr, T=0.4, overlap=0.75, win='hamming'):
    """
    Parameters
    au: ndarray. Needs to have mono shape (samples_num, ) or multi-channel shape (samples_num, channels_num)
    sr: float (Hz). Sample rate of input audio array.
    T: float (seconds). Time length of each window.
    overlap: float, proportion. Proportion of overlapping between windows.
    win: str or tuple. The window to apply to every frame. No need to provide window size. Please refer to signal.get_windows.

    Returns
    au_f: ndarray. Framed audio with mono shape (window_num, samples) or multi-channel shape (window_num, samples_num, channels_num).
    """
    step, hop = int(sr*T), int(sr*T*(1-overlap))
    q1, q2 = divmod(au.shape[0], hop)
    q3 = step - hop - q2
    if q3 > 0:
        pad_shape = list(au.shape)
        pad_shape[0] = q3
        au = np.append(au, np.zeros(pad_shape), axis=0)
    au = np.expand_dims(au, axis=0)
    au_f = au[:, 0:step,...]
    for i in range(1, q1):
        au_f = np.append(au_f, au[:, i*hop:i*hop+step,...], axis=0)
    if win:
        win_shape = [1]*au.ndim
        win_shape[1] = step
        au_f *= signal.get_window(win, step).reshape(win_shape)
    return au_f

def unframe_audio(au_f, sr, T=0.4, overlap=0.75, win='hamming'):
    step, hop = int(sr*T), int(sr*T*(1-overlap))
    if win:
        win_shape = [1]*au_f.ndim
        win_shape[1] = step
        au_f /= signal.get_window(win, step).reshape(win_shape)
    au = au_f[0, :,...]
    for i in range(1, au_f.shape[0]):
        au = np.append(au, au_f[i, -hop:,...], axis=0)
    return au
