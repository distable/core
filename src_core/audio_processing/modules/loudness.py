# Audio loudness calculation and normalization.
# The LUFS calculations here are all based on:
# ITU documentation: https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf
# EBU documentation: https://tech.ebu.ch/docs/tech/tech3341.pdf
# pyloudnorm by csteinmetz1: https://github.com/csteinmetz1/pyloudnorm
# loudness.py by BrechtDeMan: https://github.com/BrechtDeMan/loudness.py
# Special thanks to these authors!
# I just rewrote some codes to enable short-term and momentary loudness calculations and normalizations for more convinent batch processing of audio files.

import numpy as np
from scipy import signal

def amp2db(amp: float): # zero or positive amp value range between 0 and 1.
    return 20*np.log10(amp)

def db2amp(db: float): # zero or negative db value.
    return np.power(10, db/20)

def change_vol(au, db_change):
    return au*db2amp(db_change)

def check_clipping(au):
    assert np.amax(np.abs(au)) >= 1, 'Clipping has occurred.'
    
def recompare(au, au_re, sr=None):
    print('reconstruction comparison:')
    if sr:
        print(f'difference in length: {round((au_re.shape[0] - au.shape[0])/sr, 4)} seconds')
    print(f'max error: {round(amp2db(np.amax(np.abs(au_re[:au.shape[0],...] - au))), 4)} db')

def print_peak(au):
    au_abs = np.abs(au)
    peak_amp = np.amax(au_abs)
    peak_db = amp2db(peak_amp)
    print(f'peak_amp = {round(peak_amp, 5)}, peak_db = {round(peak_db, 2)}')

def get_peak(au):
    return np.amax(np.abs(au))

def get_peak_LR(au):
    peak_LR = np.amax(np.abs(au), axis=0)
    return peak_LR[0], peak_LR[1]

def norm_peak_mid(au, peak_db=-12.0):
    """
    normalize the peak amplitude of the mid channel of a stereo wav file under the normal -3db pan law.
    input array, output array, no read or write audio files.
    """
    au *= db2amp(peak_db)/np.amax(np.abs(np.average(au, axis=-1)))
    return au

def norm_peak_mono(au, peak_db=-12.0):
    """
    normalize the peak amplitude of a mono audio.
    input array, output array, no read or write audio files.
    """
    au *= db2amp(peak_db)/np.amax(np.abs(au))
    return au

class lufs_meter():
    # The momentary and integrated LUFS, defined as python class to pre-compute the window and prefilter coefficients.
    # Input audio need to have shape (nsample,) or (nsample, nchannel).
    def __init__(self, sr, T=0.4, overlap=0.75, threshold=-70.0, start_du=None):
        """
        Parameters
        sr: float (Hz). Sample rate for audio. If you want to process different sample rates, you need to set multiple meters.
        T: float (seconds). Time length of each window. You can change it to 3 to get the short-term lufs (Slufs).
        overlap: float (fraction). Proportion of overlapping between windows.
        threshold: float (LUFS or LKFS). If the LUFS is lower than this threshold, the meter will return -inf instead of very big negative numbers for runtime stability.
        start_du: float (seconds). The start seconds to only analyze.
        Only works for mono or stereo audio because I just summed all the channels and didn't calculate the different weights in case of a more-than-2-channels audio input.
        """
        
        # Calculate window.
        self.sr, self.T, self.overlap, self.threshold, self.du_start = sr, T, overlap, threshold, start_du
        self.step, self.hop = int(sr*T), int(sr*T*(1-overlap))
        self.z_threshold = np.power(10, (threshold+0.691)/10)
        if start_du is None:
            self.n_start = None
        else:
            self.n_start = int(sr*start_du)

        # Calculate prefilter (containing 2 IIR filters) coefficients.
        if sr == 48000:
            # coefficients in the ITU documentation.
            self.sos = np.array([[1.53512485958697, -2.69169618940638, 1.19839281085285, \
                                  1.0 , -1.69065929318241, 0.73248077421585], \
                                 [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]])
        elif sr == 44100:
            # coefficients calculation by BrechtDeMan.
            self.sos = np.array([[1.5308412300498355, -2.6509799951536985, 1.1690790799210682, \
                                  1.0, -1.6636551132560204, 0.7125954280732254], \
                                 [1.0, -2.0, 1.0, 1.0, -1.9891696736297957, 0.9891990357870394]])
        else:
            # coefficients calculation by BrechtDeMan. 
            # pre-filter 1
            f0 = 1681.9744509555319
            G  = 3.99984385397
            Q  = 0.7071752369554193
            K  = np.tan(np.pi * f0 / sr) 
            Vh = np.power(10.0, G / 20.0)
            Vb = np.power(Vh, 0.499666774155)
            a0_1_ = 1.0 + K / Q + K * K
            b0_1 = (Vh + Vb * K / Q + K * K) / a0_1_
            b1_1 = 2.0 * (K * K -  Vh) / a0_1_
            b2_1 = (Vh - Vb * K / Q + K * K) / a0_1_
            a0_1 = 1.0
            a1_1 = 2.0 * (K * K - 1.0) / a0_1_
            a2_1 = (1.0 - K / Q + K * K) / a0_1_
            # pre-filter 2
            f0 = 38.13547087613982
            Q  = 0.5003270373253953
            K  = np.tan(np.pi * f0 / sr)
            a0_2 = 1.0
            a1_2 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
            a2_2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
            b0_2 = 1.0
            b1_2 = -2.0
            b2_2 = 1.0

            self.sos = np.array([[b0_1, b1_1, b2_1, a0_1, a1_1, a2_1], \
                                 [b0_2, b1_2, b2_2, a0_2, a1_2, a2_2]])

    def get_mlufs(self, au):
        # Get the momentary LUFS array with size n_window.
        # The audio array will be padded zeros at the end to complete the last window.
        au = au[:self.n_start,...]
        q1, q2 = divmod(au.shape[0], self.hop)
        q3 = self.step - self.hop - q2
        if q3 > 0:
            pad_shape = list(au.shape)
            pad_shape[0] = q3
            au = np.append(au, np.zeros(pad_shape), axis=0)
        Mlufs = np.empty(0)
        for i in range(q1):
            au_f = au[i*self.hop: i*self.hop+self.step,...]
            au_f = signal.sosfilt(self.sos, au_f, axis=0)
            z = np.sum(np.average(np.square(au_f), axis=0))
            if z < self.z_threshold:
                mlufs = float('-inf')
            else:
                mlufs = -0.691 + 10*np.log10(z)
            Mlufs = np.append(Mlufs, mlufs)
        return Mlufs

    def get_mlufs_max(self, au):
        # Get the maxinum momentary LUFS value.
        return np.amax(self.get_mlufs(au))

    def get_ilufs(self, au):
        # Get the integrated LUFS value.
        # The audio array will be padded zeros at the end to complete the last window.
        Mlufs = self.get_mlufs(au)
        Z0 = Z[Mlufs > -70.0]
        if Z0.size == 0:
            ilufs = float('-inf')
        else:
            z1 = np.average(Z0)
            if z1 >= self.z_threshold:
                Z = Z[Mlufs > -0.691 + 10*np.log10(z1) - 10]
            else:
                pass 
        if Z.size == 0:
            ilufs = float('-inf')
        else:
            z2 = np.average(Z)
            if z2 >= self.z_threshold:
                ilufs = -0.691 + 10*np.log10(z2)
            else:
                ilufs = float('-inf')
        return ilufs

    def norm_mlufs_max(self, au, target=-20.0):
        # Normalize the maxinum momentary LUFS.
        return au*db2amp(target - self.get_mlufs_max(au))

    def norm_ilufs(self, au, target=-23.0):
        # Normalize the integrated LUFS.
        return au*db2amp(target - self.get_ilufs(au))

    def print_mlufs_max(self, au):
        # Print the maxinum momentary LUFS value.
        print(f'mlufs_max = {round(self.get_mlufs_max(au), 4)} LUFS')

    def print_ilufs(self, au):
        # Print the integrated LUFS value.
        print(f'ilufs = {round(self.get_ilufs(au), 4)} LUFS')
