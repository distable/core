"""
Separate or combine audio channels
'sepxtoy' means separating an x-channel audio array into y (x/y)-channel audio arrays.
'combx' means combining x audio arrays (channels) into 1 audio array.
"""
import numpy as np

def sep2to2(au):
    return au[:, 0], au[:, 1]

def sep4to4(au):
    return au[:, 0], au[:, 1], au[:, 2], au[:, 3]

def sep4to2(au):
    return au[:, 0:2], au[:, 2:4]

def comb2(au_0, au_1):
    return np.stack((au_0, au_1), axis=-1)

def comb4(au_0, au_1, au_2, au_3):
    return np.stack((au_0, au_1, au_2, au_3), axis=-1)

def sum2(au):
    return np.average(au, axis=-1)

