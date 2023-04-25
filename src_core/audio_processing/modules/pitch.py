"""
Musical pitch calculations including note, midi, frequency and cent conversions.
For pianos' 88 notes only. There're 9 octaves (2 incomplete octaves at both ends), counting from 0.
The frequency calculation follows the standard frequencies of musical notes.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

Note = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

def note2midi(note_str, middle_C='C3'):
    if note_str[-2] == '-':
        octave = int(note_str[-2:])
        if len(note_str) == 3:
            note = note_str[0]
        elif len(note_str) == 4:
            note = note_str[0: 2]
        else:
            raise ValueError('The input note_str is not supported')
    else:
        octave = int(note_str[-1])
        if len(note_str) == 2:
            note = note_str[0]
        elif len(note_str) == 3:
            note = note_str[0: 2]
        else:
            raise ValueError('The input note_str is not supported')
    note = note.upper()
    note_idx = Note.index(note)
    octave_idx = int(octave) + 4 - int(middle_C[-1])
    return idx2midi(note_idx, octave_idx)

def midi2note(midi, middle_C='C3'):
    note_idx, octave_idx = midi2idx(midi)
    note = Note[note_idx]
    octave = octave_idx + int(middle_C[-1]) - 4
    return f'{note}{octave}'

def idx2midi(note_idx, octave_idx):
    return 12 + note_idx + 12*octave_idx

def midi2idx(midi):
    a, b = divmod(midi-11, 12)
    return b-1, a # note_idx, octave_idx

def midi2midi_idx(midi):
    return midi - 21

def midi2freq(midi, middle_A=440):
    return middle_A*np.exp2((midi-69)/12)

def note2freq(note_str, middle_C='C3', middle_A=440):
    midi = note2midi(note_str, middle_C=middle_C)
    return midi2freq(midi, middle_A=middle_A)

def freq2cent(f1, f2):
    return 1200*np.log2(f2/f1)

def midi2cent(midi1, midi2):
    f1, f2 = midi2freq(midi1), midi2freq(midi2)
    return freq2cent(f1, f2)

def ratio2cent(ratio):
    return 1200*np.log2(ratio)

def ratio2semitone(ratio):
    return 12*np.log2(ratio)

def ratio2time(f_ratio):
    return np.reciprocal(f_ratio)

def cent2ratio(cent):
    return np.exp2(cent/1200)

def cent2time(cent):
    return ratio2time(cent2ratio(cent))

def time2ratio(t_ratio):
    return np.reciprocal(t_ratio)

def time2cent(t_ratio):
    return ratio2cent(time2ratio(t_ratio))

def semitone2ratio(semitone):
    return np.exp2(cent/12)

def pitch_shift_st(f, st):
    # pitch shift in semitones (st).
    return f*cent2ratio(st*100)

def pitch_shift_cent(f, cent):
    # pitch shift in cents.
    return f*cent2ratio(cent)

def interpolate_pitch(f, num):
    """
    Interpolate a frequency array.
    
    f: ndarray (Hz). The input frequency array, strictly increasing.
    num: int. Number of frequencies to interpolate between every 2 adjacent input frequencies.
    """
    from scipy.interpolate import CubicSpline
    size = (num + 1)*f.size - num
    n, n_f = np.arange(0, size), np.arange(0, size, num+1)
    return PchipInterpolator(n_f, f)(n)

def interpolate_pitch_midi(f, Midi_f, Midi):
    """
    Interpolate a frequency array given its midi array and the target midi array.

    f: ndarray (Hz). The input frequency array, strictly increasing.
    Midi_f: ndarray. The midi array corresponding to the f array.
    Midi: ndarray. The midi array to apply interpolation.
    """
    return PchipInterpolator(Midi_f, f)
