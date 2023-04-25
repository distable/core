# fan455-audio-processing-codes
The audio processing codes I wrote for python. Special thanks to the authors of the python libraries and github repositories I've used. I've referred to them in the corresponding codes. 

Needed python libraries: numpy, scipy, matplotlib, soundfile.

Please note that in this repository, the shapes of all more-than-1-channel audio arrays follow the Soundfile, i.e. if audio.ndim=2, then audio.shape=(number_of_samples, number_of_channels), while some libraries like librosa and resampy use the reverse shape.
