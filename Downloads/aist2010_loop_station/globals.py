import numpy as np
import pyaudio


SR = 44100 # sampling rate
CHUNK = 2048 # buffer size
FORMAT = pyaudio.paInt16 # 16-bit audio
CHANNELS = 1 # mono
SILENCE = np.zeros(CHUNK, dtype=np.int16)  # silence buffer
