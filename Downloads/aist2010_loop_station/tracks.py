import wave

from globals import *

track1 = wave.open("audio/track1.wav", "wb")
track2 = wave.open("audio/track2.wav", "wb")
track3 = wave.open("audio/track3.wav", "wb")
track4 = wave.open("audio/track4.wav", "wb")
track5 = wave.open("audio/track5.wav", "wb")

track1.setnchannels(CHANNELS)
track1.setsampwidth(2)
track1.setframerate(SR)

track2.setnchannels(CHANNELS)
track2.setsampwidth(2)
track2.setframerate(SR)

track3.setnchannels(CHANNELS)
track3.setsampwidth(2)
track3.setframerate(SR)

track4.setnchannels(CHANNELS)
track4.setsampwidth(2)
track4.setframerate(SR)

track5.setnchannels(CHANNELS)
track5.setsampwidth(2)
track5.setframerate(SR)

segment1 = wave.open("audio/segment1.wav", "wb")
segment2 = wave.open("audio/segment2.wav", "wb")
segment3 = wave.open("audio/segment3.wav", "wb")
segment4 = wave.open("audio/segment4.wav", "wb")

segment1.setnchannels(CHANNELS)
segment1.setsampwidth(2)
segment1.setframerate(SR)

segment2.setnchannels(CHANNELS)
segment2.setsampwidth(2)
segment2.setframerate(SR)

segment3.setnchannels(CHANNELS)
segment3.setsampwidth(2)
segment3.setframerate(SR)

segment4.setnchannels(CHANNELS)
segment4.setsampwidth(2)
segment4.setframerate(SR)