import matplotlib.pyplot as plt
import tkinter as tk
import librosa
import wave
import sounddevice as sd

from PIL import ImageTk, Image
from pedalboard import *
from pydub import AudioSegment
from pydub.playback import play
import noisereduce as nr
from scipy.io import wavfile
from scipy import signal

from globals import *
import tracks as tr
from effects import *

global tempo, bpb, bpl, buffer_cnt

# default values
tempo = 120
bpb = 4
bpl = 2
buffer_cnt = int((SR / CHUNK) * (60 / tempo) * bpb * bpl)

root = tk.Tk()
root.title("AIST2010 2024Autumn Loop Station")
root.iconbitmap("./image/icon.ico")
root.resizable(height=True, width=True)

#initialize librosa
def initialize_pitch_shift(sample_rate=22050, n_samples=2048):
    t = np.linspace(0, 1, n_samples, endpoint=False) 
    dummy = 0.5 * np.sin(2 * np.pi * 440 * t)
    _ = librosa.effects.pitch_shift(dummy, sr=sample_rate, n_steps=1)
initialize_pitch_shift()

latency = 150 # milliseconds
buffer_latency = int((SR / CHUNK) * (latency / 1000))

#piano melody generation
def key_pressed(event):
    try:
        match event.char:
            case "w":
                play(AudioSegment.from_wav(db4_file))
                tr.track5.writeframesraw(db4_data)
                note = "Db4"
            case "e":
                play(AudioSegment.from_wav(eb4_file))
                tr.track5.writeframesraw(eb4_data)
                note = "Eb4"
            case "t":
                play(AudioSegment.from_wav(gb4_file))
                tr.track5.writeframesraw(gb4_data)
                note = "Gb4"
            case "y":
                play(AudioSegment.from_wav(ab4_file))
                tr.track5.writeframesraw(ab4_data)
                note = "Ab4"
            case "u":
                play(AudioSegment.from_wav(bb4_file))
                tr.track5.writeframesraw(bb4_data)
                note = "Bb4"
            case "a":
                play(AudioSegment.from_wav(c4_file))
                tr.track5.writeframesraw(c4_data)
                note = "C4"
            case "s":
                play(AudioSegment.from_wav(d4_file))
                tr.track5.writeframesraw(d4_data)
                note = "D4"
            case "d":
                play(AudioSegment.from_wav(e4_file))
                tr.track5.writeframesraw(e4_data)
                note = "E4"
            case "f":
                play(AudioSegment.from_wav(f4_file))
                tr.track5.writeframesraw(f4_data)
                note = "F4"
            case "g":
                play(AudioSegment.from_wav(g4_file))
                tr.track5.writeframesraw(g4_data)
                note = "G4"
            case "h":
                play(AudioSegment.from_wav(a4_file))
                tr.track5.writeframesraw(a4_data)
                note = "A4"
            case "j":
                play(AudioSegment.from_wav(b4_file))
                tr.track5.writeframesraw(b4_data)
                note = "B4"
            case "k":
                play(AudioSegment.from_wav(c5_file))
                tr.track5.writeframesraw(c5_data)
                note = "C5"
        
        key_label.config(text=f"The key you are pressing: {event.char.upper()}\nThe note you are playing: {note}")
    except:
        key_label.config(text=f"The key you are pressing: {event.char.upper()}\nThe note you are playing: /")

root.bind("<KeyPress>", key_pressed)

class Loop:
    def __init__(self, is_metronome=False):
        global tempo, bpb, bpl, buffer_cnt

        buffer_cnt = int((SR / CHUNK) * (60 / tempo) * bpb * bpl)
        self.audio = np.zeros([buffer_cnt, CHUNK], dtype=np.int16)
        self.temp = np.zeros([CHUNK], dtype=np.int16)
        self.readp = 0 # read pointer
        self.writep = buffer_cnt - 1 - buffer_latency # write pointer
        self.is_recording = False
        self.is_playing = True
        self.segment = ""
        self.is_metronome = is_metronome  
        self.state = [0, 0, 0, 0, 0, 0, 0] # sync, noise, alien, cow, reverb, delay, fade

        self.vol = 70

        self.is_sync = False

        self.is_pitch_shift = False
        self.is_pitch_shift_down = False
        self.pitch_shift_steps = 0
        self.is_fade_out = False
        self.is_reverb = False
        self.reverb_effect = Pedalboard([Reverb(room_size=0.75)])
        self.is_delay = False
        self.delay_effect = Pedalboard([Delay(delay_seconds=0.5, mix=0.25)])
        self.is_noise = False
        self.is_reduced = False

    def init_loops(self):
        buffer_cnt = int((SR / CHUNK) * (60 / tempo) * bpb * bpl)
        self.audio = np.zeros([buffer_cnt, CHUNK], dtype=np.int16)
        self.temp = np.zeros([CHUNK], dtype=np.int16)
        self.readp = 0 # read pointer
        self.writep = buffer_cnt - 1 - buffer_latency # write pointer

    def _init_metronome(self):
        """Initialize metronome-specific audio clicks."""
        self.click = np.zeros(CHUNK, dtype=np.int16)
        self.click1st = np.zeros(CHUNK, dtype=np.int16)
        self.beat_buffer = int((SR / CHUNK) * (60 / tempo))
        self.metronome_len = 1
        self.beat_count = 0

        for i in range(CHUNK):
            self.click[i] = 1000 * ((i % 20) / 20 - 0.5)  # regular click
            self.click1st[i] = 10000 * ((i % 10) / 20 - 0.5)  # first beat click
        
    def metronome_play(self):
        if self.is_metronome:
            if (self.beat_count % self.beat_buffer) < self.metronome_len:
                if (self.beat_count // self.beat_buffer) % bpb == 0:
                    return self.click1st
                else:
                    return self.click
            else:
                return SILENCE
        return SILENCE

    def incp(self): # increase pointer
        self.writep = ((self.writep + 1) % buffer_cnt)
        self.readp = ((self.readp + 1) % buffer_cnt)

    def is_restarting(self): #restart loop
        return self.readp == 0

    def restart(self):
        self.readp = 0
        self.writep = buffer_cnt - buffer_latency

    def toggle_recording(self):
        self.is_recording = not self.is_recording

    def toggle_playing(self):
        self.is_playing = not self.is_playing

    def toggle_vol_change(self, vol):
        self.vol = vol

    def toggle_sync(self, filename):
        self.segment = filename
        if self.is_sync:
            print("Unsync!!")
            self.state[0] = 0
        else:
            print("Sync!!")
            self.state[0] = 1
        self.is_sync = not self.is_sync

    def track_sync(self, data):
        global bpb, bpl
        y, sr = librosa.load(self.segment, sr=SR)
        o_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        onset_frames_sample = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, units="samples")

        result = np.array([])

        for index, onset in enumerate(onset_frames):
            if o_env[onset] > 5: # boundary
                result = np.append(result, [index])
        if np.size(result) == 0:
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
            return int(onsets[0]/len(onset_frames_sample)*buffer_cnt)
        else:
            result = result.astype(int)
            first_onset = result[0]
            return int(first_onset/len(onset_frames_sample)*buffer_cnt)

    def toggle_noise(self, filename):
        if not self.is_reduced:
            rate, data = wavfile.read(filename)
            reduced_noise = nr.reduce_noise(data, rate)
            wavfile.write(filename, rate, reduced_noise)
            self.is_reduced = True
        if self.is_noise:
            self.state[1] = 0
            print("Noise Returned")
        else:
            print("Noise Reduced")
            self.state[1] = 1
        self.is_noise = not self.is_noise

    def noise_reduce(self, data):
        reduced_noise = nr.reduce_noise(y=data, sr=SR, chunk_size=CHUNK)
        return reduced_noise
    
    def toggle_shift_pitch(self): # for pitch shift up
        if self.is_pitch_shift:
            print("Alien Filter is Off")
            self.state[2] = 0
        else:
            print("Alien Filter is On")
            self.state[2] = 1
        self.is_pitch_shift = not self.is_pitch_shift
    
    def toggle_shift_pitch_down(self): # for pitch shift down
        if self.is_pitch_shift_down:
            print("Cow Filter is Off")
            self.state[3] = 0
        else:
            print("Cow Filter is On")
            self.state[3] = 1
        self.is_pitch_shift_down = not self.is_pitch_shift_down

    def pitch_shift(self, data):
        audio_float = data.astype(np.float32) / 32768.0  # normalize to [-1, 1]
        shifted_audio = librosa.effects.pitch_shift(audio_float, sr=SR, n_steps=self.pitch_shift_steps) 
        shifted_audio = np.clip(shifted_audio * 32768, -32768, 32767).astype(np.int16)
        return shifted_audio  
    
    def toggle_reverb(self):
        if self.is_reverb:
            print("Reverb Off")
            self.state[4] = 0
        else:
            print("Reverb On")
            self.state[4] = 1
        self.is_reverb = not self.is_reverb
    
    def reverb(self, data):
        data_float = data.astype(np.float32) / 32768.0 
        reverb_output = self.reverb_effect.process(data_float, SR, reset=False) 
        return np.clip(reverb_output * 32768, -32768, 32767).astype(np.int16)
    
    def toggle_delay(self):
        if self.is_delay:
            print("Delay Off")
            self.state[5] = 0
        else:
            print("Delay On")
            self.state[5] = 1
        self.is_delay = not self.is_delay

    def delay(self, data):
        data_float = data.astype(np.float32) / 32768.0 
        delay_output = self.delay_effect.process(data_float, SR, reset=False) 
        return np.clip(delay_output * 32768, -32768, 32767).astype(np.int16)

    def toggle_fade_out(self):
        if self.is_fade_out:
            print("Fade Out Filter is Off")
            self.state[6] = 0
        else:
            print("Fade Out Filter is On")
            self.state[6] = 1
        self.is_fade_out = not self.is_fade_out

    def read(self):
        temp = self.readp
        self.incp()
        
        if self.is_sync:
            temp += int(self.track_sync(self.audio[temp, :]))
        temp = temp % buffer_cnt
        result = self.audio[temp, :]

        if self.is_noise:
            result = self.noise_reduce(result)

        if self.is_pitch_shift: # shift up
            self.pitch_shift_steps = 12
            result = self.pitch_shift(result)

        if self.is_pitch_shift_down: # shift down
            self.pitch_shift_steps = -12
            result = self.pitch_shift(result)

        if self.is_reverb:
            result = self.reverb(result)
        
        if self.is_delay:
            result = self.delay(result)

        result = result.astype(np.float32)
        result = np.clip(result * int(self.vol) / 100, -32768, 32767).astype(np.int16) # adjust volume

        return result
        
    def dub(self, data):
        d = np.frombuffer(data, dtype=np.int16)
        for i in range(CHUNK):
            self.audio[self.writep, i] = self.audio[self.writep, i] * 0.9 + d[i] * 1.0

class CallBack:
    def __init__(self):
        global tempo, bpb, bpl, buffer_cnt

        self.loop1 = Loop()
        self.loop2 = Loop()
        self.loop3 = Loop()
        self.loop4 = Loop()
        self.metronome = Loop(is_metronome=True)

    loop1_plays = 0

    def restart_loops(self):
        if self.loop1.is_fade_out:
            audio_float = self.loop1.audio.astype(np.float32) / 32768.0
            fade_audio = audio_float / 2
            self.loop1.audio = np.clip(fade_audio * 32768, -32768, 32767).astype(np.int16)
        if self.loop2.is_fade_out:
            audio_float = self.loop2.audio.astype(np.float32) / 32768.0
            fade_audio = audio_float / 2
            self.loop2.audio = np.clip(fade_audio * 32768, -32768, 32767).astype(np.int16)
        if self.loop3.is_fade_out:
            audio_float = self.loop3.audio.astype(np.float32) / 32768.0
            fade_audio = audio_float / 2
            self.loop3.audio = np.clip(fade_audio * 32768, -32768, 32767).astype(np.int16)
        if self.loop4.is_fade_out:
            audio_float = self.loop4.audio.astype(np.float32) / 32768.0
            fade_audio = audio_float / 2
            self.loop4.audio = np.clip(fade_audio * 32768, -32768, 32767).astype(np.int16)
    
        self.loop1.restart()
        self.loop2.restart()
        self.loop3.restart()
        self.loop4.restart()

    def metronome_cb(self, in_data, frame_count, time_info, status):
        if cb.metronome.is_playing:
            audio = cb.metronome.metronome_play()
            cb.metronome.beat_count += 1
            return (audio.tobytes(), pyaudio.paContinue)
        else:
            return (SILENCE, pyaudio.paContinue)

    def loop1_cb(self, in_data, frame_count, time_info, status):

        if self.loop1.is_recording:
            self.loop1.dub(in_data)
            tr.track1.writeframesraw(np.frombuffer(in_data, dtype=np.int16))
            if tr.segment1._file is None: # open segment for rerecording
                tr.segment1 = wave.open("audio/segment1.wav", "wb")
                tr.segment1.setnchannels(CHANNELS)
                tr.segment1.setsampwidth(2)
                tr.segment1.setframerate(SR)
            tr.segment1.writeframesraw(np.frombuffer(in_data, dtype=np.int16))

        if self.loop1.is_restarting():
            self.loop1_plays = self.loop1_plays + 1
            self.restart_loops()

        if self.loop1.is_playing:
            self.loop1.temp = self.loop1.read()
            if not self.loop1.is_recording:
                tr.track1.writeframesraw(self.loop1.temp)
            return (self.loop1.temp, pyaudio.paContinue)
        else:
            self.loop1.incp()
            if not self.loop1.is_recording:
                tr.track1.writeframesraw(SILENCE)
            return (SILENCE, pyaudio.paContinue)

    def loop2_cb(self, in_data, frame_count, time_info, status):
        tr.track5.writeframesraw(SILENCE)

        if self.loop2.is_recording:
            self.loop2.dub(in_data)
            tr.track2.writeframesraw(np.frombuffer(in_data, dtype=np.int16))
            if tr.segment2._file is None: # open segment for rerecording
                tr.segment2 = wave.open("audio/segment2.wav", "wb")
                tr.segment2.setnchannels(CHANNELS)
                tr.segment2.setsampwidth(2)
                tr.segment2.setframerate(SR)
            tr.segment2.writeframesraw(np.frombuffer(in_data, dtype=np.int16))

        if self.loop2.is_playing:
            self.loop2.temp = self.loop2.read()
            if not self.loop2.is_recording:
                tr.track2.writeframesraw(self.loop2.temp)
            return (self.loop2.temp, pyaudio.paContinue)
        else:
            self.loop2.incp()
            if not self.loop2.is_recording:
                tr.track2.writeframesraw(SILENCE)
            return (SILENCE, pyaudio.paContinue)

    def loop3_cb(self, in_data, frame_count, time_info, status):

        if self.loop3.is_recording:
            self.loop3.dub(in_data)
            tr.track3.writeframesraw(np.frombuffer(in_data, dtype=np.int16))
            if tr.segment3._file is None: # open segment for rerecording
                tr.segment3 = wave.open("audio/segment3.wav", "wb")
                tr.segment3.setnchannels(CHANNELS)
                tr.segment3.setsampwidth(2)
                tr.segment3.setframerate(SR)
            tr.segment3.writeframesraw(np.frombuffer(in_data, dtype=np.int16))

        if self.loop3.is_playing:
            self.loop3.temp = self.loop3.read()
            if not self.loop3.is_recording:
                tr.track3.writeframesraw(self.loop3.temp)
            return (self.loop3.temp, pyaudio.paContinue)
        else:
            self.loop3.incp()
            if not self.loop3.is_recording:
                tr.track3.writeframesraw(SILENCE)
            return (SILENCE, pyaudio.paContinue)

    def loop4_cb(self, in_data, frame_count, time_info, status):

        if self.loop4.is_recording:
            self.loop4.dub(in_data)
            tr.track4.writeframesraw(np.frombuffer(in_data, dtype=np.int16))            
            if tr.segment4._file is None: # open segment for rerecording
                tr.segment4 = wave.open("audio/segment4.wav", "wb")
                tr.segment4.setnchannels(CHANNELS)
                tr.segment4.setsampwidth(2)
                tr.segment4.setframerate(SR)
            tr.segment4.writeframesraw(np.frombuffer(in_data, dtype=np.int16))

        if self.loop4.is_playing:
            self.loop4.temp = self.loop4.read()
            if not self.loop4.is_recording:
                tr.track4.writeframesraw(self.loop4.temp)
            return (self.loop4.temp, pyaudio.paContinue)
        else:
            self.loop4.incp()
            if not self.loop4.is_recording:
                tr.track4.writeframesraw(SILENCE)
            return (SILENCE, pyaudio.paContinue)

cb = CallBack()

# Stream

pa = pyaudio.PyAudio()

metronome_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SR,
    input=False,
    output=True,
    frames_per_buffer = CHUNK,
    start = False,
    stream_callback= cb.metronome_cb
)

loop1_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SR,
    input=True,
    output=True,
    frames_per_buffer = CHUNK,
    start = False,
    stream_callback = cb.loop1_cb
)

loop2_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SR,
    input=True,
    output=True,
    frames_per_buffer = CHUNK,
    start = False,
    stream_callback = cb.loop2_cb
)

loop3_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SR,
    input=True,
    output=True,
    frames_per_buffer = CHUNK,
    start = False,
    stream_callback = cb.loop3_cb
)

loop4_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SR,
    input=True,
    output=True,
    frames_per_buffer = CHUNK,
    start = False,
    stream_callback = cb.loop4_cb
)

def rec1():
    if (cb.loop1.is_recording is False): 
        cb.loop1.is_sync = False # turn off all filters before recording
        cb.loop1.is_noise = False
        cb.loop1.is_pitch_shift = False
        cb.loop1.is_pitch_shift_down = False
        cb.loop1.is_reverb = False
        cb.loop1.is_delay = False
        cb.loop1.is_fade_out = False
        cb.loop1.audio = np.zeros([buffer_cnt, CHUNK], dtype=np.int16) # clear audio
        cb.loop1.temp = np.zeros([CHUNK], dtype=np.int16)
        cb.loop1.is_recording = True
    else:
        cb.loop1.is_recording = False
        analyze_wave1() 
        if cb.loop1.state[0] == 1: # turn all filters back on
            cb.loop1.is_sync = True
        if cb.loop1.state[1] == 1:
            cb.loop1.is_noise = True
        if cb.loop1.state[2] == 1:
            cb.loop1.is_pitch_shift = True
        if cb.loop1.state[3] == 1:
            cb.loop1.is_pitch_shift_down = False
        if cb.loop1.state[4] == 1:
            cb.loop1.is_reverb = True
        if cb.loop1.state[5] == 1:
            cb.loop1.is_delay = True
        if cb.loop1.state[6] == 1:
            cb.loop1.is_fade_out =True
    cb.loop2.is_recording = False
    cb.loop3.is_recording = False
    cb.loop4.is_recording = False
    if (cb.metronome.is_playing and (cb.loop1.is_recording is False)): # terminate metronome lead-in 
        cb.metronome.is_playing = False
        metronome_stream.stop_stream() 

def rec2():
    if (cb.loop2.is_recording is False):
        cb.loop2.is_sync = False # turn off all filters before recording
        cb.loop2.is_noise = False
        cb.loop2.is_pitch_shift = False
        cb.loop2.is_pitch_shift_down = False
        cb.loop2.is_reverb = False
        cb.loop2.is_delay = False
        cb.loop2.is_fade_out = False
        cb.loop2.audio = np.zeros([buffer_cnt, CHUNK], dtype=np.int16) # clear audio
        cb.loop2.temp = np.zeros([CHUNK], dtype=np.int16)
        cb.loop2.is_recording = True
    else:
        cb.loop2.is_recording = False
        analyze_wave2()
        if cb.loop2.state[0] == 1: # turn all filters back on
            cb.loop2.is_sync = True
        if cb.loop2.state[1] == 1:
            cb.loop2.is_noise = True
        if cb.loop2.state[2] == 1:
            cb.loop2.is_pitch_shift = True
        if cb.loop2.state[3] == 1:
            cb.loop2.is_pitch_shift_down = False
        if cb.loop2.state[4] == 1:
            cb.loop2.is_reverb = True
        if cb.loop2.state[5] == 1:
            cb.loop2.is_delay = True
        if cb.loop2.state[6] == 1:
            cb.loop2.is_fade_out =True
    cb.loop1.is_recording = False
    cb.loop3.is_recording = False
    cb.loop4.is_recording = False
    if (cb.metronome.is_playing and (cb.loop2.is_recording is False)): # terminate metronome lead-in 
        cb.metronome.is_playing = False
        metronome_stream.stop_stream() 

def rec3():
    if (cb.loop3.is_recording is False):
        cb.loop3.is_sync = False # turn off all filters before recording
        cb.loop3.is_noise = False
        cb.loop3.is_pitch_shift = False
        cb.loop3.is_pitch_shift_down = False
        cb.loop3.is_reverb = False
        cb.loop3.is_delay = False
        cb.loop3.is_fade_out = False
        cb.loop3.audio = np.zeros([buffer_cnt, CHUNK], dtype=np.int16) # clear audio
        cb.loop3.temp = np.zeros([CHUNK], dtype=np.int16)
        cb.loop3.is_recording = True
    else:
        cb.loop3.is_recording = False
        analyze_wave3()
        if cb.loop3.state[0] == 1: # turn all filters back on
            cb.loop3.is_sync = True
        if cb.loop3.state[1] == 1:
            cb.loop3.is_noise = True
        if cb.loop3.state[2] == 1:
            cb.loop3.is_pitch_shift = True
        if cb.loop3.state[3] == 1:
            cb.loop3.is_pitch_shift_down = False
        if cb.loop3.state[4] == 1:
            cb.loop3.is_reverb = True
        if cb.loop3.state[5] == 1:
            cb.loop3.is_delay = True
        if cb.loop3.state[6] == 1:
            cb.loop3.is_fade_out =True
    cb.loop1.is_recording = False
    cb.loop2.is_recording = False
    cb.loop4.is_recording = False
    if (cb.metronome.is_playing and (cb.loop3.is_recording is False)): # terminate metronome lead-in 
        cb.metronome.is_playing = False
        metronome_stream.stop_stream() 

def rec4():
    if (cb.loop4.is_recording is False):
        cb.loop4.is_sync = False # turn off all filters before recording
        cb.loop4.is_noise = False
        cb.loop4.is_pitch_shift = False
        cb.loop4.is_pitch_shift_down = False
        cb.loop4.is_reverb = False
        cb.loop4.is_delay = False
        cb.loop4.is_fade_out = False
        cb.loop4.audio = np.zeros([buffer_cnt, CHUNK], dtype=np.int16) # clear audio
        cb.loop4.temp = np.zeros([CHUNK], dtype=np.int16)
        cb.loop4.is_recording = True
    else:
        cb.loop4.is_recording = False
        analyze_wave4()
        if cb.loop4.state[0] == 1: # turn all filters back on
            cb.loop4.is_sync = True
        if cb.loop4.state[1] == 1:
            cb.loop4.is_noise = True
        if cb.loop4.state[2] == 1:
            cb.loop4.is_pitch_shift = True
        if cb.loop4.state[3] == 1:
            cb.loop4.is_pitch_shift_down = False
        if cb.loop4.state[4] == 1:
            cb.loop4.is_reverb = True
        if cb.loop4.state[5] == 1:
            cb.loop4.is_delay = True
        if cb.loop4.state[6] == 1:
            cb.loop4.is_fade_out =True
    cb.loop1.is_recording = False
    cb.loop2.is_recording = False
    cb.loop3.is_recording = False
    if (cb.metronome.is_playing and (cb.loop4.is_recording is False)): # terminate metronome lead-in 
        cb.metronome.is_playing = False
        metronome_stream.stop_stream()  

def play1():
    cb.loop1.toggle_playing()

def play2():
    cb.loop2.toggle_playing()

def play3():
    cb.loop3.toggle_playing()

def play4():
    cb.loop4.toggle_playing()

def vol1_change(vol):
    cb.loop1.toggle_vol_change(vol)

def vol2_change(vol):
    cb.loop2.toggle_vol_change(vol)

def vol3_change(vol):
    cb.loop3.toggle_vol_change(vol)

def vol4_change(vol):
    cb.loop4.toggle_vol_change(vol)

def sync1():
    cb.loop1.toggle_sync("audio/segment1.wav")

def sync2():
    cb.loop2.toggle_sync("audio/segment2.wav")

def sync3():
    cb.loop3.toggle_sync("audio/segment3.wav")

def sync4():
    cb.loop4.toggle_sync("audio/segment4.wav")

def noise1():
    cb.loop1.toggle_noise("audio/segment1.wav")

def noise2():
    cb.loop2.toggle_noise("audio/segment2.wav")

def noise3():
    cb.loop3.toggle_noise("audio/segment3.wav")

def noise4():
    cb.loop4.toggle_noise("audio/segment4.wav")

def ali1():
    cb.loop1.toggle_shift_pitch()

def ali2():
    cb.loop2.toggle_shift_pitch()

def ali3():
    cb.loop3.toggle_shift_pitch()

def ali4():
    cb.loop4.toggle_shift_pitch()

def cow1():
    cb.loop1.toggle_shift_pitch_down()

def cow2():
    cb.loop2.toggle_shift_pitch_down()

def cow3():
    cb.loop3.toggle_shift_pitch_down()

def cow4():
    cb.loop4.toggle_shift_pitch_down()

def fo1():
    cb.loop1.toggle_fade_out()

def fo2():
    cb.loop2.toggle_fade_out()

def fo3():
    cb.loop3.toggle_fade_out()

def fo4():
    cb.loop4.toggle_fade_out()

def rvb1():
    cb.loop1.toggle_reverb()

def rvb2():
    cb.loop2.toggle_reverb()

def rvb3():
    cb.loop3.toggle_reverb()

def rvb4():
    cb.loop4.toggle_reverb()

def delay1():
    cb.loop1.toggle_delay()

def delay2():
    cb.loop2.toggle_delay()

def delay3():
    cb.loop3.toggle_delay()

def delay4():
    cb.loop4.toggle_delay()

def export():
    loop1_stream.close()
    loop2_stream.close()
    loop3_stream.close()
    loop4_stream.close()

    tr.track1.close()
    tr.track2.close()
    tr.track3.close()
    tr.track4.close()
    tr.track5.close()

    pa.terminate()

    audio5 = AudioSegment.from_wav("audio/track5.wav")

    audio1 = AudioSegment.from_wav("audio/track1.wav")
    audio2 = AudioSegment.from_wav("audio/track2.wav")
    stack12 = audio1.overlay(audio2)
    output = audio5.overlay(stack12)

    audio3 = AudioSegment.from_wav("audio/track3.wav")
    audio4 = AudioSegment.from_wav("audio/track4.wav")
    stack34 = audio3.overlay(audio4)
    output = output.overlay(stack34)

    output.export("audio/output.wav", format="wav")

    root.destroy()

def tempo_change(x):
    global tempo
    tempo = int(x.widget.get())
    print(f"Tempo changed to: {x.widget.get()}")

def bpb_change(x):
    global bpb
    bpb = int(x.widget.get())
    print(f"BPB changed to: {x.widget.get()}")

def bpl_change(x):
    global bpl
    bpl = int(x.widget.get())
    print(f"BPL changed to: {x.widget.get()}")

def start():
    global tempo, bpb, bpl, buffer_cnt

    beat_buffer = int((SR / CHUNK) * (60 / tempo))
    buffer_cnt = beat_buffer * bpb * bpl

    loop1_stream.start_stream()
    loop2_stream.start_stream()
    loop3_stream.start_stream()
    loop4_stream.start_stream()
    metronome_stream.start_stream()
    cb.loop1.init_loops()
    cb.loop2.init_loops()
    cb.loop3.init_loops()
    cb.loop4.init_loops()
    cb.metronome._init_metronome() # update metronome with new tempo, bpb, bpl
    cb.metronome.is_playing = True

# main frame
main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill=tk.BOTH)

# rows
input_row = tk.Frame(main_frame)
input_row.pack(fill=tk.X)

control_row = tk.Frame(main_frame)
control_row.pack(expand=True, fill=tk.X)

melody_row = tk.Frame(main_frame)
melody_row.pack(fill=tk.X)

track5_label = tk.Label(melody_row, text="Try to play the melody with your keyboard~", anchor="w", font=20)
track5_label.pack(side=tk.LEFT, padx=10)

piano_img = Image.open("./image/piano.jpg")
piano_img = piano_img.resize((300, 150))
piano_img = ImageTk.PhotoImage(piano_img)

piano_label = tk.Label(melody_row, image = piano_img)
piano_label.pack(side=tk.LEFT)

key_label = tk.Label(melody_row, text="The key you are pressing: /\nThe note you are playing: /", anchor="w", font=20)
key_label.pack(side=tk.LEFT, padx=10)

# columns
rec_col = tk.LabelFrame(control_row, borderwidth=2, text="Recording", relief="groove")
rec_col.pack(side=tk.LEFT, fill=tk.BOTH)

play_col = tk.LabelFrame(control_row, borderwidth=2, text="Playing", relief="groove")
play_col.pack(side=tk.LEFT, fill=tk.BOTH)

visual_col = tk.LabelFrame(control_row, borderwidth=2, text="Waveform", relief="groove")
visual_col.pack(side=tk.LEFT, expand=False, fill=tk.BOTH)

volume_col = tk.LabelFrame(control_row, borderwidth=2, text="Adjust Volume", relief="groove")
volume_col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

sync_col = tk.LabelFrame(control_row, borderwidth=2, text="Synchronize", relief="groove")
sync_col.pack(side=tk.LEFT, fill=tk.BOTH)

noise_col = tk.LabelFrame(control_row, borderwidth=2, text="Noise Reduction", relief="groove")
noise_col.pack(side=tk.LEFT, fill=tk.BOTH)

ali_col = tk.LabelFrame(control_row, borderwidth=2, text="Pitch Up", relief="groove")
ali_col.pack(side=tk.LEFT, fill=tk.BOTH)

cow_col = tk.LabelFrame(control_row, borderwidth=2, text="Pitch Down", relief="groove")
cow_col.pack(side=tk.LEFT, fill=tk.BOTH)

rvb_col = tk.LabelFrame(control_row, borderwidth=2, text="Reverb", relief="groove")
rvb_col.pack(side=tk.LEFT, fill=tk.BOTH)

delay_col = tk.LabelFrame(control_row, borderwidth=2, text="Delay", relief="groove")
delay_col.pack(side=tk.LEFT, fill=tk.BOTH)

fo_col = tk.LabelFrame(control_row, borderwidth=2, text="Fade Out", relief="groove")
fo_col.pack(side=tk.LEFT, fill=tk.BOTH)


# get input

tempo_label = tk.Label(input_row, text="Tempo: ")
tempo_input = tk.Entry(input_row, width=20)
tempo_input.insert(0, "120")
tempo_input.bind("<Return>", tempo_change)

bpb_label = tk.Label(input_row, text="Beats per Bar: ")
bpb_input = tk.Entry(input_row, width=20)
bpb_input.insert(0, "4")
bpb_input.bind("<Return>", bpb_change)

bpl_label = tk.Label(input_row, text="Bars per Loop: ")
bpl_input = tk.Entry(input_row, width=20)
bpl_input.insert(0, "2")
bpl_input.bind("<Return>", bpl_change)

# create buttons
start_btn = tk.Button(input_row, text="START", command=lambda:start())

# export button
export_btn = tk.Button(input_row, text="EXPORT", command=export)

tempo_label.grid(row=0, column=0, padx=5)
tempo_input.grid(row=0, column=1, padx=5)
bpb_label.grid(row=0, column=2, padx=5)
bpb_input.grid(row=0, column=3, padx=5)
bpl_label.grid(row=0, column=4, padx=5)
bpl_input.grid(row=0, column=5, padx=5)
start_btn.grid(row=0, column=6, padx=5)
export_btn.grid(row=0, column=7, padx=5)


btn_size = (50, 50)

# start / stop recording button
rec_img = Image.open("./image/start_rec.png")
rec_img = rec_img.resize(btn_size)
rec_img = ImageTk.PhotoImage(rec_img)

nrec_img = Image.open("./image/stop_rec.png")
nrec_img = nrec_img.resize(btn_size)
nrec_img = ImageTk.PhotoImage(nrec_img)

def toggle_rec1(btn):
    if btn["image"] == str(rec_img): # stop rec
        btn.config(image=nrec_img)
        tr.segment1.close()
    else:
        btn.config(image=rec_img)
    rec1()

def toggle_rec2(btn):
    if btn["image"] == str(rec_img): # stop rec
        btn.config(image=nrec_img)
        tr.segment2.close()
    else:
        btn.config(image=rec_img)
    rec2()

def toggle_rec3(btn):
    if btn["image"] == str(rec_img): # stop rec
        btn.config(image=nrec_img)
        tr.segment3.close()
    else:
        btn.config(image=rec_img)
    rec3()

def toggle_rec4(btn):
    if btn["image"] == str(rec_img): # stop rec
        btn.config(image=nrec_img)
        tr.segment4.close()
    else:
        btn.config(image=rec_img)
    rec4()

rec1_btn = tk.Button(rec_col, image=nrec_img, borderwidth=0, command=lambda: toggle_rec1(rec1_btn))
rec2_btn = tk.Button(rec_col, image=nrec_img, borderwidth=0, command=lambda: toggle_rec2(rec2_btn))
rec3_btn = tk.Button(rec_col, image=nrec_img, borderwidth=0, command=lambda: toggle_rec3(rec3_btn))
rec4_btn = tk.Button(rec_col, image=nrec_img, borderwidth=0, command=lambda: toggle_rec4(rec4_btn))

rec1_btn.grid(row=0, column=0, padx=10, pady=25)
rec2_btn.grid(row=1, column=0, padx=10, pady=25)
rec3_btn.grid(row=2, column=0, padx=10, pady=25)
rec4_btn.grid(row=3, column=0, padx=10, pady=25)

# play / pause looping button
play_img = Image.open("./image/play.png")
play_img = play_img.resize(btn_size)
play_img = ImageTk.PhotoImage(play_img)

pause_img = Image.open("./image/pause.png")
pause_img = pause_img.resize(btn_size)
pause_img = ImageTk.PhotoImage(pause_img)

def toggle_play1(btn):
    if btn["image"] == str(play_img):
        btn.config(image=pause_img)
    else:
        btn.config(image=play_img)
    play1()

def toggle_play2(btn):
    if btn["image"] == str(play_img):
        btn.config(image=pause_img)
    else:
        btn.config(image=play_img)
    play2()

def toggle_play3(btn):
    if btn["image"] == str(play_img):
        btn.config(image=pause_img)
    else:
        btn.config(image=play_img)
    play3()

def toggle_play4(btn):
    if btn["image"] == str(play_img):
        btn.config(image=pause_img)
    else:
        btn.config(image=play_img)
    play4()

play1_btn = tk.Button(play_col, image=pause_img, borderwidth=0, command=lambda: toggle_play1(play1_btn))
play2_btn = tk.Button(play_col, image=pause_img, borderwidth=0, command=lambda: toggle_play2(play2_btn))
play3_btn = tk.Button(play_col, image=pause_img, borderwidth=0, command=lambda: toggle_play3(play3_btn))
play4_btn = tk.Button(play_col, image=pause_img, borderwidth=0, command=lambda: toggle_play4(play4_btn))

play1_btn.grid(row=0, column=0, padx=10, pady=25)
play2_btn.grid(row=1, column=0, padx=10, pady=25)
play3_btn.grid(row=2, column=0, padx=10, pady=25)
play4_btn.grid(row=3, column=0, padx=10, pady=25)

def analyze_wave1():
    y1, sr = librosa.load("audio/segment1.wav")
    img = librosa.display.waveshow(y1, sr=sr)
    plt.axis("off")
    plt.savefig("./image/waveforms/wave1.jpg")
    plt.close()

    img = Image.open("./image/waveforms/wave1.jpg").resize((300, 100))
    photo = ImageTk.PhotoImage(img)
    wave1_btn.config(image=photo)
    wave1_btn.image = photo

def analyze_wave2():
    y1, sr = librosa.load("audio/segment2.wav")
    img = librosa.display.waveshow(y1, sr=sr)
    plt.axis("off")
    plt.savefig("./image/waveforms/wave2.jpg")
    plt.close()
    img = Image.open("./image/waveforms/wave2.jpg").resize((300, 100))
    photo = ImageTk.PhotoImage(img)
    wave2_btn.config(image=photo)
    wave2_btn.image = photo

def analyze_wave3():
    y1, sr = librosa.load("audio/segment3.wav")
    img = librosa.display.waveshow(y1, sr=sr)
    plt.axis("off")
    plt.savefig("./image/waveforms/wave3.jpg")
    plt.close()
    img = Image.open("./image/waveforms/wave3.jpg").resize((300, 100))
    photo = ImageTk.PhotoImage(img)
    wave3_btn.config(image=photo)
    wave3_btn.image = photo

def analyze_wave4():
    y1, sr = librosa.load("audio/segment4.wav")
    img = librosa.display.waveshow(y1, sr=sr)
    plt.axis("off")
    plt.savefig("./image/waveforms/wave4.jpg")
    plt.close()
    img = Image.open("./image/waveforms/wave4.jpg").resize((300, 100))
    photo = ImageTk.PhotoImage(img)
    wave4_btn.config(image=photo)
    wave4_btn.image = photo

wave1_img = Image.open("./image/waveforms/wave_init.jpg").resize((300, 100))
wave1_img = ImageTk.PhotoImage(wave1_img)
wave1_btn = tk.Button(visual_col, image=wave1_img, borderwidth=0, command=analyze_wave1)
wave1_btn.grid(row=0, column=0, padx=10)

wave2_img = Image.open("./image/waveforms/wave_init.jpg").resize((300, 100))
wave2_img = ImageTk.PhotoImage(wave2_img)
wave2_btn = tk.Button(visual_col, image=wave2_img, borderwidth=0, command=analyze_wave2)
wave2_btn.grid(row=1, column=0, padx=10)

wave3_img = Image.open("./image/waveforms/wave_init.jpg").resize((300, 100))
wave3_img = ImageTk.PhotoImage(wave3_img)
wave3_btn = tk.Button(visual_col, image=wave3_img, borderwidth=0, command=analyze_wave3)
wave3_btn.grid(row=2, column=0, padx=10)

wave4_img = Image.open("./image/waveforms/wave_init.jpg").resize((300, 100))
wave4_img = ImageTk.PhotoImage(wave4_img)
wave4_btn = tk.Button(visual_col, image=wave4_img, borderwidth=0, command=analyze_wave4)
wave4_btn.grid(row=3, column=0, padx=10)

# volume control
vol1_scale = tk.Scale(volume_col, from_=0, to=100, orient=tk.HORIZONTAL, command=vol1_change)
vol1_scale.set(70)
vol1_scale.pack(fill=tk.X, padx=5, pady=30)

vol2_scale = tk.Scale(volume_col, from_=0, to=100, orient=tk.HORIZONTAL, command=vol2_change)
vol2_scale.set(70)
vol2_scale.pack(fill=tk.X, padx=5, pady=30)

vol3_scale = tk.Scale(volume_col, from_=0, to=100, orient=tk.HORIZONTAL, command=vol3_change)
vol3_scale.set(70)
vol3_scale.pack(fill=tk.X, padx=5, pady=30)

vol4_scale = tk.Scale(volume_col, from_=0, to=100, orient=tk.HORIZONTAL, command=vol4_change)
vol4_scale.set(70)
vol4_scale.pack(fill=tk.X, padx=5, pady=30)

def toggle_sync1(btn: tk.Button):
    sync1()

def toggle_sync2(btn: tk.Button):
    sync2()

def toggle_sync3(btn: tk.Button):
    sync3()

def toggle_sync4(btn: tk.Button):
    sync4()

# sync button
sync1_btn = tk.Button(sync_col, text="Sync", command=lambda: toggle_sync1(sync1_btn))
sync1_btn.pack(padx=10, pady=40)
sync2_btn = tk.Button(sync_col, text="Sync", command=lambda: toggle_sync2(sync2_btn))
sync2_btn.pack(padx=10, pady=40)
sync3_btn = tk.Button(sync_col, text="Sync", command=lambda: toggle_sync3(sync3_btn))
sync3_btn.pack(padx=10, pady=40)
sync4_btn = tk.Button(sync_col, text="Sync", command=lambda: toggle_sync4(sync4_btn))
sync4_btn.pack(padx=10, pady=30)

# noise removal
noise_on_img = Image.open("./image/noise_on.png")
noise_on_img = noise_on_img.resize(btn_size)
noise_on_img = ImageTk.PhotoImage(noise_on_img)

noise_off_img = Image.open("./image/noise_off.png")
noise_off_img = noise_off_img.resize(btn_size)
noise_off_img = ImageTk.PhotoImage(noise_off_img)

def toggle_noise1(btn: tk.Button):
    if btn["image"] == str(noise_on_img):
        btn.config(image=noise_off_img)
    else:
        btn.config(image=noise_on_img)
    noise1()

def toggle_noise2(btn: tk.Button):
    if btn["image"] == str(noise_on_img):
        btn.config(image=noise_off_img)
    else:
        btn.config(image=noise_on_img)
    noise2()

def toggle_noise3(btn: tk.Button):
    if btn["image"] == str(noise_on_img):
        btn.config(image=noise_off_img)
    else:
        btn.config(image=noise_on_img)
    noise3()

def toggle_noise4(btn: tk.Button):
    if btn["image"] == str(noise_on_img):
        btn.config(image=noise_off_img)
    else:
        btn.config(image=noise_on_img)
    noise4()
    
noise1_btn = tk.Button(noise_col, image=noise_off_img, borderwidth=0, command=lambda: toggle_noise1(noise1_btn))
noise2_btn = tk.Button(noise_col, image=noise_off_img, borderwidth=0, command=lambda: toggle_noise2(noise2_btn))
noise3_btn = tk.Button(noise_col, image=noise_off_img, borderwidth=0, command=lambda: toggle_noise3(noise3_btn))
noise4_btn = tk.Button(noise_col, image=noise_off_img, borderwidth=0, command=lambda: toggle_noise4(noise4_btn))

noise1_btn.grid(row=0, column=0, padx=10, pady=25)
noise2_btn.grid(row=1, column=0, padx=10, pady=25)
noise3_btn.grid(row=2, column=0, padx=10, pady=25)
noise4_btn.grid(row=3, column=0, padx=10, pady=25)

# filter 1 (Alien - picth shifting - up)
alien_on_img = Image.open("./image/alien_on.png")
alien_on_img = alien_on_img.resize(btn_size)
alien_on_img = ImageTk.PhotoImage(alien_on_img)

alien_off_img = Image.open("./image/alien_off.png")
alien_off_img = alien_off_img.resize(btn_size)
alien_off_img = ImageTk.PhotoImage(alien_off_img)

def toggle_ali1(btn):
    if btn["image"] == str(alien_on_img):
        btn.config(image=alien_off_img)
    else:
        btn.config(image=alien_on_img)
    ali1()

def toggle_ali2(btn):
    if btn["image"] == str(alien_on_img):
        btn.config(image=alien_off_img)
    else:
        btn.config(image=alien_on_img)
    ali2()

def toggle_ali3(btn):
    if btn["image"] == str(alien_on_img):
        btn.config(image=alien_off_img)
    else:
        btn.config(image=alien_on_img)
    ali3()

def toggle_ali4(btn):
    if btn["image"] == str(alien_on_img):
        btn.config(image=alien_off_img)
    else:
        btn.config(image=alien_on_img)
    ali4()

ali1_btn = tk.Button(ali_col, image=alien_off_img, borderwidth=0, command=lambda: toggle_ali1(ali1_btn))
ali2_btn = tk.Button(ali_col, image=alien_off_img, borderwidth=0, command=lambda: toggle_ali2(ali2_btn))
ali3_btn = tk.Button(ali_col, image=alien_off_img, borderwidth=0, command=lambda: toggle_ali3(ali3_btn))
ali4_btn = tk.Button(ali_col, image=alien_off_img, borderwidth=0, command=lambda: toggle_ali4(ali4_btn))

ali1_btn.grid(row=0, column=0, padx=10, pady=25)
ali2_btn.grid(row=1, column=0, padx=10, pady=25)
ali3_btn.grid(row=2, column=0, padx=10, pady=25)
ali4_btn.grid(row=3, column=0, padx=10, pady=25)

# filter 2 (Cow - picth shifting - down)
cow_on_img = Image.open("./image/cow_on.png")
cow_on_img = cow_on_img.resize(btn_size)
cow_on_img = ImageTk.PhotoImage(cow_on_img)

cow_off_img = Image.open("./image/cow_off.png")
cow_off_img = cow_off_img.resize(btn_size)
cow_off_img = ImageTk.PhotoImage(cow_off_img)

def toggle_cow1(btn):
    if btn["image"] == str(cow_on_img):
        btn.config(image=cow_off_img)
    else:
        btn.config(image=cow_on_img)
    cow1()

def toggle_cow2(btn):
    if btn["image"] == str(cow_on_img):
        btn.config(image=cow_off_img)
    else:
        btn.config(image=cow_on_img)
    cow2()

def toggle_cow3(btn):
    if btn["image"] == str(cow_on_img):
        btn.config(image=cow_off_img)
    else:
        btn.config(image=cow_on_img)
    cow3()

def toggle_cow4(btn):
    if btn["image"] == str(cow_on_img):
        btn.config(image=cow_off_img)
    else:
        btn.config(image=cow_on_img)
    cow4()

cow1_btn = tk.Button(cow_col, image=cow_off_img, borderwidth=0, command=lambda: toggle_cow1(cow1_btn))
cow2_btn = tk.Button(cow_col, image=cow_off_img, borderwidth=0, command=lambda: toggle_cow2(cow2_btn))
cow3_btn = tk.Button(cow_col, image=cow_off_img, borderwidth=0, command=lambda: toggle_cow3(cow3_btn))
cow4_btn = tk.Button(cow_col, image=cow_off_img, borderwidth=0, command=lambda: toggle_cow4(cow4_btn))

cow1_btn.grid(row=0, column=0, padx=10, pady=25)
cow2_btn.grid(row=1, column=0, padx=10, pady=25)
cow3_btn.grid(row=2, column=0, padx=10, pady=25)
cow4_btn.grid(row=3, column=0, padx=10, pady=25)

# filter 3 (Reverb)
reverb_on_img = Image.open("./image/reverb_on.png")
reverb_on_img = reverb_on_img.resize(btn_size)
reverb_on_img = ImageTk.PhotoImage(reverb_on_img)

reverb_off_img = Image.open("./image/reverb_off.png")
reverb_off_img = reverb_off_img.resize(btn_size)
reverb_off_img = ImageTk.PhotoImage(reverb_off_img)

def toggle_rvb1(btn):
    if btn["image"] == str(reverb_on_img):
        btn.config(image=reverb_off_img)
    else:
        btn.config(image=reverb_on_img)
    rvb1()

def toggle_rvb2(btn):
    if btn["image"] == str(reverb_on_img):
        btn.config(image=reverb_off_img)
    else:
        btn.config(image=reverb_on_img)
    rvb2()

def toggle_rvb3(btn):
    if btn["image"] == str(reverb_on_img):
        btn.config(image=reverb_off_img)
    else:
        btn.config(image=reverb_on_img)
    rvb3()

def toggle_rvb4(btn):
    if btn["image"] == str(reverb_on_img):
        btn.config(image=reverb_off_img)
    else:
        btn.config(image=reverb_on_img)
    rvb4()

rvb1_btn = tk.Button(rvb_col, image=reverb_off_img, borderwidth=0, command=lambda: toggle_rvb1(rvb1_btn))
rvb2_btn = tk.Button(rvb_col, image=reverb_off_img, borderwidth=0, command=lambda: toggle_rvb2(rvb2_btn))
rvb3_btn = tk.Button(rvb_col, image=reverb_off_img, borderwidth=0, command=lambda: toggle_rvb3(rvb3_btn))
rvb4_btn = tk.Button(rvb_col, image=reverb_off_img, borderwidth=0, command=lambda: toggle_rvb4(rvb4_btn))

rvb1_btn.grid(row=0, column=0, padx=10, pady=25)
rvb2_btn.grid(row=1, column=0, padx=10, pady=25)
rvb3_btn.grid(row=2, column=0, padx=10, pady=25)
rvb4_btn.grid(row=3, column=0, padx=10, pady=25)

# filter 4 (Delay)
delay_on_img = Image.open("./image/delay_on.png")
delay_on_img = delay_on_img.resize(btn_size)
delay_on_img = ImageTk.PhotoImage(delay_on_img)

delay_off_img = Image.open("./image/delay_off.png")
delay_off_img = delay_off_img.resize(btn_size)
delay_off_img = ImageTk.PhotoImage(delay_off_img)

def toggle_delay1(btn):
    if btn["image"] == str(delay_on_img):
        btn.config(image=delay_off_img)
    else:
        btn.config(image=delay_on_img)
    delay1()

def toggle_delay2(btn):
    if btn["image"] == str(delay_on_img):
        btn.config(image=delay_off_img)
    else:
        btn.config(image=delay_on_img)
    delay2()

def toggle_delay3(btn):
    if btn["image"] == str(delay_on_img):
        btn.config(image=delay_off_img)
    else:
        btn.config(image=delay_on_img)
    delay3()

def toggle_delay4(btn):
    if btn["image"] == str(delay_on_img):
        btn.config(image=delay_off_img)
    else:
        btn.config(image=delay_on_img)
    delay4()

delay1_btn = tk.Button(delay_col, image=delay_off_img, borderwidth=0, command=lambda: toggle_delay1(delay1_btn))
delay2_btn = tk.Button(delay_col, image=delay_off_img, borderwidth=0, command=lambda: toggle_delay2(delay2_btn))
delay3_btn = tk.Button(delay_col, image=delay_off_img, borderwidth=0, command=lambda: toggle_delay3(delay3_btn))
delay4_btn = tk.Button(delay_col, image=delay_off_img, borderwidth=0, command=lambda: toggle_delay4(delay4_btn))

delay1_btn.grid(row=0, column=0, padx=10, pady=25)
delay2_btn.grid(row=1, column=0, padx=10, pady=25)
delay3_btn.grid(row=2, column=0, padx=10, pady=25)
delay4_btn.grid(row=3, column=0, padx=10, pady=25)

# filter 5 (Fade Out)
fo_on_img = Image.open("./image/fade_out_on.png")
fo_on_img = fo_on_img.resize(btn_size)
fo_on_img = ImageTk.PhotoImage(fo_on_img)

fo_off_img = Image.open("./image/fade_out_off.png")
fo_off_img = fo_off_img.resize(btn_size)
fo_off_img = ImageTk.PhotoImage(fo_off_img)

def toggle_fo1(btn):
    if btn["image"] == str(fo_on_img):
        btn.config(image=fo_off_img)
    else:
        btn.config(image=fo_on_img)
    fo1()

def toggle_fo2(btn):
    if btn["image"] == str(fo_on_img):
        btn.config(image=fo_off_img)
    else:
        btn.config(image=fo_on_img)
    fo2()

def toggle_fo3(btn):
    if btn["image"] == str(fo_on_img):
        btn.config(image=fo_off_img)
    else:
        btn.config(image=fo_on_img)
    fo3()

def toggle_fo4(btn):
    if btn["image"] == str(fo_on_img):
        btn.config(image=fo_off_img)
    else:
        btn.config(image=fo_on_img)
    fo4()

fo1_btn = tk.Button(fo_col, image=fo_off_img, borderwidth=0, command=lambda: toggle_fo1(fo1_btn))
fo2_btn = tk.Button(fo_col, image=fo_off_img, borderwidth=0, command=lambda: toggle_fo2(fo2_btn))
fo3_btn = tk.Button(fo_col, image=fo_off_img, borderwidth=0, command=lambda: toggle_fo3(fo3_btn))
fo4_btn = tk.Button(fo_col, image=fo_off_img, borderwidth=0, command=lambda: toggle_fo4(fo4_btn))

fo1_btn.grid(row=0, column=0, padx=10, pady=25)
fo2_btn.grid(row=1, column=0, padx=10, pady=25)
fo3_btn.grid(row=2, column=0, padx=10, pady=25)
fo4_btn.grid(row=3, column=0, padx=10, pady=25)

root.mainloop()