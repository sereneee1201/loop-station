<pre>
  _         _____    _____    _____      
 | |      /  ___ \  / ___ \  / ___ \    
 | |      | |   | || |   | || |   | |   
 | |      | |   | || |   | || |___| |    
 | |      | |   | || |   | ||  ____/    
 | |_____ | |___| || |___| || |         
 |_______| \_____/  \_____/ |_|          

   _____   _______   _____   _______  _______   _____   ___     _
  / ___ \ |__   __| / ___ \ |__   __||__   __|/  ___ \ / _ \   | |
 | |   |_|   | |   | |   | |   | |      | |   | |   | || |\ \  | |
  \_____     | |   | |___| |   | |      | |   | |   | || | \ \ | |
  _  __ \    | |   |  ___  |   | |      | |   | |   | || |  \ \| |
 | |___| |   | |   | |   | |   | |    __| |__ | |___| || |   \ | |
  \_____/    |_|   |_|   |_|   |_|   |_______| \_____/ |_|    \__|
</pre>

# AIST2010 2024 Fall Group Project - Group 3

This is a Python-based virtual loop station that allows users to record, playback, and layer soundtracks in real time with a user-friendly interface. 


## Features

- **Multiple Tracks**: Support recording 4 looping tracks.

- **Audio Visualization**: Display waveform of looping segments.

- **Volume Adjustment**: Allow individual volume control on each track.

- **Tracks Synchronization**: Provide Synchronization for delayed tracks. 

- **Noise Reduction**: Reduce background noises.

- **5 Audio Filters**: Pitch Up, Pitch Down, Reverb, Delay, and Fade Out.

- **Piano Melodies**: Allow piano notes playing with computer keyboard.


## Prerequisites

Before you run the code, make sure you meet the following requirements:

- **Python**

- **Required Libraries**: You need to install the following Python packages. You can do this using pip:

    pip install matplotlib tkinter librosa wave pillow pedalboard pydub noisereduce scipy numpy pyaudio


## Getting Started

1. To run the loop station:
    python main.py

2. Enter preferred tempo, beats per bar, and bar per loop. Click the "Start" button.

3. Listen to the metronome and get ready to record the first track. 

4. Click the "Record" button with microphone icon to start the recording. Click it again when you complete recording. 

5. Play and try all the features in our loop station!!

6. When you complete your own piece of music with our loop station, click "Export"

7. Obtain your amazing performance in ./audio named "output.wav". Separating tracks and looping segments will be provided in the same folder.


## Acknowledgements

Special thanks to the following authors for providing their amazing resources for this project.

**Libraries**
- [matplotlib](https://doi.org/10.1109/MCSE.2007.55)
- [pedalboard](https://doi.org/10.5281/zenodo.7817838)
- [librosa](https://doi.org/10.5281/zenodo.591533)
- [noisereduce](https://doi.org/10.5281/zenodo.3243139) 

**Reference Code on Streaming and Looping**
- [Overdub Looper](https://drive.google.com/file/d/1a5WT6QUHd2toJwf9eTc194scYMzZY-vf/view)

**Images**

- [Loop Station Icon](https://icon-icons.com/icon/loop-infinite/149437)
- [Recording On Button](https://icon-icons.com/icon/voice-recording/99906)
- [Recording Off Button](https://icon-icons.com/icon/voice-recording/100005)
- [Play Button](https://icon-icons.com/icon/control-play/66504)
- [Pause Button](https://icon-icons.com/icon/control-pause/66503)
- [Noise Reduction On Button](https://www.freepik.com/icon/no-sound_5601677#fromView=resource_detail&position=25)
- [Noise Reduction Off Button](https://www.freepik.com/icon/speakers_5601631#fromView=resource_detail&position=30)
- [Alien On Button](https://iconduck.com/emojis/131549/alien)
- [Alien Off Button](https://iconduck.com/emojis/127024/alien)
- [Cow Button](https://www.flaticon.com/free-icon/cow_4594605)
- [Reverb Button](https://logopond.com/rivingtondesignhouse/showcase/detail/221874)
- [Delay Button](https://ixintu.com/tj/2377484-2.html)
- [Fade out Button](https://www.pngwing.com/en/free-png-nxcgt)
- [Piano Keyboard Image](https://images.app.goo.gl/AoQUaLM3ksqqaojR6)
