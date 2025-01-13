# implementation of Discrete Fourier Transform (DFT)
import wave
import numpy as np

file_path = "./d_transform/song2024.wav"
wav_file = wave.open(file_path, 'r')

n_frames = wav_file.getnframes() 
frame_rate = wav_file.getframerate()

frames = wav_file.readframes(n_frames)
audio_data = np.frombuffer(frames, dtype=np.int16)  

print(f"We have this many - {n_frames} frames")
print(f"We have this many - {frame_rate} frame rate")
print(f"We have this many - {n_frames/frame_rate} sec")
print("- of audio.")



