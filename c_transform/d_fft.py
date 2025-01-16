# implementation of Fast Fourier Transform (FFT)
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


file_path = "./c_transform/voice_original.wav"
wav_file = wave.open(file_path, 'r')

n_frames = wav_file.getnframes() 
frame_rate = wav_file.getframerate()
sample_width = wav_file.getsampwidth()  # sample width in bytes, no of bytes for the amplitude of sound wave

frames = wav_file.readframes(n_frames)
audio_data = np.frombuffer(frames, dtype=np.int16)  

print(f"We have this many - {n_frames} frames")
print(f"We have this many - {frame_rate} frame rate")
print(f"We have this many - {n_frames/frame_rate} sec")
# print("- of audio.")

# visualization of the wav data - via sampling 100 data per sec.
def viz_t_domain(t, data, save=False, save_name = ''):
    plt.plot(t, data)
    plt.xlabel('Time')
    plt.ylabel('amplitude')
    plt.ylim(-9000,7000)
    plt.title(save_name)
    plt.legend()

    # Show plot
    plt.grid(True)
    if save:
        plt.savefig(save_name + "_wav_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.1)  # pad_inches adds margin
    plt.show()
    
    return

def viz_f_domain(f, bins, save=False, save_name=''):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar width (adjust this value as needed)
    width = 2

    freq_lib = f
    x = freq_lib

    # create the bar chart for each coefficient
    ax.bar(x - width * 0.5, bins[:, 0], width, label='A_cos', color='green')
    ax.bar(x + width * 0.5, bins[:, 1], width, label='A_sin', color='red')

    # Labels and title
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude')
    ax.set_title('DFT - Fourier Coefficients for Each Frequency')

    ax.legend()

    plt.xticks(rotation=45)

    # Save the plot if requested
    if save:
        plt.savefig(save_name + "_wav_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.1)  # pad_inches adds margin

    # Show the plot
    plt.tight_layout()
    plt.show()

# Now perform DFT with 192000 / 4 = 48000 data point per sec
downsample = 4
new_frame_rate = frame_rate/downsample
audio_data_downsampled = audio_data[::downsample]
n_length = audio_data_downsampled.shape[0]

# print(audio_data_downsampled.shape[0] / new_frame_rate)
t = np.arange(0, audio_data_downsampled.shape[0], 1)
print(t.shape)
