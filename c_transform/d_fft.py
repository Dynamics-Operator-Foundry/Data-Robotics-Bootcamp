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


# print(audio_data_downsampled.shape[0] / new_frame_rate)
t = np.arange(0, audio_data_downsampled.shape[0], 1)

audio_data_padded = np.zeros(2**13)
# exit()

audio_data_padded[0:audio_data_downsampled.shape[0]] = audio_data_downsampled
scaling_ratio = audio_data_padded.shape[0] / audio_data_downsampled.shape[0]
# print(scaling_ratio)
# scaling_ratio = 2
audio_data_padded_original = audio_data_padded
# audio_data_padded *= scaling_ratio

t = np.arange(0, audio_data_padded.shape[0], 1)
t_length = t[t.shape[0] - 1]
n_length = audio_data_padded.shape[0]

def fft(data):
    
    N = data.shape[0]
    
    if N == 1:
        return data

    data_even = fft(data[0:N:2])
    data_odd  = fft(data[1:N:2])

    data_fft = np.zeros((N,1), dtype=np.complex128)
    
    for k in range(int(N/2)):
        
        t = np.exp(-2j * np.pi * k / N) * data_odd[k]
        
        data_fft[k, 0] = data_even[k] + t
        data_fft[k + int(N/2), 0] = data_even[k] - t
        
    return data_fft


print("NOW DO FFT\n\n")
F = fft(audio_data_padded).flatten() / np.sqrt(n_length)
bins = np.zeros((n_length, 2))
bins[:,0] = F.real 
bins[:,1] = F.imag 

# print(n_length)

# reconstruct signal
audio_reconstruct = np.zeros(shape=audio_data_padded.shape[0], dtype=np.complex128)

for i in range(n_length):
    freq_i = i
    
    # print(freq_i)
    
    if freq_i == 0:
        # print(i)
        audio_reconstruct += (
            bins[i, 0] * np.cos(0*t) 
            - bins[i, 1] * np.sin(0*t) 
        )
        continue
    
    period = t_length / freq_i
    # np.exp(-2j * np.pi / period * t)
    audio_reconstruct += F[i] * np.exp(2j * np.pi / period * t)
    
audio_reconstruct = audio_reconstruct / np.sqrt(n_length)

# t = t[0:audio_data_downsampled.shape[0]]
# audio_reconstruct = audio_reconstruct[0:audio_data_downsampled.shape[0]]

audio_data_lala = np.array(audio_reconstruct, dtype=np.int16)

sample_rate = new_frame_rate

output_path = "fft_voice_reconstruct.wav"
write(output_path, int(sample_rate), audio_data_lala)

viz_t_domain(
    t=t,
    data=audio_data_padded_original,
    save=True,
    save_name='fft_original_data'
)

viz_t_domain(
    t=t,
    data=audio_reconstruct,
    save=True,
    save_name='fft_reconstruct_data'
)

viz_t_domain(
    t=t,
    data=audio_reconstruct-audio_data_padded,
    save=True,
    save_name='fft_reconstruct_error'
)

# viz_f_domain(
#     f=np.arange(0,n_length,1),
#     bins=bins,
#     save=True,
#     save_name='fft_DFT'
# )
    
exit()



