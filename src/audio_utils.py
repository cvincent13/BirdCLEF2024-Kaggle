import IPython.display as ipd
import torch
import matplotlib.pyplot as plt

def play_audio(waveform, sr):
    return ipd.Audio(waveform.numpy()[0], rate=sr)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    if len(waveform.shape) == 1:
        num_channels = 1
        num_frames = waveform.shape[0]
        waveform = waveform[None, :]
    else:
        num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    if len(waveform.shape) == 1:
        num_channels = 1
        num_frames = waveform.shape[0]
        waveform = waveform[None, :]
    else:
        num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)