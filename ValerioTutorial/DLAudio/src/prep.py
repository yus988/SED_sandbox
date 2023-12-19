import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "audio/blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050)  # sr*T->22050*30
# librosa.display.waveshow(y=signal,sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# # plt.show() # docker は GUI が無いので表示されない。
# plt.savefig('tmp.png') # 代わりに保存して確かめれば良さそう

# fft -> spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[: int(len(frequency) / 2)]
left_magnitude = magnitude[: int(len(frequency) / 2)]

# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# # plt.show() # docker は GUI が無いので表示されない。
# plt.savefig('tmp.png') # 代わりに保存して確かめれば良さそう

# stft -> spectrogram
n_fft = 2048
hop_len = 512
stft = librosa.core.stft(signal, hop_length=hop_len, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_len)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.savefig("tmp.png")

# MFCCs
MFCCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_len, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_len)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.savefig("tmp.png")
