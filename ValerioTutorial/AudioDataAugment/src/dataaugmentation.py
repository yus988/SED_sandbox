import numpy as np
import librosa
import soundfile as sf
from helper import _plot_signal_and_augmented_signal
import random

# adding white noise
def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal

# time stretch
def time_stretch(signal, stretch_rate):
    return librosa.effects.time_stretch(y=signal, rate=stretch_rate)

# pitch scaling
def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=num_semitones)

# polarity inversion
def invert_polarity(signal):
    return signal * -1

# random gain
def random_gain(signal, min_gain_factor, max_gain_factor):
    gain_factor = random.uniform(min_gain_factor, max_gain_factor)
    return signal * gain_factor

if __name__ == "__main__":
    signal, sr = librosa.load("scale.wav")
    # augmented_signal = add_white_noise(signal, 0.1)
    # augmented_signal = time_stretch(signal, 0.8)
    # augmented_signal = pitch_scale(signal, sr, 2)
    augmented_signal = random_gain(signal, 2, 4)
    sf.write("augmented.wav", augmented_signal, sr)
    _plot_signal_and_augmented_signal(signal, augmented_signal, sr)
