import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
augment = Compose(
    [
        AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.2, p=1),
        PitchShift(min_semitones=-8, max_semitones=8, p=1),
        HighPassFilter(min_cutoff_freq=2000,max_cutoff_freq=4000,p=1)
    ]
)

if __name__ == "__main__":
    signal, sr= librosa.load("scale.wav")
    augmented_signal = augment(signal, sr)
    sf.write("augmented.wav", augmented_signal, sr)
