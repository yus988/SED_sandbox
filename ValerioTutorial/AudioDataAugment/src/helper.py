import librosa.display
import matplotlib.pyplot as plt

def _plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(y=signal, sr=sr, ax=ax[0])
    plt.plot(augmented_signal)
    librosa.display.waveshow(y=augmented_signal, sr=sr, ax=ax[1])
    # plt.show()
    plt.savefig('plot.png')
