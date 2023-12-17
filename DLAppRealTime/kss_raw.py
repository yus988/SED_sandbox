# modification to adapt input audio buffer
import keras
import numpy as np
import librosa

MODEL_PATH = "./model.h5"
# NUM_SAMPLES_TO_CONSIDER = 22050  # 1sec

class _Keyword_Spotting_Service:
    _instance = None
    model = None
    _mappings = [
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "house",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "wow",
        "yes",
        "zero",
    ]

    def predict(self, audio_buffer_data, sample_rate):
        # extract MFCCs
        MFCCs = self.preprocess(audio_buffer_data, sample_rate)  # (#segments, #coefficients)
        # convert 2d MFCCs array into 4d array -> (#samples, #segments, #coefficients, #channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # make prediction
        predictions = self.model.predict(MFCCs, verbose=None)  # [ [0.1, 0.6, ...] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]
        return predicted_keyword

    def preprocess(self, audio_buffer_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        # signal, sr = librosa.load(file_path)
        signal = audio_buffer_data
        # ensure consistency in the audio file length
        if len(signal) > sample_rate:
            signal = signal[:sample_rate]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service():
    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
       _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
       _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

# if __name__ == "__main__":
    # kss = Keyword_Spotting_Service()
    # keyword1 = kss.predict("test/down.wav")
    # keyword2 = kss.predict("test/left.wav")
    # print(f"predicted keywords: {keyword1}, {keyword2}")
