from ref.keyword_spotting_service import Keyword_Spotting_Service
# TEST_AUDIO_FILE_PATH = "./test/down.wav"
TEST_AUDIO_FILE_PATH = "./wav/rec_once.wav"

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    predicted_keyword = kss.predict(TEST_AUDIO_FILE_PATH)
    print(f"Predicted keyword is: {predicted_keyword}")

# python ref/pyqtgraph/dispWave.py