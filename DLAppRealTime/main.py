from ref.keyword_spotting_service import Keyword_Spotting_Service
import pyaudio
import numpy as np
import time
import array
import queue
from kss_raw import Keyword_Spotting_Service

TEST_AUDIO_FILE_PATH = "./test/left.wav"
SAMPLE_RATE = 22050
CHUNK = int(SAMPLE_RATE/10)
CHUNK_MEL = 22050 # 推定したい音の長さに合わせる。1秒=sample_rate


    # # 使用する推論を代入
    # def setPredictor(self, predictor):
    #     self.pred = predictor

class AudioInputStream:
    def __init__(self):
        # マイクインプット設定
        self.CHUNK = CHUNK  # 1度に読み取る音声のデータ幅
        self.audio = pyaudio.PyAudio()
        self.audio_buffer_queue = queue.Queue(maxsize=CHUNK)  # メススペクトログラムから推定する用のバッファ。アプリによって長さは異なる
        self.audio_buffer_data = []  # 推論用の配列
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.update,
        )
    
    # 音声を取り込む度に実行する関数
    # 音声取り込み => 推論用のAudio配列にFIFOで挿入 => 推論して結果を表示
    def update(self, in_data, frame_count, time_info, status):
        # 0. 取得したデータを16進数で配列化
        wave = array.array("h", in_data)
        # print(f"wave len = {len(wave)}")
        # print(f"buf len = {len(self.audio_buffer_data)}")
        # 1. que が full になるまで入れる（基本開始直後のみ）
        while not self.audio_buffer_queue.full():
            self.audio_buffer_queue.put(wave)
        # 2. full になった que から CHUNK 分のデータを捨てる
        for i in range(1, CHUNK):
            self.audio_buffer_queue.get()
        # 3. 空いたところに新しい CHUNK 分のデータを入れる
        if len(wave)==CHUNK:
            self.audio_buffer_queue.put(wave)
        # self.audio_buffer_data = self.audio_buffer_queue
        return (None, pyaudio.paContinue)
        

    # 後々消す
    def AudioInput(self):
        ret = self.stream.read(self.CHUNK)  # 音声の読み取り(バイナリ)
        # バイナリ → 数値(int16)に変換
        # 32768.0=2^16で割ってるのは正規化(絶対値を1以下にすること)
        ret = np.frombuffer(ret, dtype="int16") / 32768.0
        return ret

if __name__ == "__main__":
    # pro_size = 10
    # for i in range(1, pro_size + 1):
    #     pro_bar = ('=' * i) + (' ' * (pro_size - i))
    #     print('\r[{0}] {1}%'.format(pro_bar, i / pro_size * 100.), end='')
    #     time.sleep(0.5)
    kss = Keyword_Spotting_Service()
    ais = AudioInputStream()
    while ais.stream.is_active():
        if ais.audio_buffer_queue.full():
            predicted_keyword = kss.predict(ais.audio_buffer_queue, SAMPLE_RATE)
            print(f"Predicted keyword is: {predicted_keyword}")
        else:
            print("queue is not full")
        
        # val = ais.AudioInput()[0]
        # rms = np.sqrt(np.mean(val**2))
        # print("\rVal = {0}".format(rms), end="")
        time.sleep(1) # 推論頻度を決定

    ais.stream.stop_stream()
    ais.stream.close()
    ais.close()

    # predicted_keyword = kss.predict(TEST_AUDIO_FILE_PATH)
    # print(f"Predicted keyword is: {predicted_keyword}")


# python ref/pyqtgraph/dispWave.py
