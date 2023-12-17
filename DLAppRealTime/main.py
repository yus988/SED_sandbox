from ref.keyword_spotting_service import Keyword_Spotting_Service
import pyaudio
import numpy as np
import time
import array
import queue
from kss_raw import Keyword_Spotting_Service
import pyqtgraph as pg
import sys
import scipy.io.wavfile
import wave


TEST_AUDIO_FILE_PATH = "./test/left.wav"
SAMPLE_RATE = 22050
CHUNK = int(SAMPLE_RATE / 10)
CHUNK_MEL = 22050  # 推定したい音の長さに合わせる。1秒=sample_rate


# # 使用する推論を代入
# def setPredictor(self, predictor):
#     self.pred = predictor
pred_buffer = []


class PlotWindow:
    def __init__(self):
        # プロット初期設定
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("realtime plot")
        self.plt = self.win.addPlot()  # プロットのビジュアル関係
        self.plt.setYRange(-1, 1)  # y軸の上限、下限の設定
        self.curve = self.plt.plot()  # プロットデータを入れる場所

        # アップデート時間設定
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)  # 10msごとにupdateを呼び出し

    def update(self):
        global pred_buffer
        buf = np.array(pred_buffer) / 32768.0
        self.curve.setData(buf)


class AudioInputStream:
    def __init__(self):
        # マイクインプット設定
        self.CHUNK = CHUNK  # 1度に読み取る音声のデータ幅
        self.audio = pyaudio.PyAudio()
        self.audio_buffer_queue = queue.Queue(maxsize=CHUNK)  # メススペクトログラムから推定する用のバッファ。アプリによって長さは異なる
        self.audio_buffer_data = []  # 推論用の配列
        self.channel = 1
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channel,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.update,
        )

    # 音声を取り込む度に実行する関数
    # 音声取り込み => 推論用のAudio配列にFIFOで挿入 => 推論して結果を表示
    def update(self, in_data, frame_count, time_info, status):
        global pred_buffer
        # 0. 取得したデータを16進数で配列化
        wave = array.array("h", in_data)
        # # 1. 取得したデータを配列に追加
        pred_buffer.extend(wave)
        # 2. 超えている場合、最初のCHUNK分を削除
        if len(pred_buffer) > CHUNK_MEL:
            # print(f"buf len before: {len(pred_buffer)}")
            del pred_buffer[:CHUNK]
            # print(f"buf len after: {len(pred_buffer)}")
        # # 3. 推論用の配列を更新（-1~1にマップするため 32767 で割る）
        # if len(self.audio_buffer_data) == CHUNK_MEL:
        #     pred_buffer = np.array(self.audio_buffer_data) / 32768.0

        # self.audio_buffer_queue.put(wave, True)
        # print(f"wave len = {wave}")
        # print(f"buf len = {len(pred_buffer)}")

        # 1. que が full になるまで入れる（基本開始直後のみ）

        # while not self.audio_buffer_queue.empty():
        #     self.audio_buffer_data.append()
        # 2. full になった que から CHUNK 分のデータを捨てる
        # for i in range(1, CHUNK):
        #     self.audio_buffer_queue.get()
        # # 3. 空いたところに新しい CHUNK 分のデータを入れる
        # if len(wave)==CHUNK:
        #     self.audio_buffer_queue.put(wave)
        # # self.audio_buffer_data = self.audio_buffer_queue
        return (None, pyaudio.paContinue)

    # 後々消す
    def AudioInput(self):
        ret = self.stream.read(self.CHUNK)  # 音声の読み取り(バイナリ)
        # バイナリ → 数値(int16)に変換
        # 32768.0=2^16で割ってるのは正規化(絶対値を1以下にすること)
        ret = np.frombuffer(ret, dtype="int16") / 32768.0
        return ret

    def RecordWav(self):
        wavFile = wave.open("./wav/test.wav", "wb")
        wavFile.setnchannels(self.channel)
        wavFile.setsampwidth(2)
        wavFile.setframerate(SAMPLE_RATE)
        buf = np.array(pred_buffer) / 32768.0
        wavFile.writeframes(buf)  # Python3用
        wavFile.close()


if __name__ == "__main__":
    # pro_size = 10
    # for i in range(1, pro_size + 1):
    #     pro_bar = ('=' * i) + (' ' * (pro_size - i))
    #     print('\r[{0}] {1}%'.format(pro_bar, i / pro_size * 100.), end='')
    #     time.sleep(0.5)
    i = 0
    kss = Keyword_Spotting_Service()
    ais = AudioInputStream()
    while ais.stream.is_active():
        ais.RecordWav()
        # print(f"{pred_buffer}")
        
        # if len(pred_buffer) == CHUNK_MEL:
        #     buf = np.array(pred_buffer) / 32768.0
        #     predicted_keyword = kss.predict(buf, SAMPLE_RATE)
        #     print(f"Predicted keyword is: {predicted_keyword}")
        # else:
        #     print("queue is not full")

        # val = ais.AudioInput()[0]
        # rms = np.sqrt(np.mean(val**2))
        # print("\rVal = {0}".format(rms), end="")

        time.sleep(1)  # 推論頻度を決定

    ais.stream.stop_stream()
    ais.stream.close()
    ais.close()

    # predicted_keyword = kss.predict(TEST_AUDIO_FILE_PATH)
    # print(f"Predicted keyword is: {predicted_keyword}")


# python ref/pyqtgraph/dispWave.py
