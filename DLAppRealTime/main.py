import pyaudio
import numpy as np
import time
import array
import queue
from kss_raw import Keyword_Spotting_Service
import wave
import keyboard
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

TEST_AUDIO_FILE_PATH = "./test/left.wav"
SAMPLE_RATE = 22050
CHUNK = int(SAMPLE_RATE / 10)
CHUNK_MEL = 22050  # 推定したい音の長さに合わせる。1秒=sample_rate


# 推論に使われる音声を描画
class PlotWindow:
    def __init__(self):
        self.fig, (self.ax1) = plt.subplots(1, 1, figsize=(12, 8))
        # self.ax1.plot([1,2,3], [3,4,5])

    def update(self, xdata, ydata):
        plt.cla()
        buf = convert_buffer(ydata)
        self.ax1.plot(xdata, buf)
        plt.pause(0.01)


pred_buffer = []  # 推論用のリスト


# 取得した音声を np.array に変換＆1~-1にマップ
def convert_buffer(arg: list) -> np.array:
    return np.array(arg) / 32768.0


class AudioInputStream:
    def __init__(self):
        # マイクインプット設定
        self.CHUNK = CHUNK  # 1度に読み取る音声のデータ幅
        self.audio = pyaudio.PyAudio()
        self.channel = 1
        self.rate = SAMPLE_RATE
    
    def start_stream(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channel,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.update,
        )
        print("stream started!")

    def recordOnce(self, filename, data: list):
        w = wave.Wave_write(filename)
        w.setparams(
            (
                self.channel,  # channel
                2,  # byte width
                self.rate,  # sampling rate
                len(data),  # number of frames
                "NONE",
                "not compressed",  # no compression
            )
        )
        w.writeframes(array.array("h", data).tostring())
        w.close()

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
            del pred_buffer[:CHUNK]
        return (None, pyaudio.paContinue)


# 連続的に推論を実行
def pred_loop():
    win = PlotWindow()
    global kss
    kss = Keyword_Spotting_Service()
    i = 0
    xdata = np.linspace(0, CHUNK_MEL, CHUNK_MEL)
    # グラフ描画用
    ais = AudioInputStream()
    ais.start_stream()
    # 連続的に実行
    while ais.stream.is_active():
        if len(pred_buffer) == CHUNK_MEL:
            win.update(xdata, pred_buffer)  # 音声波形を表示
            # ais.recordOnce("./wav/rec_{}.wav".format(i), pred_buffer)
            buf = convert_buffer(pred_buffer)  # 推論用に変換
            predicted_keyword = kss.predict(buf, SAMPLE_RATE)
            # print(f"Predicted keyword is: {predicted_keyword}")
            if predicted_keyword == "dog":
                detect = predicted_keyword
            else:
                detect = "____"
            print("\rdetect: {}".format(detect), end="")
        i += 1
        time.sleep(0.1)  # 推論頻度を決定
    ais.stream.stop_stream()
    ais.stream.close()

def pred_once():
    win = PlotWindow()
    global kss
    kss = Keyword_Spotting_Service()
    # グラフ描画用
    xdata = np.linspace(0, CHUNK_MEL, CHUNK_MEL)
    ais = AudioInputStream()
    # キー入力を待ち受け
    input('press any key to proceed: ')
    ais.start_stream()
    while ais.stream.is_active():
        if len(pred_buffer) == CHUNK_MEL:
            win.update(xdata, pred_buffer)  # 音声波形を表示
            ais.recordOnce("./wav/rec_once.wav", pred_buffer)
            buf = convert_buffer(pred_buffer)  # 推論用に変換
            predicted_keyword = kss.predict(buf, SAMPLE_RATE)
            print(f"Predicted keyword is: {predicted_keyword}")
            break
    print("Finish!")
    ais.stream.stop_stream()
    ais.stream.close()
    # input('press any key to close: ')

if __name__ == "__main__":
    # pred_once()
    pred_loop()

# python ref/pyqtgraph/dispWave.py

# ################ 予測用の配列抽出部分 #######################
# # きちんとオーバーラップは考慮されている。
# # 0:44100+4410 をメルスペクトログラムに変換
# audio_to_convert = np.array(raw_audio_buffer[:conf.mels_convert_samples]) / 32767
# # 0:4410 を捨てる
# raw_audio_buffer = raw_audio_buffer[conf.mels_onestep_samples:]
# ###########################################################
