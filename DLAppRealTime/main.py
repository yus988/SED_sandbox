import pyaudio
import numpy as np
import time
import array
from inference import Inference_instance
import wave
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import librosa
import configparser
import os
import errno
import serial
import heapq
import statistics


config_ini = configparser.ConfigParser()
config_ini_path = "config.ini"
config_ini.read(config_ini_path, encoding="utf-8")
if not os.path.exists(config_ini_path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)

SAMPLE_RATE = int(config_ini["DEFAULT"]["sample_rate"])
CHUNK = int(config_ini["DEFAULT"]["chunk"])
RECORD_DURATION = float(config_ini["DEFAULT"]["record_duration"])
INFERENCE_INTERVAL = float(config_ini["DEFAULT"]["inference_interval"])
CHUNK_MEL = int(SAMPLE_RATE * RECORD_DURATION)  # 推定したい音の長さに合わせる。1秒=sample_rate

# serial port configuration
COM_PORT = config_ini["Serial"]["com_port"]
COM_SPEED = config_ini["Serial"]["com_speed"]
SERIAL_INTERVAL = float(config_ini["Serial"]["serial_interval"])

# calculate magnitute
MAX_EXTRACT_LENGTH = int(config_ini["Calc"]["max_extract_length"])
RANGE_TO_MAX = int(config_ini["Calc"]["range_to_max"])

pred_buffer = []  # 推論用のリスト


# 推論に使われる音声を描画
def updateWindow(xdata, ydata):
    plt.plot(xdata, ydata)
    plt.draw()
    plt.ylim(-4000, 4000)
    plt.pause(0.001)
    plt.cla()


# 取得した音声を np.array に変換＆1~-1にマップ
def convert_buffer(arg: list) -> np.array:
    return np.array(arg) / 32768.0


# buffer から最大の平均を取り出す。
def get_max_mean(arg: list) -> float:
    max_values = heapq.nlargest(MAX_EXTRACT_LENGTH, abs(arg))
    return statistics.mean(max_values)


# マイコン送信用に0--255の値にマップ
def map_to_mcu(arg: float) -> int:
    val = round(arg * RANGE_TO_MAX)
    return val


def convert_send_data(category, pos, id, isStereo, L_Vol, R_Vol) -> str:
    # 改行コードを入れないと待ち受けが発生する模様
    return f"{category},{pos},{id},{isStereo},{L_Vol},{R_Vol}\r\n"


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
        w.writeframes(array.array("h", data).tobytes())
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
    global pred_buffer
    xdata = np.linspace(0, CHUNK_MEL, CHUNK_MEL)
    ser = serial.Serial(COM_PORT, COM_SPEED, timeout=0)
    # グラフ描画用
    ais = AudioInputStream()
    ais.start_stream()
    i = 0
    start = time.perf_counter()
    # 連続的に実行
    while ais.stream.is_active():
        try:
            if len(pred_buffer) == CHUNK_MEL:
                start = time.perf_counter()
                # updateWindow(xdata, pred_buffer)  # 音声波形を表示 0.055秒程度かかる
                buf = convert_buffer(pred_buffer)  # 推論用に変換
                predicted_keyword = inf.classify_audio(buf)
                if predicted_keyword.get("clapping") > 0.7:
                    i += 1
                if i == 3:
                    mag = get_max_mean(buf)
                    mag = map_to_mcu(mag)
                    data = convert_send_data(0, 0, 0, 0, mag, mag)
                    ser.write(data.encode("utf-8"))
                    # time.sleep(SERIAL_INTERVAL)
                    pred_buffer = []  # 連続で判定されるのを避ける
                    i = 0
                    convert_send_data

                end = time.perf_counter()
                mag = get_max_mean(buf)
                # mag = map_to_mcu(mag)
                print(
                    f"Predicted keyword is: {predicted_keyword}"
                    + "\n"
                    + "loundness is: {:.6g}".format(mag)
                    + "\n"
                    + "interval is:  {}".format(end - start)
                    + "\033[2A\r",
                    end="",
                )

                # print(end - start)
            # time.sleep(INFERENCE_INTERVAL)  # 推論頻度を決定
        except KeyboardInterrupt:
            break
    ser.close()
    ais.stream.stop_stream()
    ais.stream.close()


# 1回のみ、デバッグ用
def pred_once():
    # グラフ描画用
    xdata = np.linspace(0, CHUNK_MEL, CHUNK_MEL)
    ais = AudioInputStream()
    # キー入力を待ち受け
    input("press any key to proceed: ")
    ais.start_stream()
    while ais.stream.is_active():
        if len(pred_buffer) == CHUNK_MEL:
            updateWindow(xdata, pred_buffer)  # 音声波形を表示
            ais.recordOnce("./wav/rec_once.wav", pred_buffer)
            buf = convert_buffer(pred_buffer)  # 推論用に変換
            predicted_keyword = inf.classify_audio(buf)
            print(f"Predicted keyword is: {predicted_keyword}")
            break
    print("Finish!")
    ais.stream.stop_stream()
    ais.stream.close()


if __name__ == "__main__":
    global inf
    inf = Inference_instance()
    if config_ini["DEFAULT"]["runtype"] == "loop":
        pred_loop()
    if config_ini["DEFAULT"]["runtype"] == "once":
        pred_once()
