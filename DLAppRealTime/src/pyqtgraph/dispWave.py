# -*- coding:utf-8 -*-

# プロット関係のライブラリ
import pyqtgraph as pg
import numpy as np
import sys

# 音声関係のライブラリ
import pyaudio
import struct


class PlotWindow:
    def __init__(self):
        # プロット初期設定
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("realtime plot")
        self.plt = self.win.addPlot()  # プロットのビジュアル関係
        self.plt.setYRange(-1, 1)  # y軸の上限、下限の設定
        self.curve = self.plt.plot()  # プロットデータを入れる場所

        # マイクインプット設定
        self.CHUNK = 1024  # 1度に読み取る音声のデータ幅
        self.RATE = 44100  # サンプリング周波数
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK
        )

        # アップデート時間設定
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)  # 10msごとにupdateを呼び出し

        # 音声データの格納場所(プロットデータ)
        self.data = np.zeros(self.CHUNK)

    def update(self):
        self.data = self.AudioInput()
        self.curve.setData(self.data)  # プロットデータを格納

    def AudioInput(self):
        ret = self.stream.read(self.CHUNK)  # 音声の読み取り(バイナリ)
        # バイナリ → 数値(int16)に変換
        # 32768.0=2^16で割ってるのは正規化(絶対値を1以下にすること)
        ret = np.frombuffer(ret, dtype="int16") / 32768.0
        return ret


if __name__ == "__main__":
    plotwin = PlotWindow()
    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        pg.exec()
