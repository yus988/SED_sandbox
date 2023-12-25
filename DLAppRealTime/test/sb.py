# サンドボックス用スクリプト。動作確認用に使用

import time
import array
from pynput import keyboard


pred_buffer = []
CHUNK = 3
CHUNK_MEL = 30

def testArray():
    wave = array.array("h", [i, i + 1, i + 2])
    pred_buffer.extend(wave)
    if len(pred_buffer) > CHUNK_MEL:
    # print(f"buf len before: {len(pred_buffer)}")
        del pred_buffer[:CHUNK]
    print(pred_buffer)
    i += 3
    time.sleep(0.5)
    
def print_update():
    pro_size = 10
    for i in range(1, pro_size + 1):
        pro_bar = ('=' * i) + (' ' * (pro_size - i))
        print('\r[{0}] {1}%'.format(pro_bar, i / pro_size * 100.), end='')
        time.sleep(0.5)
    
    
def testKeyboard():
     
    if keyboard.is_pressed("r"):
        print("pressed")
        
    # 離れた時一回だけ検知したい
    
    isPressedRecKey = True


if __name__ == "__main__":
    i = 0
    while 1:
        testKeyboard()
        time.sleep(0.1)
        print(i)
        i+=1