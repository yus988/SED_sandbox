import time
import array

pred_buffer = []
CHUNK = 3
CHUNK_MEL = 30

if __name__ == "__main__":
    i = 0
    while 1:
        wave = array.array("h", [i, i + 1, i + 2])
        pred_buffer.extend(wave)
        if len(pred_buffer) > CHUNK_MEL:
            # print(f"buf len before: {len(pred_buffer)}")
            del pred_buffer[:CHUNK]
        print(pred_buffer)
        i += 3
        time.sleep(0.5)
