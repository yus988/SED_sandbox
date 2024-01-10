# modification to adapt input audio buffer
from transformers import pipeline
import configparser
config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')

MOEL_NAME = config_ini['DEFAULT']['model_name']
CHECKPOINT = config_ini['DEFAULT']['checkpoint']
MODEL_PATH = f"./model/{MOEL_NAME}/{CHECKPOINT}/"

class _Inference:
    _instance = None

    def __init__(self) -> None:
        self.pipe = pipeline(
            "audio-classification",
            model=MODEL_PATH,
            top_k=1,
        )

    def classify_audio(self, filepath):
        preds = self.pipe(filepath)
        # return preds[0]['label']
        outputs = {}
        for p in preds:
            outputs[p["label"]] = p["score"]
        return outputs


def Inference_instance():
    # ensure that we only have 1 instance of KSS
    if _Inference._instance is None:
        _Inference._instance = _Inference()
    return _Inference._instance
