# modification to adapt input audio buffer
from transformers import pipeline

MODEL_PATH = (
    "./pytorch/model/ast-finetuned-audioset-10-10-0.4593-finetuned-us8k/checkpoint-983/"
)


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
        return preds[0]['label']
        outputs = {}
        for p in preds:
            outputs[p["label"]] = p["score"]
        return outputs


def Inference_instance():
    # ensure that we only have 1 instance of KSS
    if _Inference._instance is None:
        _Inference._instance = _Inference()
    return _Inference._instance
