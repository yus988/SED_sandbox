from transformers import pipeline
import torch
from datasets import load_dataset, Audio


def ex1():
    classifier = pipeline(task="sentiment-analysis")
    results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


if __name__ == "__main__":
    # ex1()
    speech_recognizer = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    dataset = load_dataset(path="PolyAI/minds14", name="en-US", split="train")
