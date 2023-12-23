from datasets import load_dataset, Audio
from transformers import pipeline
import pprint

def AudioClassification():
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    classifier = pipeline(
        "audio-classification",
        model="anton-l/xtreme_s_xlsr_300m_minds14",
    )
    example = minds[0]
    # pprint.pprint(classifier(example["audio"]["array"]))
    id2label = minds.features["intent_class"].int2str
    pprint.pprint(id2label(example["intent_class"]))

def AutomaticSpeechRecognition():
    # minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    # asr = pipeline("automatic-speech-recognition")
    asr = pipeline("automatic-speech-recognition", model="maxidl/wav2vec2-large-xlsr-german")
    example = minds[0]
    pprint.pprint(asr(example["audio"]["array"]))
    pprint.pprint(example["transcription"])

# https://huggingface.co/learn/audio-course/chapter2/tts_pipeline
# from IPython.display import Audio
def AudioGeneration():
    pipe = pipeline("text-to-speech", model="suno/bark-small")
    text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
    fr_text = "Contrairement à une idée répandue, le nombre de points sur les élytres d'une coccinelle ne correspond pas à son âge, ni en nombre d'années, ni en nombre de mois. "
    song = "♪ In the jungle, the mighty jungle, the ladybug was seen. ♪ "
    output = pipe(text)
    output = pipe(fr_text)
    output = pipe(song)
    Audio(output["audio"], rate=output["sampling_rate"])

    music_pipe = pipeline("text-to-audio", model="facebook/musicgen-small")
    text = "90s rock song with electric guitar and heavy drums"
    forward_params = {"max_new_tokens": 512}
    output = music_pipe(text, forward_params=forward_params)
    Audio(output["audio"][0], rate=output["sampling_rate"])


if __name__ == "__main__":
    # AudioClassification()
    # AutomaticSpeechRecognition()
    AudioGeneration()
    