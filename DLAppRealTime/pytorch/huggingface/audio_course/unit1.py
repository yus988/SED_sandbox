from datasets import load_dataset
import pprint
import gradio as gr

# load and explore an audio dataset
def LoadAndExploreAudioDataset():
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    print(minds)
    example = minds[0]
    pprint.pprint(example)
    # show the name of label
    id2label = minds.features["intent_class"].int2str
    pprint.pprint(id2label(example["intent_class"]))
    columns_to_remove = ["lang_id", "english_transcription"]
    minds = minds.remove_columns(columns_to_remove)
    pprint.pprint(minds)

    def generate_audio():
        example = minds.shuffle()[0]
        audio = example["audio"]
        return (audio["sampling_rate"], audio["array"]), id2label(example["intent_class"])

    with gr.Blocks() as demo:
        with gr.Column():
            for _ in range(4):
                audio, label = generate_audio()
                output = gr.Audio(audio, label=label)

    demo.launch(debug=True)


# Preprocessing an audio dataset
from datasets import Audio
import librosa
from transformers import WhisperFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt


def PreprocessingAudioDataset():
    ## how to re-sample the data from dataset
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    # remove unnesesary column
    columns_to_remove = ["lang_id", "english_transcription"]
    minds = minds.remove_columns(columns_to_remove)
    # pprint.pprint(minds[0])
    ### filtering the dataset = unify the wave length
    MAX_DURATION_IN_SEC = 20.0

    def is_audio_length_in_range(input_length):
        return input_length < MAX_DURATION_IN_SEC

    ## use librosa to get examples's duration from the audio file
    new_column = [librosa.get_duration(path=x) for x in minds["path"]]
    minds = minds.add_column("duration", new_column)

    ## use hf datasets' `filter` method to apply the filtering function
    minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])
    ## remove the temporary helper column
    # pprint.pprint(minds)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    # write more specfic preparation if you need
    def prepare_dataset(example):
        audio = example["audio"]
        features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], padding=True)
        return features

    minds = minds.map(prepare_dataset)
    pprint.pprint(minds)

    example = minds[0]
    input_features = example["input_features"]
    plt.figure().set_figwidth(12)
    librosa.display.specshow(
        np.asarray(input_features[0]),
        x_axis="time",
        y_axis="mel",
        sr=feature_extractor.sampling_rate,
        hop_length=feature_extractor.hop_length,
    )
    plt.colorbar()
    plt.show()


#### Streaming audio data
def StreamingAudioData():
    # ## gigaspeech is now in private, so you cannot access without login
    gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
    pprint.pprint(next(iter(gigaspeech["train"])))
    gigaspeech_head = gigaspeech["train"].take(2)
    list(gigaspeech_head)


if __name__ == "__main__":
    # LoadAndExploreAudioDataset()
    # PreprocessingAudioDataset()
    StreamingAudioData()