from datasets import load_dataset
from transformers import pipeline
import pprint
import torch

# from IPython.display import Audio


def Minds14():
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    classifier = pipeline(
        "audio-classification",
        model="anton-l/xtreme_s_xlsr_300m_minds14",
    )
    pprint.pprint(classifier(minds[0]["path"]))


def SpeechCommand():
    speech_commands = load_dataset("speech_commands", "v0.02", split="validation", streaming=True)
    sample = next(iter(speech_commands))
    classifier = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")
    pprint.pprint(classifier(sample["audio"].copy()))
    Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])


def FLEURS():
    fleurs = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(fleurs))
    classifier = pipeline("audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id")
    pprint.pprint(classifier(sample["audio"]))


def ZeroShot():
    dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
    audio_sample = next(iter(dataset))["audio"]["array"]
    # candidate_labels = ["Sound of a dog", "Sound of vacuum cleaner"]
    candidate_labels = ["human", "vacuum cleaner", "dog"]

    classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    pprint.pprint(classifier(audio_sample, candidate_labels=candidate_labels))


# https://huggingface.co/learn/audio-course/chapter4/fine-tuning
import gradio as gr
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import Audio
import numpy as np
import evaluate


def Finetuning():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gtzan = load_dataset("marsyas/gtzan", "all")
    gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
    # pprint.pprint(gtzan)
    # pprint.pprint(gtzan["train"][0])
    id2label_fn = gtzan["train"].features["genre"].int2str
    # pprint.pprint(id2label_fn(gtzan["train"][0]["genre"]))
    # def generate_audio():
    #     example = gtzan["train"].shuffle()[0]
    #     audio = example["audio"]
    #     return(audio["sampling_rate"], audio["array"]), id2label_fn(example["genre"])
    # with gr.Blocks() as demo:
    #     with gr.Column():
    #         for _ in range(4):
    #             audio, label = generate_audio()
    #             output = gr.Audio(audio,label=label)
    # demo.launch(debug=True)
    model_id = "ntu-spml/distilhubert"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True, return_attention_mask=True)
    sampling_rate = feature_extractor.sampling_rate
    gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

    sample = gtzan["train"][0]["audio"]
    # pprint.pprint(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")
    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
    # pprint.pprint(f"inputs keys: {list(inputs.keys())}")
    print(f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}")

    max_duration = 30.0

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
        )
        return inputs

    gtzan_encoded = gtzan.map(
        preprocess_function, remove_columns=["audio", "file"], batched=True, batch_size=100, num_proc=1
    )
    # pprint.pprint(gtzan_encoded)
    gtzan_encoded = gtzan_encoded.rename_column("genre", "label")

    id2label = {str(i): id2label_fn(i) for i in range(len(gtzan_encoded["train"].features["label"].names))}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    ).to(device)

    model_name = model_id.split("/")[-1]
    batch_size = 8
    gradient_accumulation_steps = 1
    num_train_epochs = 10

    training_args = TrainingArguments(
        f"{model_name}-finetuned-gtzan",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        hub_token="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",
        push_to_hub=True,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=gtzan_encoded["train"].with_format("torch"),
        eval_dataset=gtzan_encoded["test"].with_format("torch"),
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def MusicClassifierDemo():
    model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"
    pipe = pipeline("audio-classification", model=model_id)

    def classify_audio(filepath):
        preds = pipe(filepath)
        outputs = {}
        for p in preds:
            outputs[p["label"]] = p["score"]
        return outputs

    # AttributeError: module 'gradio' has no attribute 'outputs'
    demo = gr.Interface(fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.outputs.Label())
    demo.launch(debug=True)


if __name__ == "__main__":
    # Minds14():
    # SpeechCommand()
    # FLEURS()
    # ZeroShot()
    Finetuning()
    # MusicClassifierDemo()
