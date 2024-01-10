from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModel,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    
)
import torch
from torch import nn
from datasets import load_dataset, Audio


##### how to use pipeline
def ex1():
    classifier = pipeline(task="sentiment-analysis")
    results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


def ex2():
    speech_recognizer = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    dataset = load_dataset(path="PolyAI/minds14", name="en-US", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
    result = speech_recognizer(dataset[:4]["audio"])
    print([d["text"] for d in result])


# Use another model and tokenizer in the pipeline
def ex3():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
    result = classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
    print(result)


# AutoTokenizer
def ex4():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
    print(encoding)


def ex5():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pt_batch = tokenizer(
        ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    pt_outputs = pt_model(**pt_batch)
    pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
    print(pt_predictions)
    pt_save_directory = "./data/pt_save_pretrained"
    tokenizer.save_pretrained(pt_save_directory)
    pt_model.save_pretrained(pt_save_directory)
    # when you load pretrained model
    pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
    # Custom model builds
    my_config = AutoConfig.from_pretrained("distilbert-base-uncased", n_heads=12)
    my_model = AutoModel.from_config(my_config)

# fine tuning?
def ex6():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    training_args = TrainingArguments(
        output_dir="./data/traintest",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("rotten_tomatoes")
    def tokenize_dataset(dataset):
        return tokenizer(dataset["text"])
    dataset = dataset.map(tokenize_dataset, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

if __name__ == "__main__":
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    # ex5()
    ex6()
