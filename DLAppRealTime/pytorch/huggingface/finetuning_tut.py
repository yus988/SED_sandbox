from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenize_datasets = dataset.map(tokenize_function, batched=True)
