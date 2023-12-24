from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Audio
import numpy as np
import evaluate
from datasets import load_dataset
import torch


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # finetuning 用のデータセットを local から読み込み
    # ここでは UrbanSound8k を使用
    
    
    
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    tokenize_datasets = dataset.map(tokenize_function, batched=True)
