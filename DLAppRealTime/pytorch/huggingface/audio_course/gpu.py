from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import torch
from torch import cuda
from datasets import Dataset, concatenate_datasets

device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(torch.cuda.get_device_name(torch.device('cuda:0')))
# print(torch.cuda.get_device_name(torch.device('cuda:1')))