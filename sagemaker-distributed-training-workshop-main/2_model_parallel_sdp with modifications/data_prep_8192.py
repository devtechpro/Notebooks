"""
Download and preprocess the openwebtext dataset using HuggingFace's dataset library


The script imports various libraries and modules for distributed training, including tools for model parallelism (SageMaker, PyTorch, Hugging Face Transformers). 
It defines training and evaluation functions using SageMaker's Model Parallel (SMP) toolkit, optimizing model training with gradient accumulation and data parallelism.
The code sets up data loaders for pretraining and validation, handles checkpointing, logs memory usage, and manages hyperparameters using the 'args' dictionary.
Additionally, loss and perplexity are computed during evaluation, and training progress is controlled by maximum steps and epochs.
"""

import torch
from datasets import load_dataset
# from transformers import GPT2TokenizerFast
from transformers import LlamaTokenizerFast

import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('TF_TOKEN')


# download the unprocessed dataset
# dataset = load_dataset('openwebtext', split='train')
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

dataset_name = "EleutherAI/pile"
subset_name ='enron_emails'
model_id = "meta-llama/Meta-Llama-3-8B"

dataset = load_dataset(dataset_name,subset_name)
tokenizer = LlamaTokenizerFast.from_pretrained(model_id, token=HF_TOKEN)

# Process the dataset and split it into train and test subsets
# dataset = dataset.map(lambda e: tokenizer(e['text'], max_length=512, truncation=True), num_proc=96)
# print(dataset)
# dataset = dataset.filter(lambda e: len(e['input_ids']) >= 512, num_proc=96)
# print(dataset)

# Process the dataset and split it into train and test subsets
tokenizer_max_len = len(tokenizer) - tokenizer.num_special_tokens_to_add()

dataset = dataset.map(lambda e: tokenizer(e['text'], max_length=tokenizer_max_len, truncation=True), num_proc=96) # tokenize the dataset with the tokenizer
print(dataset)
dataset = dataset.filter(lambda e: 50 <= len(e['input_ids']) <= tokenizer_max_len, num_proc=96) # filter the dataset to only include sequences of length between 50 and 8192
print(dataset)


dataset = dataset.remove_columns('text')
if "pile" in dataset_name:
    try:
        dataset = dataset.remove_columns('meta') 
    except:
        # Handle the case where 'meta' column does not exist or other issues
        print("Column 'meta' could not be removed.")
        
        
shuffled_dataset = dataset.shuffle(seed=42)
print(shuffled_dataset)
dataset=shuffled_dataset.train_test_split(test_size=0.1)
print(dataset)

train_dataset=dataset['train']
test_dataset=dataset['test']


# Write the processed dataset into files
# Specify your own path to save the files
# test_path = "/home/ubuntu/openwebtext_seq_512_no_pad_filtered/val" 
# train_path = "/home/ubuntu/openwebtext_seq_512_no_pad_filtered/train"

test_path = ""              # local file paths on the system where code is running.
train_path = ""             # local file paths on the system where code is running.

#  https://huggingface.co/docs/datasets/en/process#sort-shuffle-select-split-and-shard

num_shards=64
print(test_dataset)

for i in range(0, num_shards):
    shard_test=test_dataset.shard(num_shards=num_shards, index=i)
    name=f"{test_path}/test_dataset_8192_filtered_{i}"
    print(name)
    print(shard_test)
    shard_test.to_json(f"{name}.json", orient="records", lines=True)

num_shards=512
print(train_dataset)

for i in range(0, num_shards):
    name=f"{train_path}/train_dataset_8192_filtered_{i}"
    print(name)
    shard=train_dataset.shard(num_shards=num_shards, index=i)
    print(shard)
    shard.to_json(f"{name}.json", orient="records", lines=True)