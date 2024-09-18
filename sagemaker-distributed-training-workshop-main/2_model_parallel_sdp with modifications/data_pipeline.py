# Importing necessary libraries and modules
import gzip
import json
import os
import h5py
from typing import List, Tuple
import random

from datasets import load_dataset


import numpy as np
import smdistributed.modelparallel.torch as smp
import torch

# Defining the WikiPretrainingDataset class
class WikiPretrainingDataset(torch.utils.data.Dataset): # class for the wiki pretraining dataset
    
    def __init__(self, input_file, max_pred_length): # initialize the dataset with the input file and the maximum prediction length
        # Initializing input file and max prediction length
        self.input_file = input_file # set the input file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        # Denotes the total number of samples
        return len(self.inputs[0])

    def __getitem__(self, index):
        # Getting items for the given index
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            torch.from_numpy(input[index].astype(np.int64))
            if indice < 5
            else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]

        # Processing masked language model labels
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        padded_mask_indices = (masked_lm_positions == 0).nonzero(as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels]

# Defining the OpenwebtextPretrainingDataset class
class OpenwebtextPretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, input_paths: List[str], max_sequence_length=None, zipped=True, use_last_file_only=False):
        # Initializing input paths and dataset parameters
        if zipped:
            for path in input_paths:
                with gzip.open(path, "rt") as f:
                    self.input_data.extend([ln for ln in f])
        else:
            for path in input_paths:
                with open(path, "r") as f:
                    self.input_data.extend([ln for ln in f])

    def __len__(self) -> int:
        # Returns the length of the dataset
        return len(self.input_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Getting item at the specified index
        obj = json.loads(self.input_data[index])
        input_ids = torch.tensor(obj["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(obj["attention_mask"], dtype=torch.long)
        self.actual_sequence_length = len(obj["input_ids"])

        # Truncating sequences if longer than max sequence length
        if self.actual_sequence_length > self.max_sequence_length:
            s_idx = np.random.randint(0, self.actual_sequence_length - self.max_sequence_length)
            e_idx = s_idx + self.max_sequence_length
            input_ids = input_ids[s_idx:e_idx]
            attention_mask = attention_mask[s_idx:e_idx]
        return input_ids, attention_mask
    

    
# class ThepilePretrainingSubset(torch.utils.data.Dataset):
#     def __init__(self, dataset_name: str, split: str, max_sequence_length=None):
#         # Loading dataset from Hugging Face without tokenizing
#         self.dataset = load_dataset(dataset_name, split=split, download_mode="reuse_dataset_if_exists")
#         self.max_sequence_length = max_sequence_length

#     def __len__(self) -> int:
#         # Returns the length of the dataset
#         return len(self.dataset)

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Getting item at the specified index and tokenize it
#         text_data = self.dataset[index]['text']
#         input_ids = text_data['input_ids'].squeeze(0)
#         attention_mask = text_data['attention_mask'].squeeze(0)
#         return input_ids, attention_mask
 
 
 
 
 
class ThepilePretrainingSubset(torch.utils.data.Dataset):
    def __init__(self, input_paths: List[str], max_sequence_length=None, zipped=True, use_last_file_only=False):
        self.input_paths = input_paths                  # List of file paths to read data from
        self.max_sequence_length = max_sequence_length  # Maximum length of sequences to process
        self.zipped = zipped                            # Flag indicating if files are gzip-compressed
        self.use_last_file_only = use_last_file_only    # Flag to decide if only the last file should be used
        self.__read_examples(self.input_paths)          # Read the examples from the input paths

    def __read_examples(self, paths: List[str]):
        self.input_data = []                    # Initialize an empty list to store lines from the files
        if self.zipped:                         # Check if the files are gzip-compressed
            if self.use_last_file_only:         # Check if only the last file should be used
                with gzip.open(paths[-1], "rt") as f:                       # Open the last gzip-compressed file
                    self.input_data = [ln for _, ln in enumerate(f, 1)]     # Read lines from the file and store them
            else:
                for path in paths:                          # Iterate over all file paths
                    with gzip.open(path, "rt") as f:        # Open each gzip-compressed file
                        self.input_data.extend([ln for _, ln in enumerate(f, 1)])  # Read lines and add them to the list
        else:                                           # If files are not gzip-compressed
            if self.use_last_file_only:                 # Check if only the last file should be used
                with open(paths[-1], "r") as f:         # Open the last plain text file
                    self.input_data = [ln for ln in f]  # Read lines from the file and store them
            else:
                for path in paths:                                  # Iterate over all file paths
                    with open(path, "r") as f:                      # Open each plain text file
                        self.input_data.extend([ln for ln in f])    # Read lines and add them to the list

    def __len__(self) -> int:
        return len(self.input_data)  # Return the number of lines read from the files

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = json.loads(self.input_data[index])                   # Load the JSON object at the specified index
        input_ids = torch.tensor(obj["input_ids"], dtype=torch.long)  # Convert "input_ids" to a PyTorch tensor
        attention_mask = torch.tensor(obj["attention_mask"], dtype=torch.long)  # Convert "attention_mask" to a PyTorch tensor

        self.actual_sequence_length = len(obj["input_ids"])         # Determine the length of the sequence
        if self.actual_sequence_length > self.max_sequence_length:  # Check if the sequence needs to be truncated
            s_idx = np.random.randint(0, self.actual_sequence_length - self.max_sequence_length)  # Random starting index for truncation
            e_idx = s_idx + self.max_sequence_length        # Ending index for the truncated sequence
            input_ids = input_ids[s_idx:e_idx]              # Truncate "input_ids" to the specified length
            attention_mask = attention_mask[s_idx:e_idx]    # Truncate "attention_mask" to the specified length
        return input_ids, attention_mask                    # Return the processed input_ids and attention_mask tensors    





            
# Function to create pretraining dataloader
def create_pretraining_dataloader(input_paths: List[str], batch_size: int,  max_sequence_length: int,  seed: int, shuffle: bool = False, data_type="thepile_subset", # data_type="wiki"
                                  ):
    # Creating dataloader based on data type
    if data_type ==  "thepile" or data_type == "thepile_subset":
        data = ThepilePretrainingSubset(input_paths=input_paths, max_sequence_length=max_sequence_length, zipped=zipped, use_last_file_only=use_last_file_only)
    elif data_type == "openwebtext":
        data = OpenwebtextPretrainingDataset(input_paths=input_paths, max_sequence_length=max_sequence_length, zipped=zipped, use_last_file_only=use_last_file_only)
    elif data_type == "wiki":
        if len(input_paths) > 1:
            print(f"Wiki data only support single file when calling create_pretraining_dataloader, reading the first file instead..")
        data = WikiPretrainingDataset(input_file=input_paths[0], max_pred_length=max_sequence_length)
    else:
        raise ValueError(f"Unsupported data type {data_type}")
            
    # Setting up distributed sampler and dataloader
    sampler = torch.utils.data.DistributedSampler(
        data,
        shuffle=shuffle,
        seed=seed,
        rank=smp.dp_rank(),
        num_replicas=smp.dp_size(),
        drop_last=True,
    )
    dataloader = torch.utils.data.DataLoader(
        data,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
