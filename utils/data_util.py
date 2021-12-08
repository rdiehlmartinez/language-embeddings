__author__ = 'Richard Diehl Martinez'
'''Utilities for preprocessing and loading dataset '''

import torch 
from torch.utils.data import IterableDataset, DataLoader
from transformers import XLMRobertaTokenizer
from itertools import cycle

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

class BaseIterableDataset(IterableDataset): 
    ''' Iterable dataset that reads data from a provided file path '''
    def __init__(self, fp, cycle_data=False): 
        super().__init__()
        self.fp = fp 
        self.cycle_data = cycle_data

    def get_stream(self):
        with open(self.fp, 'r') as f:
            for line in f:
                yield tokenizer(line)

    def __iter__(self): 
        if self.cycle_data:
            return cycle(self.get_stream())
        else:
            return self.get_stream()


# Utility functionas for data

def custom_collate(batch):
    batch_size = len(batch)
    max_len = max([len(sample["input_ids"]) for sample in batch])

    # NOTE: Padding token needs to be 1, in order to be consistent 
    # with hugging face tokenizer: 
    # https://huggingface.co/transformers/model_doc/xlmroberta.html#transformers.XLMRobertaTokenizer
    padded_input = torch.ones((batch_size, max_len))

    for idx, sample in enumerate(batch):
        padded_input[idx, :len(sample["input_ids"])] = torch.tensor(sample["input_ids"])

    mask = (padded_input != 1)

    processed_batch = {
        'attention_mask': mask.int(),
        'input_ids': padded_input.long(),
    }

    return processed_batch

def get_dataloader(fp, **kwargs):
    ''' Helper function that wraps BaseIterableDataset  '''
    dataset = BaseIterableDataset(fp, **kwargs)
    return DataLoader(dataset,
                        batch_size=32, # TODO: let users override this 
                        collate_fn=custom_collate)


def move_to_device(batch, device):
    updated_batch = {}
    for key, val in batch.items():
        if isinstance(val, dict):
            if key not in updated_batch:
                updated_batch[key] = {}
            for sub_key, sub_val in val.items():
                if sub_val is not None:
                    updated_batch[key][sub_key] = sub_val.to(device)
        else:
            if val is not None:
                updated_batch[key] = val.to(device)
    return updated_batch