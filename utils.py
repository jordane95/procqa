import argparse
from typing import Dict

from datasets import DatasetDict, load_dataset



def load_dataset_from_path(path: str):
    train_set = load_dataset("json", data_files=path, split='train[:80%]')
    dev_set = load_dataset("json", data_files=path, split='train[80%:90%]')
    test_set = load_dataset("json", data_files=path, split='train[90%:]')

    dataset = DatasetDict({
        "train": train_set,
        "validation": dev_set,
        "test": test_set,
    })

    corpus: Dict[int, str] = {} # answer id -> answer str
    for k in dataset:
        for item in dataset[k]:
            corpus[item["answer_id"]] = item["answer"]

    return dataset

