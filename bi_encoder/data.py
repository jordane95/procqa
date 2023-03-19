import collections
import logging
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
import random

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.file_utils import PaddingStrategy

from .arguments import DataArguments


import argparse
from typing import Dict

from datasets import DatasetDict, load_dataset

from typing import List, Dict
from .logger_config import logger

def group_doc_ids(examples: Dict[str, List],
                  negative_size: int,
                  offset: int,
                  use_bm25: bool = False) -> List[int]:
    pos_doc_ids: List[int] = examples['answer_id']
    
    neg_doc_ids: List[List[int]] = []

    if use_bm25:
        neg_doc_ids: List[List[str]] = examples['bm25_answer_ids']
        for ex_neg in neg_doc_ids:
            cur_neg_doc_ids = random.sample(ex_neg, negative_size)
            cur_neg_doc_ids = [int(doc_id) for doc_id in cur_neg_doc_ids]
            neg_doc_ids.append(cur_neg_doc_ids)

    assert len(pos_doc_ids) == len(neg_doc_ids), '{} != {}'.format(len(pos_doc_ids), len(neg_doc_ids))
    assert all(len(doc_ids) == negative_size for doc_ids in neg_doc_ids)

    input_doc_ids: List[int] = []
    for pos_doc_id, neg_ids in zip(pos_doc_ids, neg_doc_ids):
        input_doc_ids.append(pos_doc_id)
        input_doc_ids += neg_ids

    return input_doc_ids


def load_dataset_from_path(path: str):
    train_set = load_dataset("json", data_files=path, split='train[:80%]')
    dev_set = load_dataset("json", data_files=path, split='train[80%:90%]')
    test_set = load_dataset("json", data_files=path, split='train[90%:]')

    dataset = DatasetDict({
        "train": train_set,
        "validation": dev_set,
        "test": test_set,
    })

    corpus: Dict[str, str] = {} # answer id -> answer str
    for k in dataset:
        for item in dataset[k]:
            corpus[str(item["answer_id"])] = item["answer"]

    return dataset, corpus


class RetrievalDataLoader:

    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.negative_size = args.train_group_size - 1
        assert self.negative_size >= 0
        self.tokenizer = tokenizer

        # for training
        self.hf_dataset, self.corpus = load_dataset_from_path(args.data_file)
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # for prediction & evaluation
        self.test_queries_dataset, self.corpus_dataset = self._get_transformed_test_set()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_eval_dataset(self):
        return self.eval_dataset

    def _get_transformed_datasets(self) -> Tuple:
        train_dataset, eval_dataset = None, None

        if "train" not in self.hf_dataset:
            raise ValueError("--do_train requires a train dataset")
        hf_train_dataset = self.hf_dataset["train"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(hf_train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {hf_train_dataset[index]}.")
        self.train_dataset = TrainDatasetForBiE(self.args, hf_train_dataset, self.tokenizer)

        if "validation" not in self.hf_dataset:
            raise ValueError("--do_eval requires a validation dataset")
        hf_eval_dataset = self.hf_dataset["validation"]
        self.eval_dataset = TrainDatasetForBiE(self.args, hf_eval_dataset, self.tokenizer)

        return self.train_dataset, self.eval_dataset
    
    def _get_transformed_test_set(self):
        self.test_dataset = self.hf_dataset["test"]
        self.test_qrels: Dict[str, Dict[str, int]] = {}
        self.test_queries: Dict[str, str] = {}
        for item in self.test_dataset:
            self.test_qrels[str(item["question_id"])] = {str(item["answer_id"]): 1}
            self.test_queries[str(item["question_id"])] = item["title"] + self.tokenizer.sep_token + item["question"]
        
        self.corpus_dataset = PredictionDataset(self.args, self.corpus, self.tokenizer)
        self.test_queries_dataset = PredictionDataset(self.args, self.test_queries, self.tokenizer)
        return self.test_queries_dataset, self.corpus_dataset


class TrainDatasetForBiE(Dataset):
    def __init__(
        self,
        args: DataArguments,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer
    ):
        self.dataset = dataset

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        example = self.dataset[item]
        input_docs = example['answer']

        query_batch_dict = self.tokenizer(
            text=example['title'],
            text_pair=example['question'],
            max_length=self.args.query_max_len,
            padding=False,
            truncation='longest_first',
        )

        doc_batch_dict = self.tokenizer(
            text=input_docs,
            max_length=self.args.passage_max_len,
            padding=False,
            truncation='longest_first',
        )

        # print(self.tokenizer.decode(query_batch_dict['input_ids']))
        # print(self.tokenizer.decode(doc_batch_dict['input_ids']))
        # import pdb; pdb.set_trace()

        return query_batch_dict, doc_batch_dict


def generate_random_neg(qids, pids, k=30):
    qid_negatives = {}
    for q in qids:
        negs = random.sample(pids, k)
        qid_negatives[q] = negs
    return qid_negatives


class PredictionDataset(Dataset):
    def __init__(
        self,
        args: DataArguments,
        texts: Dict,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 256
    ):
        self.args = args
        self.text_ids = list(texts.keys()) # List[str]
        self.encode_data = [texts[i] for i in self.text_ids]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            self.encode_data[item],
            truncation='only_first',
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item


@dataclass
class BiCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]


        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer.pad(
            query,
            padding=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            passage,
            padding=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )

        return {"query": q_collated, "passage": d_collated}


@dataclass
class PredictionCollator(DataCollatorWithPadding):
    is_query: bool = True

    def __call__(
            self, features
    ):
        if self.is_query:
            return {"query": super().__call__(features), "passage": None}
        else:
            return {"query": None, "passage": super().__call__(features)}
