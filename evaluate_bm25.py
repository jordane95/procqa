import random
import logging
import argparse
from tqdm import tqdm
from typing import Dict, List
import json

import BM25Search as BM25

from datasets import load_dataset

from .custom_metrics import mrr, recall_cap


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Script to evaluate the BM25 performance.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="data/qa.en.c.bm25.list.json", help="Path to the dataset with bm25 scores.")
    parser.add_argument("--top_k", type=int, default=1000, help="Top k BM25 results as negatives.")
    args = parser.parse_args()

    # fix random seed for bm25 hard negative samplilng
    random.seed(args.seed)

    # load dataset to search for bm25 negatives
    test_set = load_dataset("json", data_files=args.data_path, split='train[90%:]')

    qrels: Dict[str, Dict[str, int]] = {}
    results: Dict[str, Dict[str, float]] = {}

    for item in test_set:
        gold_answer_id = str(item["answer_id"])
        qrels[str(item["question_id"])] = {gold_answer_id: 1}
        bm25_answer_ids = [str(answer_id) for answer_id in item["bm25_answer_ids"]]
        bm25_scores = item["bm25_answer_scores"]
        results[str(item["question_id"])] = dict(zip(bm25_answer_ids, bm25_scores))
    
    k_values = [1, 10, 50, 100]
    m = mrr(qrels, results, k_values)
    r = recall_cap(qrels, results, k_values)



if __name__ == "__main__":
    main()
