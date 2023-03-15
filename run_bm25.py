import random
import logging
import argparse
from tqdm import tqdm
from typing import Dict, List
import json

from lexical import BM25Search as BM25

from datasets import load_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Script to mine BM25 hard negatives.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="../data/qa.en.c.json", help="Name of the huggingface dataset")
    parser.add_argument("--top_k", type=int, default=1000, help="Top k BM25 results as negatives.")
    parser.add_argument("--save_path", type=str, default="data/qa.en.c.bm25.json")

    args = parser.parse_args()

    # fix random seed for bm25 hard negative samplilng
    random.seed(args.seed)

    # constract bm25 retriever
    hostname = "http://localhost:9200"
    index_name = "procqa"

    #### Intialize #### 
    # (1) True - Delete existing index and re-index all documents from scratch 
    # (2) False - Load existing index
    initialize = True # False

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
    # SciFact is a relatively small dataset! (limit shards to 1)
    number_of_shards = 1

    # load dataset to search for bm25 negatives
    dataset = load_dataset("json", data_files=args.data_path, split='train')

    queries: Dict[str, str] = {}
    corpus: Dict[str, Dict[str, str]] = {} # answer id -> answer str
    for item in dataset:
        queries[str(item["question_id"])] = " ".join([item["title"], item["question"]])
        corpus[str(item["answer_id"])] = {
            # "title": "",
            "text": item["answer"],
        }
        # question_corpus[item["question_id"]] = {
        #     "title": item["title"],
        #     "text": item["question"],
        # }
    
    retriever = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    results = retriever.search(corpus, queries, args.top_k)

    with open(args.save_path, 'w') as f:
        for item in dataset:
            try:
                hits = results[str(item["question_id"])]
            except:
                hits = {"-1": .0}
            item["bm25_question2answer"] = hits
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()
