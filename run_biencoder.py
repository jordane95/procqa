import logging
import os
from pathlib import Path
from typing import Dict
from functools import partial

import numpy as np
import torch
from bi_encoder.modeling import BiEncoderModel
from bi_encoder.trainer import BiTrainer
from bi_encoder.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from bi_encoder.data import RetrievalDataLoader, PredictionDataset, BiCollator, PredictionCollator
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    EvalPrediction,
)
import transformers

# transformers.logging.set_verbosity_error()
import logging
logging.disable(logging.WARNING)

from metrics import accuracy, batch_mrr

logger = logging.getLogger(__name__)

def _compute_metrics(eval_pred: EvalPrediction, eval_group_size: int = 8) -> Dict[str, float]:
    # field consistent with BiencoderOutput 
    preds = eval_pred.predictions
    scores = torch.tensor(preds[-1]).float() # (num_samples, num_samples * eval_group_size) 
    # import pdb; pdb.set_trace()
    labels = torch.arange(0, scores.shape[0], dtype=torch.long) * eval_group_size
    labels = labels % scores.shape[1]
    
    topk_metrics = accuracy(output=scores, target=labels, topk=(1, 3))
    mrr = batch_mrr(output=scores, target=labels)

    
    return {'mrr': mrr, 'acc1': topk_metrics[0], 'acc3': topk_metrics[1]}

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    if training_args.do_train:
        model = BiEncoderModel.build(
            model_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    else:
        model = BiEncoderModel.load(
            model_args.model_name_or_path,
            normlized=model_args.normlized,
            sentence_pooling_method=model_args.sentence_pooling_method
        )

    # Get datasets
    retrieval_dataloader = RetrievalDataLoader(data_args, tokenizer)
    train_dataset = retrieval_dataloader.get_train_dataset()
    eval_dataset = retrieval_dataloader.get_eval_dataset()
    
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=BiCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer,
        compute_metrics=partial(_compute_metrics, eval_group_size=data_args.train_group_size) if training_args.do_eval else None,
    )
    retrieval_dataloader.trainer = trainer

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # # For convenience, we also re-save the tokenizer to the same directory,
        # # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_process_zero():
        #     tokenizer.save_pretrained(training_args.output_dir)
    
    if training_args.do_eval:
        logging.info("*** Evaluation ***")
        metrics = trainer.evaluate(metric_key_prefix='eval')
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    if training_args.do_predict:
        logging.info("*** Prediction ***")
        # if os.path.exists(data_args.prediction_save_path):
        #     raise FileExistsError(f"Existing: {data_args.prediction_save_path}. Please save to other paths")

        if data_args.encode_corpus:
            logging.info("*** Corpus Prediction ***")
            passage_path = os.path.join(data_args.prediction_save_path, 'passage_reps')
            Path(passage_path).mkdir(parents=True, exist_ok=True)

            trainer.data_collator = PredictionCollator(tokenizer=tokenizer, is_query=False)
            test_dataset = retrieval_dataloader.corpus_dataset
            pred_scores = trainer.predict(test_dataset=test_dataset).predictions

            if trainer.is_world_process_zero():
                assert len(test_dataset) == len(pred_scores)
                np.save(os.path.join(passage_path, 'passage.npy'), pred_scores)
                with open(os.path.join(passage_path, 'offset2passageid.txt'), "w") as writer:
                    for offset, cid in enumerate(test_dataset.text_ids):
                        writer.write(f'{offset}\t{cid}\t\n')

        if data_args.encode_query:
            logging.info("*** Query Prediction ***")
            query_path = os.path.join(data_args.prediction_save_path, 'query_reps')
            Path(query_path).mkdir(parents=True, exist_ok=True)

            trainer.data_collator = PredictionCollator(tokenizer=tokenizer, is_query=True)
            test_dataset = retrieval_dataloader.test_queries_dataset
            pred_scores = trainer.predict(test_dataset=test_dataset).predictions

            if trainer.is_world_process_zero():
                assert len(test_dataset) == len(pred_scores)
                np.save(os.path.join(query_path, 'query.npy'), pred_scores)
                with open(os.path.join(query_path, 'offset2queryid.txt'), "w") as writer:
                    for offset, cid in enumerate(test_dataset.text_ids):
                        writer.write(f'{offset}\t{cid}\t\n')
                
                # save qrels
                test_qrels = retrieval_dataloader.test_qrels
                with open(os.path.join(data_args.prediction_save_path, 'qrels.test.tsv'), "w") as writer:
                    for qid in test_qrels:
                        for did, score in test_qrels[qid].items():
                            writer.write(f'{qid}\t{did}\t{score}\n')


if __name__ == "__main__":
    main()
