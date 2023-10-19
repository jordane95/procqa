import os
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    sentence_pooling_method: str = field(default='cls')
    normlized: bool = field(default=False)


@dataclass
class DataArguments:
    sample_neg_from_topk: int = field(
        default=200, metadata={"help": "sample negatives from top-k"}
    )

    data_file: str = field(
        default=None, metadata={"help": "Path to raw data file"}
    )

    prediction_save_path: Union[str] = field(
        default=None, metadata={"help": "Path to save prediction"}
    )

    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    encode_corpus: bool = field(
        default=False,
        metadata={"help": "encode corpus and save to disk"}
    )
    encode_query: bool = field(
        default=False,
        metadata={"help": "encode query and save to disk"}
    )


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=1.0)
    contrastive_loss_weight: Optional[float] = field(default=0.0001)
    freeze_d: bool = field(default=False)
