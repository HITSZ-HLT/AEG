#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import pandas as pd
from pathlib import Path

import logging
import os
import sys
import shutil
from dataclasses import dataclass, field
from typing import Optional
import re
from collections import defaultdict

import datasets
import nltk  # Here to have a nice missing dependency error message early on
nltk.data.path = ['nltk_data']
import numpy as np
from datasets import load_dataset, load_metric

import transformers
import sklearn
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import math
from trainer_ef import EF_Trainer
from model_ef import BART_EF, DataCollatorForSeq2SeqEF
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, ncols=0)

import datetime

now = datetime.datetime.now()
now_time_string = "{:0>4d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}".format(
    now.year, now.month, now.day, now.hour, now.minute, now.second
)
import torch


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.14.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)


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
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    after_pretrain: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=500,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=700,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=700,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    bleu_metric_path: str = field(
        default=None, metadata={"help": "The path of the bleu.py file."}
    )
    use_top_k: bool = field(
        default=False,
    )
    top_k: Optional[int] = field(
        default=40,
    )
    temperature: Optional[float] = field(
        default=0.7,
    )
    min_length: Optional[int] = field(
        default=200,
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    training_args.output_dir = os.path.join(training_args.output_dir, now_time_string)
    # save codes in current dir
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    codes_save_dir = os.path.join(training_args.output_dir, '_codes')
    os.makedirs(codes_save_dir)
    current_dir = os.path.split(os.path.realpath(__file__))[0]
    for src_file_name in os.listdir(current_dir):
        src_file_path = os.path.join(current_dir, src_file_name)
        if os.path.isfile(src_file_path):
            shutil.copy(src_file_path, codes_save_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}

        def replace_LB(ori_fp):
            tmp_df = pd.read_csv(ori_fp)
            tmp_df[data_args.text_column] = tmp_df[data_args.text_column].apply(lambda _: _.replace(' Ċ ', '\n'))
            tmp_df[data_args.summary_column] = tmp_df[data_args.summary_column].apply(lambda _: _.replace(' Ċ ', '\n'))
            modified_fp =  '.'.join(ori_fp.split('.')[:-1]) + '.tmp.' + ori_fp.split('.')[-1]
            tmp_df.to_csv(modified_fp, index=False)
            return modified_fp

        if data_args.train_file is not None:
            data_args.train_file = replace_LB(data_args.train_file)
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_args.validation_file = replace_LB(data_args.validation_file)
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_args.test_file = replace_LB(data_args.test_file)
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # tokenizer.add_tokens(['[KEYWORDS]', '[ESSAY]'])
    # config.essay_token_id = tokenizer.vocab['[ESSAY]'] # 50266
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = BART_EF(bart_model.config)
    print('#### ' + str(model_args.after_pretrain))
    if model_args.after_pretrain or (not training_args.do_train):
        loaded_dict = torch.load(os.path.join(model_args.model_name_or_path, 'pytorch_model.bin'))
        model.resize_token_embeddings(len(tokenizer))
        model.load_dual_bart()
        model.wrap_decoder()
        print(model.load_state_dict(loaded_dict))
    else:
        model.load_state_dict(bart_model.state_dict())
        model.resize_token_embeddings(len(tokenizer))
        model.load_dual_bart()
        model.wrap_decoder()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        # get_kw_fn = lambda x: x[:x.find(' [ESSAY] [ESSAY]')+len(' [ESSAY] [ESSAY]')]
        # get_essay_fn = lambda x: x[x.find(' [ESSAY] [ESSAY] ')+len(' [ESSAY] '):]
        get_kw_fn = lambda x: x[:x.find(' [ESSAY] [ESSAY]')].strip('[KEYWORDS] ')
        get_essay_fn = lambda x: x[x.find(' [ESSAY] [ESSAY] ')+len(' [ESSAY] [ESSAY] '):]
        kw_strs = list(map(get_kw_fn, targets))
        essay_strs = list(map(get_essay_fn, targets))

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(essay_strs, max_length=max_target_length, padding=padding, truncation=True)
        kw_labels = tokenizer(kw_strs, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["kw_labels"] = [tmp[:-1] for tmp in kw_labels["input_ids"]]
        model_inputs["kw_labels"] = kw_labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2SeqEF(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric(data_args.bleu_metric_path)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        preds = [nltk.word_tokenize(pred) for pred in preds]
        labels = [[nltk.word_tokenize(label)] for label in labels]

        return preds, labels

    def make_compute_metrics(mode='eval'):
        if mode == 'eval':
            prompts = raw_datasets['validation']['prompt']
        elif mode == 'test':
            prompts = raw_datasets['test']['prompt']
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            decoded_preds = [pred.strip().replace('\n', ' Ċ ') for pred in decoded_preds]
            decoded_labels = [label.strip().replace('\n', ' Ċ ') for label in decoded_labels]
            # # Some simple post-processing
            # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            # # result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            # # Extract a few results from ROUGE
            # # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

            # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            # result["gen_len"] = np.mean(prediction_lens)
            # (result["precisions_1"], result["precisions_2"],
            # result["precisions_3"], result["precisions_4"]) = (
            #     result["precisions"][0], result["precisions"][1],
            #     result["precisions"][2], result["precisions"][3]
            #     )

            # result["bleu_1"] = result["precisions_1"]
            # result["bleu_2"] = math.exp(sum((1. / 2.) * math.log(p) for p in result["precisions"][:2])) \
            #                     if min(result["precisions"][:2]) > 0 else 0
            # result["bleu_3"] = math.exp(sum((1. / 3.) * math.log(p) for p in result["precisions"][:3])) \
            #                     if min(result["precisions"][:3]) > 0 else 0
            # del result["precisions"]

            
            def extract_kw_essay(pred):
                if '[ESSAY]' in pred:
                    kw_str = pred.split('[ESSAY]')[0].strip()
                    essay = pred.split('[ESSAY]')[-1].strip()
                    if kw_str.startswith('[KEYWORDS]'):
                        kw_str = kw_str.strip('[KEYWORDS]').strip()
                    pattern = re.compile(r'[^ \s]*#[0-9]+')
                    kw_list = pattern.findall(kw_str)
                    kw_list = [kw.split('#')[0] for kw in kw_list]
                    kw_list = list(filter(lambda _: _!='' ,kw_list))
                else:
                    # pattern = re.compile(r'(([^ \s,:]*:[0-9]*, ){30,})')
                    try:
                        pattern = re.compile(r'[^ \s]*#[0-9]+')
                        kw_list = pattern.findall(pred)
                        essay = pred[pred.find(kw_list[-1])+len(kw_list[-1]):].strip().strip('|').strip()
                        kw_list = [kw.split('#')[0] for kw in kw_list]
                        kw_list = list(filter(lambda _: _!='' ,kw_list))
                    except:
                        kw_list = []
                        essay = pred

                return kw_list, essay
            
            keywords_preds, essay_preds = [], []
            for pred in decoded_preds:
                kw_list, essay = extract_kw_essay(pred)
                keywords_preds.append(kw_list)
                essay_preds.append(essay)
            
            keywords_labels, essay_labels = [], []
            for label in decoded_labels:
                kw_list, essay = extract_kw_essay(label)
                keywords_labels.append(kw_list)
                essay_labels.append(essay)

            if essay_labels[0] == '':
                preds_token = [nltk.word_tokenize(pred) for pred in decoded_preds]
                labels_token = [[nltk.word_tokenize(label)] for label in decoded_labels]
            else:
                preds_token = [nltk.word_tokenize(pred) for pred in essay_preds]
                labels_token = [[nltk.word_tokenize(label)] for label in essay_labels]
            essay_result = metric.compute(predictions=preds_token, references=labels_token)

            prediction_lens = [len(pred) for pred in preds_token]
            essay_result["gen_len"] = np.mean(prediction_lens)
            (essay_result["precisions_1"], essay_result["precisions_2"],
            essay_result["precisions_3"], essay_result["precisions_4"]) = (
                essay_result["precisions"][0], essay_result["precisions"][1],
                essay_result["precisions"][2], essay_result["precisions"][3]
                )
            essay_result["bleu_1"] = essay_result["precisions_1"]
            essay_result["bleu_2"] = math.exp(sum((1. / 2.) * math.log(p) for p in essay_result["precisions"][:2])) \
                                if min(essay_result["precisions"][:2]) > 0 else 0
            essay_result["bleu_3"] = math.exp(sum((1. / 3.) * math.log(p) for p in essay_result["precisions"][:3])) \
                                if min(essay_result["precisions"][:3]) > 0 else 0
            del essay_result["precisions"]
        
        


 
            return essay_result
        return compute_metrics

    # Initialize our Trainer
    trainer = EF_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics('eval') if training_args.predict_with_generate else None,
        metric=metric,
        data_args=data_args,
        test_dataset=predict_dataset
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_results = trainer.predict(eval_dataset, max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics = eval_results['metrics']
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                eval_predictions = tokenizer.batch_decode(
                    eval_results['predictions'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                eval_predictions = [pred.strip().replace('\n', ' Ċ ') for pred in eval_predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_eval_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(eval_predictions))

    if training_args.do_predict:
        logger.info("*** Predict ***")

        trainer.compute_metrics = make_compute_metrics('test')

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results['metrics']
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                kw_predictions = tokenizer.batch_decode(
                    predict_results['kw_predictions'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                output_kw_prediction_file = os.path.join(training_args.output_dir, "generated_test_kw_predictions.txt")
                with open(output_kw_prediction_file, "w") as writer:
                    writer.write("\n".join(kw_predictions))

                predictions = tokenizer.batch_decode(
                    predict_results['predictions'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip().replace('\n', ' Ċ ') for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_test_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
