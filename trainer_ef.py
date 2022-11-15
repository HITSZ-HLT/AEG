from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers.trainer import *
from transformers.utils import logging

from transformers import Seq2SeqTrainer

from copy import deepcopy
import time
import math
import nltk
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from functools import partialmethod
import json
from evaluation import eval_fn

logger = logging.get_logger(__name__)


class EF_Trainer(Seq2SeqTrainer):

    def __init__(
        self,
        model,
        args,
        data_collator,
        train_dataset,
        eval_dataset,
        tokenizer,
        compute_metrics,
        metric,
        data_args,
        test_dataset,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.metric = metric
        self.predict_kw = None
        self.data_args = data_args
        self.test_dataset = test_dataset


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        _ = self.predict(eval_dataset, metric_key_prefix="eval", max_length=max_length, num_beams=num_beams)
        return _['metrics']

    def predict(
        self,
        eval_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_essay_labels = self.tokenizer.batch_decode(
                    eval_dataset['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        eval_kw_labels = self.tokenizer.batch_decode(
                    eval_dataset['kw_labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        self.model.predict_kw = True
        self.predict_kw = True
        self._max_length = 100
        self._min_length = None
        kw_output = eval_loop(
            eval_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        kw_preds = self.tokenizer.batch_decode(
                    kw_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        # t_labels =  kw_output.label_ids
        # t_labels = np.where(t_labels != -100, t_labels, self.tokenizer.pad_token_id)

        kw_input_ids = self.tokenizer(kw_preds, max_length=self.data_args.max_target_length, padding=False, truncation=True)["input_ids"]
        eval_dataset = eval_dataset.remove_columns('kw_labels')
        eval_dataset = eval_dataset.add_column('kw_labels', kw_input_ids)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.predict_kw = False
        self.predict_kw = False
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._min_length = self.data_args.min_length
        output = eval_loop(
            eval_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)


        # save
        eval_essay_preds = self.tokenizer.batch_decode(
                    output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        eval_kw_preds = self.tokenizer.batch_decode(
                    kw_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        self.eval_preds = [pred.strip().replace('\n', ' Ċ ') for pred in eval_essay_preds]
        self.eval_kw_preds = eval_kw_preds
        self.eval_res = eval_fn(eval_kw_labels, eval_essay_labels, eval_kw_preds, eval_essay_preds)

        test_dataset = self.test_dataset
        test_essay_labels = self.tokenizer.batch_decode(
                    test_dataset['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        test_kw_labels = self.tokenizer.batch_decode(
                    test_dataset['kw_labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        test_dataloader = self.get_eval_dataloader(test_dataset)
        self.model.predict_kw = True
        self.predict_kw = True
        self._max_length = 100
        self._min_length = None
        test_kw_output = eval_loop(
            test_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        test_kw_preds = self.tokenizer.batch_decode(
                    test_kw_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

        kw_input_ids = self.tokenizer(test_kw_preds, max_length=self.data_args.max_target_length, padding=False, truncation=True)["input_ids"]
        test_dataset = test_dataset.remove_columns('kw_labels')
        test_dataset = test_dataset.add_column('kw_labels', kw_input_ids)

        test_dataloader = self.get_eval_dataloader(test_dataset)
        self.model.predict_kw = False
        self.predict_kw = False
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._min_length = self.data_args.min_length
        test_output = eval_loop(
            test_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        test_essay_preds = self.tokenizer.batch_decode(
                    test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        test_kw_preds = self.tokenizer.batch_decode(
                    test_kw_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        self.test_preds = [pred.strip().replace('\n', ' Ċ ') for pred in test_essay_preds]
        self.test_kw_preds = test_kw_preds
        self.test_res = eval_fn(test_kw_labels, test_essay_labels, test_kw_preds, test_essay_preds)

        return {
            'predictions': output.predictions,
            'label_ids': output.label_ids,
            'metrics': output.metrics,
            'kw_predictions': kw_output.predictions
        }
        # return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        Seq2SeqTrainer._save_checkpoint(self, model, trial, metrics)

        output_prediction_file = os.path.join(self.args.output_dir, checkpoint_folder, "generated_eval_predictions.txt")
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(self.eval_preds))
        output_kw_prediction_file = os.path.join(self.args.output_dir, checkpoint_folder, "generated_eval_kw_predictions.txt")
        with open(output_kw_prediction_file, "w") as writer:
            writer.write("\n".join(self.eval_kw_preds))
        
        output_prediction_file = os.path.join(self.args.output_dir, checkpoint_folder, "generated_test_predictions.txt")
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(self.test_preds))
        output_kw_prediction_file = os.path.join(self.args.output_dir, checkpoint_folder, "generated_test_kw_predictions.txt")
        with open(output_kw_prediction_file, "w") as writer:
            writer.write("\n".join(self.test_kw_preds))

        with open(os.path.join(self.args.output_dir, checkpoint_folder, 'eval_res.json'), 'w') as fp:
            json.dump(self.eval_res, fp)
        with open(os.path.join(self.args.output_dir, checkpoint_folder, 'test_res.json'), 'w') as fp:
            json.dump(self.test_res, fp)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        # predict_kw: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well       
        if self.data_args.use_top_k:
            gen_kwargs = {
                "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
                "top_k": self.data_args.top_k,
                "temperature": self.data_args.temperature,
                "do_sample": True,
                "num_beams": self._num_beams,
                "min_length": self._min_length,
                "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            }
        else:
            gen_kwargs = {
                "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
                "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
                "min_length": self._min_length,
                "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            }

        if self.tokenizer is not None:
            generation_inputs = {k: v for k, v in inputs.items() if k in self.tokenizer.model_input_names}
            # very ugly hack to make it work
            generation_inputs["input_ids"] = generation_inputs.pop(self.tokenizer.model_input_names[0])
        else:
            generation_inputs = {"input_ids": inputs["input_ids"]}
        
        generation_inputs['kw_labels_input_essay'] = inputs['kw_labels_input_essay']
        generation_inputs['kw_labels_input_essay_mask'] = inputs['kw_labels_input_essay_mask']
        # generation_inputs['predict_kw'] = inputs['predict_kw']

        generated_tokens = self.model.generate(
            **generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        if self.predict_kw == True:
            inputs['decoder_input_ids'] = inputs['kw_decoder_input_ids']
            inputs['labels'] = inputs['kw_labels']
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
