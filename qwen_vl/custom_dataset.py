import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from datasets import load_dataset
from transformers import ChameleonProcessor
import pdb
import logging
from itertools import count

logging.basicConfig(
    filename='qwenvl_sqa_4.log',  
    level=logging.DEBUG,          
    format='[%(asctime)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S' 
)


def get_gqa_dataset(dataset, tokenizer, processor, max_size=1000000000):

    def tokenize_sample(sample, max_length=3400):
        image = sample["image"]
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]
        # Tokenize question
        question_tokenized = sample["input_ids"]
        # logging.debug(msg=f"step length: {len(sample["steps"])}")
        # Tokenize steps
        steps_tokenized = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]] # 设置为False会成为纯文本文件

        answer_tokenized = tokenizer.encode(sample["full_answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        
        # Calculate total sequence length
        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )
        print("question length: ", len(question_tokenized))

        # If total length exceeds max_length, truncate steps_tokenized (将推理的步数进行减小)
        if total_length > max_length:
            # Calculate how much to reduce
            excess_length = total_length - max_length
            # Reduce steps_tokenized
            new_steps_tokenized = []
            current_length = 0
            ### 如果所有步数加起来超过max_len，就尽可能的减小一个step的token ###
            for step in steps_tokenized:
                if current_length + len(step) <= (sum(len(s) for s in steps_tokenized) - excess_length):
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized
        # Build the final sample
        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ori_image": sample["ori_image"],
            "idx": sample["idx"],
        }
        
        return sample

    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)]
        else:
            processed_dataset = [None]

        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)

    return dataset


def get_onethink_dataset(dataset, tokenizer, processor, max_size=1000000000):

    def tokenize_sample(sample, max_length=3400):
        image = sample["image"]
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]
        # Tokenize question
        question_tokenized = sample["input_ids"]
        # logging.debug(msg=f"step length: {len(sample["steps"])}")
        # Tokenize steps
        steps_tokenized = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]] # 设置为False会成为纯文本文件
  
        # Tokenize answer
        answer_tokenized = tokenizer.encode(
            "Therefore, the answer is " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        # Calculate total sequence length
        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )
        print("question length: ", len(question_tokenized))

        # If total length exceeds max_length, truncate steps_tokenized (将推理的步数进行减小)
        if total_length > max_length:
            # Calculate how much to reduce
            excess_length = total_length - max_length
            # Reduce steps_tokenized
            new_steps_tokenized = []
            current_length = 0
            ### 如果所有步数加起来超过max_len，就尽可能的减小一个step的token ###
            for step in steps_tokenized:
                if current_length + len(step) <= (sum(len(s) for s in steps_tokenized) - excess_length):
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized
        # Build the final sample
        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ori_image": sample["ori_image"],
            "idx": sample["idx"],
        }
        
        return sample

    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)]
        else:
            processed_dataset = [None]

        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)

    return dataset


def get_m3cot_dataset(dataset, tokenizer, processor, max_size=1000000000):

    def tokenize_sample(sample, max_length=3400):
        image = sample["image"]
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]
        
        processed_question = sample["question"]

        # Tokenize question
        question_tokenized = sample["input_ids"]
        # logging.debug(msg=f"step length: {len(sample["steps"])}")
        # Tokenize steps
        steps_tokenized = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]]
  
        # Tokenize answer
        answer_tokenized = tokenizer.encode(
            "Therefore, the answer is " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        # Calculate total sequence length
        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )
        print("question length: ", len(question_tokenized))

        # If total length exceeds max_length, truncate steps_tokenized (将推理的步数进行减小)
        if total_length > max_length:
            # Calculate how much to reduce
            excess_length = total_length - max_length
            # Reduce steps_tokenized
            new_steps_tokenized = []
            current_length = 0
            ### 如果所有步数加起来超过max_len，就尽可能的减小一个step的token ###
            for step in steps_tokenized:
                if current_length + len(step) <= (sum(len(s) for s in steps_tokenized) - excess_length):
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized
        # Build the final sample
        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ori_image": sample["ori_image"],
            "idx": sample["idx"],
        }
        
        return sample

    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)]
        else:
            processed_dataset = [None]

        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)

    return dataset


def get_sqa_dataset(dataset, tokenizer, processor, max_size=1000000000):

    def tokenize_sample(sample, max_length=3400):
        image = sample["image"]
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]
        
        processed_question = sample["question"]

        # Tokenize question
        question_tokenized = sample["input_ids"]
        # logging.debug(msg=f"step length: {len(sample["steps"])}")
        # Tokenize steps
        steps_tokenized = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]]
        # 将数字转化为A, B, C, D
        sample["answer"] = chr(65 + sample["answer"]) # use when sqa
        # Tokenize answer
        answer_tokenized = tokenizer.encode(
            "Therefore, the answer is " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        # Calculate total sequence length
        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )
        print("question length: ", len(question_tokenized))

        # If total length exceeds max_length, truncate steps_tokenized (将推理的步数进行减小)
        if total_length > max_length:
            # Calculate how much to reduce
            excess_length = total_length - max_length
            # Reduce steps_tokenized
            new_steps_tokenized = []
            current_length = 0
            ### 如果所有步数加起来超过max_len，就尽可能的减小一个step的token ###
            for step in steps_tokenized:
                if current_length + len(step) <= (sum(len(s) for s in steps_tokenized) - excess_length):
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized
        # Build the final sample
        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ori_image": sample["ori_image"],
            "idx": sample["idx"],
        }
        
        return sample

    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)]
        else:
            processed_dataset = [None]

        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        # features: [[], [], ..., []]
        assert self.tokenizer.padding_side == "right"

        earliest_latent = [feature["input_ids"].index(self.latent_id) for feature in features if self.latent_id in feature["input_ids"]]

        if len(earliest_latent) > 0:  
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(self.latent_id)
                else:
                    n_tok_pad = 0

                ### 虽然padding有问题，但是计算loss的时候，不会对这部分的token产生实际的影响 ###
                feature["position_ids"] = [0] * n_tok_pad + list(range(len(feature["input_ids"])))
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None

        position_ids = ([feature["position_ids"] for feature in features] if "position_ids" in features[0].keys() else None)
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        ### padding ###
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = [label + [self.label_pad_token_id] * (max_label_length - len(label)) for label in labels]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = [position_id + [0] * (max_pos_length - len(position_id)) for position_id in position_ids]
            batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.int64)

        return batch

## 构造CoT数据 ##
def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        scheduled_stage_to_train = scheduled_stage # 0 in first epoch

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(len(sample["steps_tokenized"]), configs.max_latent_stage)
        else:
            n_skip_steps, n_latent_tokens = (scheduled_stage_to_train, scheduled_stage_to_train,) # 2

        # 用几个latent token, 就保存max - latent token||如果比max小的话，就采用实际的
        tokens = (
            sample["question_tokenized"]
            + [latent_id] * n_latent_tokens
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )
        
        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
            )
            + tokens[len(sample["question_tokenized"]) + n_latent_tokens:], ## 只有答案不是-100 ##

            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))), ## 位置编码也很简单，如果中间加了一些别的东西，位置编码也要进行相应的改变
            "pixel_values": torch.tensor(sample["pixel_values"]),
            "image_grid_thw": sample["image_grid_thw"],
            "ori_image": sample["ori_image"]
        }

    # 固定套路
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(process_dataset, remove_columns=list(base_dataset.features), num_proc=32)
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        processed_dataset = base_dataset.map(process_dataset, remove_columns=list(base_dataset.features), num_proc=32)
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset
