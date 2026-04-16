import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import timedelta
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.optim import AdamW
import shutil
import pickle
import numpy as np
from torch.utils.data import Subset
from collections import OrderedDict
import re
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoProcessor
from models.modeling_qwen2_vl_router import Qwen2VLForConditionalGeneration
from models.modeling_qwen2_5_vl_router import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset, load_from_disk, config

import logging
logging.basicConfig(
    filename='qwenvl4.log',  
    level=logging.DEBUG,          
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S' 
)

from qwenvl_vegas import VEGAS
from dataset import (
    get_m3cot_dataset,
    get_onethink_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import os, sys
import yaml
import argparse
from utils import Config, set_seed
import pdb
from peft import LoraConfig, get_peft_model

# LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)

def main():
    print("Initializing DeepSpeed Training!")
    parser = argparse.ArgumentParser(description="vegas")
    parser.add_argument("config_file")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", default="ds_config.json", help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    parser.add_argument("--collect_grad", type=bool, default=False, help="Show Gradient for Visualization")
    parser.add_argument("--use_data_flag", type=str, default="full", help="use different training subset")
    parser.add_argument("--progressive", type=bool, default=False, help="use curriculum learning")
    parser.add_argument("--ratio", type=float, default=0.2, help="use weight of self-distillation loss")
    parser.add_argument("--use_tokensr", type=bool, default=True, help="use tokenSR module")
    parser.add_argument("--pattern", type=str, default="32_patch", help="soft_mix, 16_patch, 32_patch")
    parser.add_argument("--model_version", type=str, default="v_2", help="v2, v2_5")
    args = parser.parse_args()

    if args.model_version == 'v_2.5':
        model_path = "./modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct"
    else:
        model_path = "./modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct"

    # Initialize DeepSpeed
    deepspeed.init_distributed()
    local_rank = args.local_rank
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        
        
    print("start loading vegas model")
    if args.model_version == 'v_2':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager")
    elif args.model_version == 'v_2.5':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.lr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    # print("latent_id: ", latent_id) # 151659
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")

    # initialize the new token embeddings with a known token, it helps stablize the training
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[token_id]
        embeddings.weight.data[token_id] = target_embedding
        # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
    # model.print_trainable_parameters()

    model = VEGAS(args, model_path, model, latent_id, start_id, end_id, tokenizer.eos_token_id, image_token_id, visual_start_id, visual_end_id)

    print(f"Running Deepspeed on rank = {rank}, world size = {world_size}")
    model = model.to(rank)
    
    if configs.bf16:
        model.to(torch.bfloat16)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        # optimizer = optimizer,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters())
    )
    
    # check
    # model_engine.module.base_causallm.model.model.set_tracking_epoch(1)
    # model_engine.module.base_causallm.model.model.qkv_collector.start_tracking(1)

    del model

    dataset = load_dataset(
            "json",
            data_files={
                "train": "./train.jsonl",
                "test": "./test.jsonl"
            }
    )

    def process_example(example):

        example["image"] = example["image"].replace("\\", "/") # ./data/images/physical-commonsense-1426.png
        example["image"] = os.path.join("./m3cot_data/images", example["image"].split("data/images/")[-1])

        ### process the reasoning steps ###
        rationale = example["rationale"].replace("\n", " ").strip() # remove head and tail
        example["steps"] = rationale.split(". ")

        if example["steps"][-1] == "":
            example["steps"].pop()

        if len(example["steps"]) > 3:
            total_steps = len(example["steps"])
            step_size = total_steps // 3
            remainder = total_steps % 3

            new_steps = []
            start = 0

            for i in range(3):
                end = start + step_size + (1 if i < remainder else 0)
                new_steps.append(". ".join(example["steps"][start:end]))
                start = end

            example["steps"] = new_steps


        question = example["question"]
        choices = example["choices"]
        
        choices_str = "[Options]:\n"+"\n".join([
            f"({chr(65 + i)}).{{{choice.strip()}}}"
            for i, choice in enumerate(choices)
        ])

        question = question
        question_with_braces = f"{{{question.strip()}}}"
        prefix_str = "Answer:"
        example["question"] = f"[Question]:{question_with_braces}\n{choices_str}\n{prefix_str}\n"

        del example["rationale"]
        del example["choices"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"], "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": example["question"]}
            ]
        }]

        example["question"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # <|vision_start|><|image_pad|><|vision_end|>
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[example["question"]],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.tolist() for k, v in inputs.items()}
        example["input_ids"] = torch.tensor(inputs["input_ids"][0])
        example["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
        example["pixel_values"] = torch.tensor(inputs["pixel_values"])
        img_np = np.array(image_inputs[0])
        example["ori_image"] = torch.from_numpy(img_np).permute(2, 0, 1) 

        return example

    def has_image(example):
        return ("image" in example and example["image"] is not None)

    data_len = len(dataset["train"])
    use_cache = True
    
    if use_cache:
        train_dataset = load_from_disk("./datasets")
    else:
        train_dataset = dataset["train"].select(range(data_len)).filter(has_image)
        train_dataset = train_dataset.map(process_example, num_proc=4) # 32
        train_dataset.save_to_disk("./datasets")


    print("Datatset have been Loaded!")
    base_dataset_train = get_m3cot_dataset(train_dataset, tokenizer, processor, max_size=5000 if configs.debug else 100000000)
    total_train_steps = 0

    if not configs.debug and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None


    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):
        if args.progressive:
            if configs.num_epochs  == 16:
                scheduled_stage = epoch // configs.epochs_per_stage # totally 4 stages
            else:
                scheduled_stage = 4 * epoch // configs.epochs_per_stage # totally 4 stages
        else:
            scheduled_stage = 4

        np.random.seed(epoch) 

        dataset_train = get_cot_latent_dataset(
            scheduled_stage,
            base_dataset_train,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=True,
            shuffle=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=configs.batch_size_training, # 64
            collate_fn=collator,
            sampler=DistributedSampler(dataset_train, shuffle=True),
        )

        if args.collect_grad:
            model_engine.module.base_causallm.model.model.set_tracking_epoch(epoch)
            model_engine.module.base_causallm.model.model.qkv_collector.start_tracking(epoch)
            
        model_engine.train()
        total_length = len(train_dataloader) // configs.gradient_accumulation_steps 
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(train_dataloader):
            if step == 0 and wandb_run and rank == 0:
                print("logging training data")
                cur_bs = len(batch["input_ids"])
                text_str = ""
                for data_idx in range(cur_bs):
                    for token_idx in range(len(batch["input_ids"][data_idx])):
                        text_str += (
                            str(batch["input_ids"][data_idx][token_idx].item())
                            + " "
                            + str(batch["labels"][data_idx][token_idx].item())
                            + " "
                            + tokenizer.decode(
                                batch["input_ids"][data_idx][token_idx]
                            )
                            + "\n"
                        )
                    text_str += "====" * 10 + "\n"

                text_table.add_data(total_train_steps, text_str)

            total_train_steps += 1
            batch = {
                key: batch[key].to(rank) for key in batch.keys() if key != "idx"
            }

            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            
            if wandb_run and rank == 0:
                log_dict = {
                    "train/epoch": epoch + 1,
                    "train/step": epoch * len(train_dataloader) + step,
                    "train/loss": loss.detach().float()
                }
                wandb_run.log(log_dict)

            if rank == 0:
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float()), 4)}"
                )
        if args.collect_grad:
            model_engine.module.base_causallm.model.model.finish_epoch()
            # save qkvo
            epoch_stats = model_engine.module.base_causallm.model.model.qkv_collector.end_epoch()
            model_engine.module.base_causallm.model.model.qkv_collector._save_epoch_aggregated(epoch_stats, f"./grad_stats_epoch_{epoch}.pkl")

        pbar.close()
        dist.barrier()
    
    ### collect the gradient ###
    if args.collect_grad:
        gradient_data = model_engine.module.base_causallm.model.model.gradient_collector.get_data()
        with open("./{}_grad_data.pkl".format(args.use_data_flag), 'wb') as f:
            pickle.dump(gradient_data, f)
            print("data gradient has been saved...")

    ### save for re-training ###
    epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}_checkpoint")
    model_engine.save_checkpoint(
        save_dir=epoch_save_dir,
        tag=f"epoch_{epoch+1}_zero3_bf32",
        client_state={"best_acc": best_acc, "current_epoch": epoch+1}
    )
    
    ### save for inference ###
    if rank == 0:
        fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(epoch_save_dir, tag=f"epoch_{epoch+1}_zero3_bf32")
        fp32_output = os.path.join(save_dir, f"epoch_{epoch+1}_full_model_fp32.pth") 
        torch.save(fp32_state_dict, fp32_output)        
        print(f"Epoch {epoch+1} FP32 save to {fp32_output}")


if __name__ == "__main__":
    main()