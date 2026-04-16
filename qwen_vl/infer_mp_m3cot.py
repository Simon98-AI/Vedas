from transformers import AutoTokenizer, AutoProcessor
from qwen_vegas_router_one_patch import VEGAS
# from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from models.modeling_qwen2_vl_router import Qwen2VLForConditionalGeneration
from models.modeling_qwen2_5_vl_router import Qwen2_5_VLForConditionalGeneration
import torch
import argparse
import deepspeed
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
import os, sys
from datasets import load_dataset, Dataset, load_from_disk
from custom_dataset import get_m3cot_dataset
import re
import logging
import json
from tqdm import tqdm
import os
import time
import numpy as np
from datetime import timedelta
logging.basicConfig(
    filename='qwenvl_32_infer_time.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import pdb
import math
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_inference_model(checkpoint_path, model_path, args):
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right"
    )
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start-latent|>",
            "<|end-latent|>",
            "<|latent|>"
        ]
    })
    if args.model_version == 'v_2.5':
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        )
    else:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        )
    base_model.resize_token_embeddings(len(tokenizer))
    processor.tokenizer = tokenizer

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    base_model = get_peft_model(base_model, lora_config)
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    model = VEGAS(
        args, 
        model_path, 
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
        visual_start_id=visual_start_id, 
        visual_end_id=visual_end_id
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    print("Successfully load")
    
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer


def evaluate_and_save(eval_dataset, model, processor, args):
    model.eval()
    correct = 0
    total = 0
    total_generated_tokens = 0 
    total_generate_time = 0.0  

    eval_dataset = get_chunk(eval_dataset, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if args.num_chunks > 1:
        output_name = f"qwen2vl_{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = "full"

    output_path = os.path.join("./output/", f"{output_name}.json")
    cnt = 0
    with open(output_path, "a", encoding="utf-8") as f_out:
        for ex in tqdm(eval_dataset):
            input_text = ex["question_raw"]
            ex["image_raw"] = ex["image_raw"].replace("\\", "/") # ./data/images/physical-commonsense-1426.png
            ex["image_raw"] = os.path.join("./images", ex["image_raw"].split("data/images/")[-1])
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                    {"type": "text", "text": input_text}
                ]
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + "<|latent|>" + "<|latent|>" + "<|latent|>"

            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            img_np = np.array(image_inputs[0])
            ori_image = torch.from_numpy(img_np).permute(2, 0, 1) 

            input_ids = inputs["input_ids"] # [1, xxx]
            prompt_length = input_ids.shape[1]
            generate_start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=torch.tensor(inputs["input_ids"]), 
                    attention_mask=torch.tensor(inputs["attention_mask"]),
                    pixel_values=torch.tensor(inputs["pixel_values"]),
                    ori_image=ori_image,
                    image_grid_thw=torch.tensor(inputs["image_grid_thw"]),
                    max_new_tokens=512
            )
            cnt = cnt + 1
            generate_end_time = time.time()
            sample_generate_time = generate_end_time - generate_start_time
            total_generate_time += sample_generate_time
                        
            generated_tokens = outputs[0, prompt_length:] # 仅仅生成生成的东西
            new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            output_text = processor.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"[OUTPUT] {output_text}")
            
            num_generated_tokens = len(generated_tokens)
            total_generated_tokens += num_generated_tokens

            cleaned_text = re.sub(
                r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                '',
                output_text,
                flags=re.IGNORECASE
            )
            matches = re.finditer(
                r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                cleaned_text,
                flags=re.IGNORECASE | re.DOTALL
            )
            candidates = {match.group(1).upper() for match in matches}
            gt_answer = ex["gt_answer"].strip().upper()

            if gt_answer in candidates: # candidates是正确答案的候选
                correct += 1
                logging.debug(f"correct: True")
            total += 1
            logging.debug(f"[TOTAL] {total}")

            # pdb.set_trace()
            message_question = ex["question_raw"]
            message_question = message_question.replace("<image>", "", 1).replace("Answer:", "", 1).strip()
            message_question = message_question.split("Answer:")[0].strip()

            result = {
                "id": ex["id"],
                "choices": ex["choices"],
                "answer": ex["gt_answer"],
                "domain": ex["domain"],
                "topic": ex["topic"],
                "messages": [
                    message_question,
                    new_generated_text
                ]
            }
            print(result["messages"])
            print(ex["rationale"])
            f_out.write(json.dumps(obj=result, ensure_ascii=False) + "\n")
            f_out.flush() 
            
        avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
        avg_time_per_sample = total_generate_time / total if total > 0 else 0
    
        logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
        logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
        logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    

def format_prompt(example):
    question = example["question"].strip()
    rationale = example["rationale"].replace("\n", " ").strip()
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    user_prompt = (
        f"[Question]:{{{question}}}\n"
        f"[Options]:\n{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, rationale, answer, image

def process_func(example):
    prompt, rationale, answer, image = format_prompt(example)

    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": example["id"],
        "choices": example["choices"],
        "domain": example["domain"],
        "topic": example["topic"]
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.5, help="Show Gradient for Visualization")
    parser.add_argument("--use_tokensr", type=bool, default=True, help="Show Gradient for Visualization")
    parser.add_argument("--pattern", type=str, default="32_patch", help="soft_mix, 16_patch, 32_patch")
    parser.add_argument("--model_version", type=str, default="v_2.5", help="version")
    args = parser.parse_args()

    ## model ##
    if args.model_version == 'v_2.5':
        base_model = "./hub/models/Qwen/Qwen2.5-VL-7B-Instruct" 
        model_sft_checkpoint = "./save_checkpoint/m3cot_VEGAS/epoch_4_full_model_fp32.pth" 
    else:
        base_model = "./hub/models/Qwen/Qwen2-VL-7B-Instruct"
        model_sft_checkpoint = "./save_checkpoint/m3cot_VEGAS/epoch_16_full_model_fp32.pth"
    
    model, processor, tokenizer = load_inference_model(model_sft_checkpoint, base_model, args)

    os.makedirs("output", exist_ok=True)

    ## dataset ##
    dataset = load_dataset(
            "json",
            data_files={
                "train": "./train.jsonl",
                "test": "./test.jsonl"
            }
    )

    val_dataset = dataset["test"]
    val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)
    val_dataset = val_dataset.to_list()
    evaluate_and_save(val_dataset, model, processor, args)

