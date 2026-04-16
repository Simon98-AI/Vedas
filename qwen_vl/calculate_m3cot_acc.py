import json
import jsonlines
import re
import logging
import glob
import os
import pickle
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def merge_jsonl_files(input_pattern, output_path, sort_by_id=True, remove_duplicates=True):

    input_files = sorted(glob.glob(input_pattern))
    logging.info(f"Found {len(input_files)} files: {[os.path.basename(f) for f in input_files]}")
    
    seen_ids = set()
    merged_count = 0
    duplicate_count = 0
    all_records = []
    
    for filepath in input_files:
        logging.info(f"Processing: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get("id")
                    
                    if remove_duplicates and record_id in seen_ids:
                        duplicate_count += 1
                        logging.debug(f"Duplicate id={record_id} in {filepath}:{line_num}")
                        continue
                    
                    if record_id:
                        seen_ids.add(record_id)
                    
                    all_records.append(record)
                    merged_count += 1
                    
                except json.JSONDecodeError as e:
                    continue
    
    if sort_by_id and all_records and "id" in all_records[0]:
        logging.info("Sorting by id...")
        all_records.sort(key=lambda x: x.get("id", ""))
    
    logging.info(f"Writing {len(all_records)} records to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for record in all_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    
    return merged_count, len(seen_ids)


if __name__ == "__main__":

    INPUT_DIR = "./output/"
    INPUT_PATTERN = os.path.join(INPUT_DIR, "qwen2vl_8_*.json")
    OUTPUT_PATH = os.path.join(INPUT_DIR, "final_merged.jsonl")
    merged, unique = merge_jsonl_files(
        INPUT_PATTERN, 
        OUTPUT_PATH,
        sort_by_id=True,         
        remove_duplicates=True 
    )

    with jsonlines.open(OUTPUT_PATH) as reader:
        data = list(reader)

    correct = 0
    total = 0
    for item in data:
        total = total + 1
        output_text = item['messages'][1] 

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
        gt_answer = item["answer"].strip().upper()

        if gt_answer in candidates:
            correct += 1

    print("Acc:", correct / total)
