import pickle
import json

with open("./hard_ids.pkl", 'rb') as f:
    select_ids = set(pickle.load(f))

with open("./total.jsonl", 'r', encoding='utf-8') as f_in, \
    open("./hard.jsonl", 'w', encoding='utf-8') as f_hard_out, \
        open("./easy.jsonl", 'w', encoding='utf-8') as f_easy_out:
    
    for line in f_in:
        record = json.loads(line.strip())
        if record.get("id") in select_ids: 
            f_hard_out.write(line)
        else:
            f_easy_out.write(line)
