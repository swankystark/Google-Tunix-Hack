from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
ds = load_from_disk("processed_arc_cot_gemma_final")

print(tokenizer.apply_chat_template(ds[0]["messages"], tokenize=False))
print(tokenizer.apply_chat_template(ds[1]["messages"], tokenize=False))
print(tokenizer.apply_chat_template(ds[2]["messages"], tokenize=False))
print(tokenizer.apply_chat_template(ds[3]["messages"], tokenize=False))
print(tokenizer.apply_chat_template(ds[4]["messages"], tokenize=False))
