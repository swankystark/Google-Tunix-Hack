from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
dataset = load_from_disk("processed_math_dataset")

# View the first 2 rows as the model will see them
for i in range(5):
    print(f"\n--- RENDERED ROW {i} ---")
    # This applies the template and includes the control tokens
    rendered = tokenizer.apply_chat_template(dataset[i]['messages'], tokenize=False)
    print(rendered)