from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_from_disk("output/processed_creative_writing_dataset")

print(f"Dataset size: {len(dataset)} examples")
print(f"\nToken length range: {min(dataset['length'])} - {max(dataset['length'])}")
print(f"Average tokens: {sum(dataset['length'])/len(dataset):.0f}")

# View the first 5 rows as the model will see them
for i in range(min(5, len(dataset))):
    print(f"\n{'='*70}")
    print(f"--- RENDERED ROW {i} ---")
    print(f"Token count: {dataset[i]['length']}")
    print('='*70)
    
    # This applies the template and includes the control tokens
    try:
        rendered = tokenizer.apply_chat_template(dataset[i]['messages'], tokenize=False)
    except:
        # Fallback display
        msgs = dataset[i]['messages']
        rendered = f"SYSTEM: {msgs[0]['content']}\n\nUSER: {msgs[1]['content']}\n\nASSISTANT: {msgs[2]['content']}"
    
    print(rendered)
    
    # Show preview only
    if len(rendered) > 1000:
        print(f"\n... [truncated, full length: {len(rendered)} chars] ...")
