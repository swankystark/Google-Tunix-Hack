from datasets import load_dataset
import os

print("Downloading Qwill-RP-CreativeWriting-Reasoning dataset...")
print("="*70)

# Load the full dataset
ds = load_dataset("marcuscedricridia/Qwill-RP-CreativeWriting-Reasoning")

print(f"\nDataset loaded successfully!")
print(f"Train split: {len(ds['train'])} examples")

# Display dataset info
print("\nDataset structure:")
print(ds)

print("\nFirst example fields:")
for key in ds['train'][0].keys():
    print(f"  - {key}")

# Save to local directory
output_dir = "c:/tunix-project/data/qwill_creative_writing"
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving dataset to: {output_dir}")
ds.save_to_disk(output_dir)

# Also save as JSONL for easy processing
output_file = "c:/tunix-project/output/qwill_creative_writing_raw.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"\nExporting to JSONL: {output_file}")
ds['train'].to_json(output_file)

print("\n" + "="*70)
print("✓ Download complete!")
print(f"✓ Saved to disk: {output_dir}")
print(f"✓ Exported JSONL: {output_file}")
print(f"\nTotal examples: {len(ds['train'])}")
print("\nReady for preprocessing!")
