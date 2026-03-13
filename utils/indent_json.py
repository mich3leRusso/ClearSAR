import json

input_file = "data/ClearSAR/data/annotations/instances_train.json"
output_file = "data/ClearSAR/data/annotations/instances_train_reshaped.json"

# Read the single-line JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Write it back with indentation (tabs)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent="\t", ensure_ascii=False)

print(f"Formatted JSON saved to {output_file}")