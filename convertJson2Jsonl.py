import json

input_file = "test_cases.jsonl"   # your jsonl file
output_file = "test_cases.json"   # output json file

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:                     # skip empty lines
            data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Converted {len(data)} objects to {output_file}")