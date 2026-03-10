import json

NLDESCFILE = "all_cases.json"
INPUT = "test_cases.jsonl"
OUTPUT = "converted_test_cases.jsonl"


def title_case(x):
    return x.replace("_", " ").title()


def normalize(x):
    return x.strip().upper()


# -------------------------
# Rewrite prompt to ER-style
# -------------------------
def rewrite_prompt(text):

    text = text.strip()

    replacements = [
        ("Generate a", ""),
        ("Generate", ""),
        ("Create a", ""),
        ("Create", ""),
        ("dataset", "system"),
        ("Include", "The system stores"),
    ]

    for a, b in replacements:
        text = text.replace(a, b)

    text = text.strip()

    if not text.endswith("."):
        text += "."

    text += "\n\nDesign a relational database schema for this system."

    return text


# -------------------------
# Load NL descriptions
# -------------------------
with open(NLDESCFILE, "r", encoding="utf-8") as f:
    cases = json.load(f)

prompt_map = {}

for case in cases:
    key = normalize(case["title"])
    prompt_map[key] = rewrite_prompt(case["value"])


# -------------------------
# Convert schema
# -------------------------
def convert_schema(segment):

    schema = {}
    pk_map = {}

    for table in segment["tables"]:
        tname = title_case(table["name"])
        pk_map[tname] = title_case(table["pk"])

    for table in segment["tables"]:

        table_name = title_case(table["name"])

        attrs = [title_case(c["name"]) for c in table["columns"]]

        pk = [title_case(table["pk"])]

        schema[table_name] = {
            "Attributes": attrs,
            "Primary key": pk,
            "Foreign key": {}
        }

    for fk in segment.get("relationships", []):

        src = title_case(fk["referencing_table"])
        dst = title_case(fk["referred_table"])
        col = title_case(fk["referencing_column"])

        schema[src]["Foreign key"][col] = {
            dst: pk_map[dst]
        }

    return schema


# -------------------------
# Convert dataset
# -------------------------
with open(INPUT, "r") as f, open(OUTPUT, "w") as out:

    for line in f:

        seg = json.loads(line)

        chunk = seg["chunk_title"]
        #chunk = normalize(seg["chunk_title"])

        question = prompt_map[chunk]

        sample = {
            "id": seg["chunk_title"],
            "question": question,
            "answer": convert_schema(seg)
        }

        out.write(json.dumps(sample) + "\n")

print("Converted dataset saved to:", OUTPUT)