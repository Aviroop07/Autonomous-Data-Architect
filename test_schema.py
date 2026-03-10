import json
from pathlib import Path

# adjust import if needed depending on your project structure
from src.pipeline.stage2.models.schema import SchemaSegment


JSONL_FILE = "converted_rschema.jsonl"


def load_segments(path):
    segments = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                seg = SchemaSegment(**obj)

                # normalize naming (optional but recommended)
                seg.normalize()

                segments.append(seg)

            except Exception as e:
                print(f"❌ Parse error on line {i}: {e}")

    return segments


def validate_segments(segments):
    print("\nSchema validation results:\n")

    total_errors = 0

    for seg in segments:
        errors = seg._validate()

        print(seg.chunk_title, errors)

        total_errors += len(errors)

    print("\n-----------------------")
    print(f"Segments checked: {len(segments)}")
    print(f"Total errors: {total_errors}")

    if total_errors == 0:
        print("✅ ALL SCHEMAS VALID")
    else:
        if errors:
            print(f"❌ {seg.chunk_title}")
            for e in errors:
                print("   ", e)
        else:
            print(f"✅ {seg.chunk_title}")

    


def main():
    path = Path(JSONL_FILE)

    if not path.exists():
        print(f"File not found: {JSONL_FILE}")
        return

    segments = load_segments(path)
    validate_segments(segments)


if __name__ == "__main__":
    main()