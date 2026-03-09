import json
import re
from collections import defaultdict


INPUT_FILE = "annotation.jsonl"
OUTPUT_FILE = "annotation_cases1.jsonl"


# -----------------------------
# Naming utilities
# -----------------------------

def snake_case(text: str) -> str:
    text = text.strip()
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    text = re.sub(r'[^a-zA-Z0-9]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_').lower()


def upper_snake(text: str) -> str:
    return snake_case(text).upper()


# -----------------------------
# Graph schema representation
# -----------------------------

class SchemaGraph:

    def __init__(self):

        self.tables = {}
        self.columns = defaultdict(set)
        self.pk = {}
        self.fks = []

    # -------------------------
    # Remove forbidden suffix
    # -------------------------
    FORBIDDEN = {"FACT","DIM","ID","ATTR","TABLE"}

    @staticmethod
    def clean_table_name(name):

        tokens = name.split("_")

        tokens = [t for t in tokens if t not in SchemaGraph.FORBIDDEN]

        if not tokens:
            tokens = ["ENTITY"]

        return "_".join(tokens)
    
    # -------------------------
    # Singularize table names (simple heuristic)
    # -------------------------
    @staticmethod
    def singularize(name):

        if name.endswith("IES"):
            return name[:-3] + "Y"

        if name.endswith("SES"):
            return name[:-2]

        if name.endswith("S") and not name.endswith("SS"):
            return name[:-1]

        return name
    
    # -------------------------
    # Column normalization
    # -------------------------
    @staticmethod
    def normalize_column(col):

        col = snake_case(col)

        col = re.sub(r'([a-z])([0-9])', r'\1_\2', col)

        return col

    # -------------------------
    # add table
    # -------------------------

    def add_table(self, table_name, columns, pk):

        t = upper_snake(table_name)

        cols = set(snake_case(c) for c in columns)

        if not pk:
            pk = snake_case(table_name) + "_id"

        pk = snake_case(pk)

        cols.add(pk)

        self.tables[t] = True
        self.columns[t] |= cols
        self.pk[t] = pk

    # -------------------------
    # add FK
    # -------------------------

    def add_fk(self, src_table, src_col, dst_table):

        src_table = upper_snake(src_table)
        dst_table = upper_snake(dst_table)
        src_col = snake_case(src_col)

        if src_table not in self.tables:
            return

        if dst_table not in self.tables:
            return

        self.columns[src_table].add(src_col)

        self.fks.append((src_table, src_col, dst_table))

    # -------------------------
    # repair PK
    # -------------------------

    def repair_primary_keys(self):

        for t in self.tables:

            pk = self.pk[t]

            if pk not in self.columns[t]:

                self.columns[t].add(pk)

    # -------------------------
    # repair foreign keys
    # -------------------------

    def repair_foreign_keys(self):

        valid = []

        for src, col, dst in self.fks:

            if src not in self.tables:
                continue

            if dst not in self.tables:
                continue

            if col not in self.columns[src]:
                self.columns[src].add(col)

            valid.append((src, col, dst))

        self.fks = valid

    # -------------------------
    # detect isolated tables
    # -------------------------

    def repair_isolated_tables(self):

        if len(self.tables) <= 1:
            return

        connected = set()

        for src, col, dst in self.fks:
            connected.add(src)
            connected.add(dst)

        all_tables = set(self.tables)

        isolated = all_tables - connected

        if not isolated:
            return

        anchor = list(all_tables)[0]

        for t in isolated:

            if t == anchor:
                continue

            fk = snake_case(t) + "_id"

            self.columns[t].add(fk)

            self.fks.append((t, fk, anchor))

    # -------------------------
    # remove duplicate columns
    # -------------------------

    def deduplicate(self):

        for t in self.columns:

            self.columns[t] = set(self.columns[t])

    # -------------------------
    # normalize table names
    # -------------------------

    def normalize(self):

        new_columns = defaultdict(set)
        new_pk = {}
        table_map = {}

        for t in list(self.tables.keys()):

            name = upper_snake(t)

            name = SchemaGraph.clean_table_name(name)
            name = SchemaGraph.singularize(name)

            table_map[t] = name

            cols = {SchemaGraph.normalize_column(c) for c in self.columns[t]}

            new_columns[name] |= cols

            new_pk[name] = SchemaGraph.normalize_column(self.pk[t])

        new_fks = []

        for src, col, dst in self.fks:

            src = table_map.get(src, src)
            dst = table_map.get(dst, dst)

            col = SchemaGraph.normalize_column(col)

            new_fks.append((src, col, dst))

        self.tables = {k: True for k in new_columns}
        self.columns = new_columns
        self.pk = new_pk
        self.fks = new_fks
    # -------------------------
    # export
    # -------------------------

    def to_schema_segment(self, chunk_title):

        tables = []

        for t in sorted(self.tables.keys()):

            cols = sorted(self.columns[t])

            pk = self.pk[t]

            if pk not in cols:
                cols.append(pk)

            tables.append({
                "name": t,
                "columns": [{"name": c} for c in cols],
                "pk": pk
            })

        relationships = []

        for src, col, dst in self.fks:

            relationships.append({
                "referencing_table": src,
                "referencing_column": col,
                "referred_table": dst
            })

        return {
            "chunk_title": chunk_title,
            "tables": tables,
            "relationships": relationships if relationships else None
        }



# -----------------------------
# Convert RSchema sample
# -----------------------------

def convert_sample(sample):

    answer = sample.get("answer", {})

    graph = SchemaGraph()

    # tables

    for table_name, table_data in answer.items():

        attrs = table_data.get("Attributes", [])
        pk_list = table_data.get("Primary key", [])

        pk = pk_list[0] if pk_list else None

        graph.add_table(table_name, attrs, pk)

    # foreign keys

    for table_name, table_data in answer.items():

        fk_data = table_data.get("Foreign key", {})

        for col, ref in fk_data.items():

            for ref_table, ref_col in ref.items():

                graph.add_fk(table_name, col, ref_table)

    # repair

    graph.repair_primary_keys()
    graph.repair_foreign_keys()
    graph.repair_isolated_tables()
    graph.deduplicate()
    graph.normalize()

    return graph.to_schema_segment(sample["id"])


# -----------------------------
# Run pipeline
# -----------------------------

def run():

    total = 0
    failed = 0

    with open(INPUT_FILE, "r", encoding="utf8") as f, \
         open(OUTPUT_FILE, "w", encoding="utf8") as out:

        for line in f:

            try:

                sample = json.loads(line)

                converted = convert_sample(sample)

                out.write(json.dumps(converted) + "\n")

                total += 1

            except Exception as e:

                failed += 1
                print("skip:", e)

    print("\nPipeline finished")
    print("Converted:", total)
    print("Failed:", failed)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":

    run()