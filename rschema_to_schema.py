import json
import re
from collections import defaultdict

INPUT_FILE = "annotation.jsonl"
OUTPUT_FILE = "annotation_cases.jsonl"

FORBIDDEN = ["TABLE"]
SINGULAR_EXCEPTIONS = {"STATUS","NEWS","BUS","CLASS","SPECIES","SERIES","DATA"}

# -------------------------
# name utilities
# -------------------------

def to_upper_snake(name):

    name = str(name)

    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)

    name = name.strip("_").upper()

    return name


def to_snake(name):

    name = str(name)

    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)

    return name.strip("_").lower()


# -------------------------
# singularize
# -------------------------

def singularize(word):

    if word in SINGULAR_EXCEPTIONS:
        return word

    if word.endswith("IES"):
        return word[:-3] + "Y"

    if word.endswith("S") and not word.endswith(("SS","US","IS")):
        return word[:-1]

    return word


# -------------------------
# sanitize table name
# -------------------------

def clean_table_name(name):

    name = to_upper_snake(name)

    tokens = name.split("_")

    # remove forbidden suffixes
    tokens = [t for t in tokens if t not in FORBIDDEN]

    if not tokens:
        tokens = ["ENTITY"]

    tokens[-1] = singularize(tokens[-1])

    return "_".join(tokens)


# -------------------------
# schema graph
# -------------------------

class SchemaGraph:

    def __init__(self):

        self.tables = set()
        self.columns = defaultdict(set)
        self.pk = {}
        self.fks = []

    # ---------------------

    def add_table(self,name,cols,pk):

        name = clean_table_name(name)

        self.tables.add(name)

        cols = [to_snake(c) for c in cols]

        if not pk:
            pk = name.lower()+"_id"

        pk = to_snake(pk)

        cols.append(pk)

        self.columns[name].update(cols)

        self.pk[name] = pk

    # ---------------------

    def add_fk(self,src,col,dst):

        src = clean_table_name(src)
        dst = clean_table_name(dst)

        if src not in self.tables or dst not in self.tables:
            return

        col = to_snake(col)

        self.columns[src].add(col)

        self.fks.append((src,col,dst))

    # ---------------------

    def repair_pks(self):

        for t in self.tables:

            pk = self.pk[t]

            self.columns[t].add(pk)

    # ---------------------

    def repair_isolated(self):

        connected = set()

        for s,c,d in self.fks:

            connected.add(s)
            connected.add(d)

        tables = list(self.tables)

        if not tables:
            return

        root = tables[0]

        for t in tables:

            if t not in connected and t != root:

                fk = self.pk[root]

                self.columns[t].add(fk)

                self.fks.append((t,fk,root))

    # ---------------------

    def normalize_columns(self):

        for t in self.columns:

            fixed=set()

            for c in self.columns[t]:

                fixed.add(to_snake(c))

            self.columns[t]=fixed

    # ---------------------

    def build_output(self,chunk):

        tables=[]

        for t in sorted(self.tables):

            cols=sorted(self.columns[t])

            tables.append({
                "name":t,
                "columns":[{"name":c} for c in cols],
                "pk":self.pk[t]
            })

        rel=[]

        for s,c,d in self.fks:

            rel.append({
                "referencing_table":s,
                "referencing_column":c,
                "referred_table":d
            })

        return {
            "chunk_title":chunk,
            "tables":tables,
            "relationships":rel if rel else None
        }


# -------------------------
# convert sample
# -------------------------

def convert_sample(sample):

    graph=SchemaGraph()

    answer=sample["answer"]

    for t,data in answer.items():

        attrs=data.get("Attributes",[])
        pk=data.get("Primary key",[])

        pk=pk[0] if pk else None

        graph.add_table(t,attrs,pk)

    for t,data in answer.items():

        fks=data.get("Foreign key",{})

        for col,ref in fks.items():

            for dst in ref:

                graph.add_fk(t,col,dst)

    graph.repair_pks()
    graph.normalize_columns()
    graph.repair_isolated()

    return graph.build_output(sample["id"])


# -------------------------
# pipeline
# -------------------------

def run():

    total=0

    with open(INPUT_FILE) as f,open(OUTPUT_FILE,"w") as out:

        for line in f:

            sample=json.loads(line)

            seg=convert_sample(sample)

            out.write(json.dumps(seg)+"\n")

            total+=1

    print("Converted",total,"segments")


if __name__=="__main__":
    run()