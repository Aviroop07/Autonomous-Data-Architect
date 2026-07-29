# ScribbleDB benchmark -- case authoring contract

A case is ONE line of JSON in a `.jsonl` file. It pairs a natural-language
database description with the ground truth a competent data modeller should
produce from that text.

`validate_dataset.py` is the authority. If it reports an error, the case is
wrong, regardless of how reasonable it looks. Run it and iterate until clean:

```
python validate_dataset.py --cases <your file>            # errors + warnings
python validate_dataset.py --cases <your file> --coverage # plus a coverage report
```

Exit code 0 means no errors. Warnings are informational and do not block.

---

## 1. Top-level shape

```json
{
  "id": 42,
  "domain": "Veterinary Clinic",
  "profile": "Semi-Technical",
  "nl_description": "...",
  "ground_truth_schema": { "tables": [...], "relationships": [...] },
  "ground_truth_distributions": { "TABLE.column": {...} },
  "ground_truth_constraints": [ ... ]
}
```

All seven fields are required. `id` must be unique across the whole dataset --
you will be told which id range is yours.

`profile` must be exactly one of:

- `"Non-Technical"` -- a business or domain person. Writes prose, no jargon, no
  types, says "we keep track of" rather than "we store". Often rambling.
- `"Semi-Technical"` -- an analyst or product owner. Knows what a table and an
  ID are, is loose about types and normalisation.
- `"Technical Expert"` -- an engineer or DBA. Names types, keys, cardinalities,
  sometimes distributions and check constraints explicitly.

---

## 2. `ground_truth_schema`

```json
{
  "tables": [
    { "name": "PATIENT",
      "columns": [
        { "name": "patient_id", "data_type": "INTEGER" },
        { "name": "weight_kg",  "data_type": "FLOAT" }
      ],
      "pk": "patient_id" }
  ],
  "relationships": [
    { "referencing_table": "VISIT", "referencing_column": "patient_id",
      "referred_table": "PATIENT", "referred_column": "patient_id" }
  ]
}
```

### Schema size

Across the whole dataset, table counts must span **3 to 50** with a mean near
**10**. Per batch of ten cases that means roughly:

| band | tables | cases per batch |
|------|--------|-----------------|
| small | 3-5 | 3 |
| medium | 6-9 | 5 |
| large | 12-18 | 1 |
| very large | 30-50 | 1 |

The very large one matters disproportionately and is not optional: it is the only
thing in the dataset that exceeds the pipeline's per-shard table budget, so it is
the only thing that exercises cross-shard behaviour at all. Build it as a real
enterprise schema -- a full operational system with its reference/lookup tables,
its junctions, its event and history tables, its audit trail -- not a small
schema with padding. The prose must still read like a person describing a system
they work with, which for something this size means an experienced engineer or a
long handover document rather than a chatty summary.

Hard rules:

- Table names `UPPER_SNAKE_CASE`, **singular** (`PATIENT`, not `PATIENTS`).
- Column names `lower_snake_case`.
- `data_type` is exactly one of:
  `INTEGER`, `VARCHAR`, `FLOAT`, `DECIMAL`, `BOOLEAN`, `DATE`, `DATETIME`,
  `TIMESTAMP`, `TIME`, `TEXT`, `UUID`.
  Money -> `DECIMAL`. Counts -> `INTEGER`. Measurements -> `FLOAT`.
  Short labels/codes/enums -> `VARCHAR`. Long prose -> `TEXT`.
- `pk` is a column name, or a list of column names for a composite key. Every
  name in it must be a column of that table.
- Every FK endpoint must resolve: both tables must exist, and both columns must
  exist on their respective tables. `referred_column` should be the referred
  table's primary key.
- Model a many-to-many association as an explicit junction table with a
  composite PK, since that is what the relational output looks like.

---

## 3. `ground_truth_distributions`

Keyed `"TABLE.column"`; the table and column must exist. Only these seven
families, and the parameter names are exact -- a wrong name is an error:

| family        | params (exactly these)      | notes |
|---------------|-----------------------------|-------|
| `normal`      | `mean`, `std`               | `std` > 0 |
| `lognormal`   | `mean`, `variance`          | log-space; `variance` > 0 |
| `uniform`     | `min`, `max`                | `max` > `min` |
| `poisson`     | `lambda`                    | > 0; counts |
| `exponential` | `lambda`                    | > 0; waiting times |
| `zipf`        | `a`                         | **> 1.0**; heavy tails |
| `categorical` | `weights`                   | map label -> probability, summing to 1.0 (+/- 0.01) |

```json
"PATIENT.weight_kg":  { "distribution": "normal",      "params": { "mean": 18.4, "std": 6.2 } },
"VISIT.cost":         { "distribution": "lognormal",   "params": { "mean": 4.6, "variance": 0.5 } },
"PATIENT.species":    { "distribution": "categorical", "params": { "weights": { "DOG": 0.52, "CAT": 0.38, "OTHER": 0.10 } } }
```

Note the key is `"distribution"`, not `"family"`.

Put distributions only on columns whose spread the text actually says something
about. 3-6 per case is typical; a large schema may carry more.

Nominal (non-numeric) categorical labels are realistic and allowed. The
validator warns that such columns cannot yet be scored at the data level -- that
is a known pipeline gap, not a defect in your case. Use them where they are
natural, and prefer numeric-coded categoricals (`"1"`, `"2"`, ...) for things
genuinely measured on a numeric scale, like a 1-5 rating.

---

## 4. `ground_truth_constraints`

A list. Two types only.

### `range`

```json
{ "type": "range", "table": "APPLICANT", "column": "credit_score",
  "min": 300, "max": 850 }
```

At least one of `min`/`max`; `max` must not be below `min`. An optional
`condition` (same node grammar as below) makes it conditional.

### `ifthen`

```json
{ "type": "ifthen", "table": "LOAN",
  "condition": { "type": "and", "conditions": [
      { "type": "gt", "column": "credit_score", "table_ref": "APPLICANT",
        "join": { "from": "LOAN.applicant_id", "to": "APPLICANT.applicant_id" },
        "value": 750 },
      { "type": "lt", "column": "loan_amount", "value": 50000 } ] },
  "result": { "type": "eq", "column": "interest_rate", "value": 3.5 } }
```

Condition-node grammar:

- Boolean: `{"type": "and"|"or", "conditions": [ ... ]}`, `{"type": "not", "condition": {...}}`
- Leaf: `{"type": "eq"|"neq"|"gt"|"gte"|"lt"|"lte", "column": "...", <rhs>}`
- A leaf defaults to the constraint's own `table`. To reference another table,
  add `"table_ref"` plus a `"join"` with `from`/`to` as `"TABLE.column"`.
- Every column and join endpoint must resolve, or it is an error.

### Right-hand side: exactly one of three forms

A leaf must carry exactly one `<rhs>`. Two of them is an error, and so is none.

**1. A literal.** `"value": 750`

**2. Another column -- a CROSS-COLUMN constraint.** This is the important one,
and most real business rules are of this shape.

```json
{ "type": "lte", "column": "minutes_watched",
  "rhs_column": "runtime_minutes", "rhs_table_ref": "TITLE",
  "rhs_join": { "from": "VIEWING.title_id", "to": "TITLE.title_id" } }
```

`rhs_table_ref` and `rhs_join` are optional -- omit both when the other column
lives on the same table, which is the common case:

```json
{ "type": "gt", "column": "check_out_date", "rhs_column": "check_in_date" }
{ "type": "lte", "column": "amount_paid",   "rhs_column": "amount_due" }
{ "type": "gte", "column": "closed_at",     "rhs_column": "opened_at" }
```

**3. An arithmetic expression over a column.** One operation, `+ - * /`:

```json
{ "type": "lte", "column": "discount_amount",
  "rhs_expr": { "column": "list_price", "op": "*", "value": 0.5 } }

{ "type": "lte", "column": "quantity_picked",
  "rhs_expr": { "column": "quantity_ordered", "op": "-",
                "rhs_column": "quantity_cancelled" } }
```

`rhs_expr` takes `column` (plus optional `table_ref` / `join`), an `op`, and
exactly one of `value` or `rhs_column`.

**Author these heavily.** Cross-column rules are what make a benchmark
discriminating: a date ordering, a quantity that cannot exceed its parent, a
paid amount bounded by an invoiced amount, a measured value inside a
per-specification tolerance, a discount capped as a fraction of list price, a
child count bounded by a declared capacity. Aim for a good half of your
constraints to compare two columns rather than a column and a number, and
include several that cross a join.

Typical volume: 4-12 constraints per case, more for a large schema.

---

## 5. The natural language

This is the part that matters most, and the part a validator cannot check.

- **Write like the profile writes.** A `Non-Technical` description should read
  like a real person explaining their business to a contractor: digressions,
  "oh and", vague quantities, no types. A `Technical Expert` one can name types
  and constraints outright. Vary sentence length and register. Do not write
  three descriptions that share a skeleton.
- **Never enumerate the schema.** If the text reads like a table listing, it is
  wrong. The pipeline's job is to infer structure from prose.
- **Ground truth must be entailed by the text.** Every table, column, FK,
  distribution and constraint must be something a careful reader could derive.
  This is the single most important rule: ground truth the text does not support
  makes the case unscorable, and it is the easiest mistake to make when you
  design the schema first and then write the prose.
- **Conversely, do not leave stated facts out of the ground truth.** If the text
  gives an average, that is a distribution. If it gives a bound, that is a range
  constraint. If it gives a tiered rule, that is an `ifthen`.
- **Ambiguity is a deliberate axis.** When your assignment calls for it, leave
  things genuinely underspecified -- an attribute whose type is unclear, an
  entity mentioned once with no properties, a synonym used for something already
  named, a quantity given only qualitatively. The ground truth should then be
  the *defensible reading* a good modeller would commit to, not a guess only you
  could make. Do not create ambiguity that has no correct answer.
- **Include the mess real specs have** where the profile suits it: typos,
  inconsistent naming for one concept ("client"/"customer"/"user"), an
  afterthought at the end, a requirement stated twice with different numbers
  (and then pick one in the ground truth and make the text's intent recoverable).
- **Data scale belongs in the prose.** "a few hundred members" versus "about
  40 million call records a month" changes what the pipeline should do, so say
  it in the text when your assignment asks for a given scale.

Length guide: `Non-Technical` tends to run long and loose (1500-4000 chars),
`Technical Expert` shorter and denser (600-1500 chars). Do not make every case
the same length.

---

## 6. Output

- One JSON object per line. No trailing commas, no comments, no wrapping array.
- UTF-8, plain ASCII punctuation only: straight quotes, `--` not em-dash, `...`
  not an ellipsis character, `->` not an arrow.
- Write ONLY to the file you are assigned. Do not modify any other file, and do
  not edit `validate_dataset.py` to make your cases pass.
