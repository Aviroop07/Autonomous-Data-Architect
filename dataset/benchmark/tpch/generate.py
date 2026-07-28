"""
TPC-H Data Generator wrapper.

TPC-H data is produced by the official `dbgen` tool (C source released by TPC).
This script:
  1. Downloads dbgen source from GitHub mirror
  2. Compiles it (Linux/macOS/WSL2 only -- Windows needs WSL2)
  3. Runs dbgen at the requested scale factor
  4. Converts the pipe-delimited .tbl files to CSVs
  5. Optionally mines distributions from numeric columns

Usage
-----
    python dataset/benchmark/tpch/generate.py --scale 1
    python dataset/benchmark/tpch/generate.py --scale 0.1 --mine

Note: compilation requires gcc and make (run in WSL2 on Windows).

TPC-H table -> CSV mapping
--------------------------
  lineitem.tbl  -> LINEITEM.csv
  orders.tbl    -> ORDER.csv
  customer.tbl  -> CUSTOMER.csv
  supplier.tbl  -> SUPPLIER.csv
  part.tbl      -> PART.csv
  partsupp.tbl  -> PARTSUPP.csv
  nation.tbl    -> NATION.csv
  region.tbl    -> REGION.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

DBGEN_REPO = "https://github.com/electrum/tpch-dbgen.git"

# TPC-H column headers (pipe-delimited .tbl files have no header)
TABLE_COLUMNS: Dict[str, List[str]] = {
    "region":   ["r_regionkey", "r_name", "r_comment"],
    "nation":   ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
    "supplier": ["s_suppkey", "s_name", "s_address", "s_nationkey",
                 "s_phone", "s_acctbal", "s_comment"],
    "customer": ["c_custkey", "c_name", "c_address", "c_nationkey",
                 "c_phone", "c_acctbal", "c_mktsegment", "c_comment"],
    "part":     ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
                 "p_size", "p_container", "p_retailprice", "p_comment"],
    "partsupp": ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost",
                 "ps_comment"],
    "orders":   ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice",
                 "o_orderdate", "o_orderpriority", "o_clerk",
                 "o_shippriority", "o_comment"],
    "lineitem": ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
                 "l_quantity", "l_extendedprice", "l_discount", "l_tax",
                 "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
                 "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
}

# CSV output names (UPPER_SNAKE_CASE to match pipeline convention)
TBL_TO_CSV = {
    "region":   "REGION.csv",
    "nation":   "NATION.csv",
    "supplier": "SUPPLIER.csv",
    "customer": "CUSTOMER.csv",
    "part":     "PART.csv",
    "partsupp": "PARTSUPP.csv",
    "orders":   "ORDER.csv",
    "lineitem": "LINEITEM.csv",
}


def clone_and_build(build_dir: Path) -> Path:
    """Clone dbgen repo and compile. Returns path to the dbgen binary."""
    if not build_dir.exists():
        print(f"[build] Cloning dbgen from {DBGEN_REPO} ...")
        subprocess.run(["git", "clone", DBGEN_REPO, str(build_dir)], check=True)
    else:
        print("[build] dbgen source already present — skipping clone")

    dbgen_bin = build_dir / "dbgen"
    if not dbgen_bin.exists():
        print("[build] Compiling dbgen ...")
        subprocess.run(["make"], cwd=str(build_dir), check=True)
    else:
        print("[build] dbgen binary already compiled — skipping make")

    return dbgen_bin


def run_dbgen(dbgen_bin: Path, scale: float, output_dir: Path) -> None:
    """Run dbgen to produce .tbl files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dbgen] Generating TPC-H data at scale factor {scale} ...")
    subprocess.run(
        [str(dbgen_bin), "-vf", "-s", str(scale)],
        cwd=str(output_dir),
        check=True,
    )
    print("[dbgen] Generation complete.")


def convert_tbl_to_csv(output_dir: Path) -> None:
    """Convert pipe-delimited .tbl files to CSV with proper headers."""
    print("[convert] Converting .tbl files to CSV ...")
    for tbl_name, columns in TABLE_COLUMNS.items():
        tbl_path = output_dir / f"{tbl_name}.tbl"
        csv_name = TBL_TO_CSV.get(tbl_name, tbl_name.upper() + ".csv")
        csv_path = output_dir / csv_name

        if not tbl_path.exists():
            print(f"  WARNING: {tbl_path} not found — skipping")
            continue

        rows = 0
        with tbl_path.open(encoding="utf-8") as f_in, \
             csv_path.open("w", encoding="utf-8") as f_out:
            f_out.write(",".join(columns) + "\n")
            for line in f_in:
                # dbgen produces trailing pipe: "val1|val2|...|"
                parts = line.rstrip("\n").rstrip("|").split("|")
                f_out.write(",".join(parts) + "\n")
                rows += 1

        print(f"  {csv_name}: {rows:,} rows")
        tbl_path.unlink()   # remove raw .tbl file


def mine_and_report(output_dir: Path) -> None:
    """Mine distributions from the generated numeric columns."""
    try:
        import pandas as pd
        from src.util.analysis.dist_miner import mine_schema_distributions, to_ground_truth_spec
    except ImportError:
        print("\n[mining] pandas or dist_miner not available — skipping")
        return

    import json

    print("\n[mining] Loading CSVs and mining distributions ...")
    tables = {}

    numeric_cols = {
        "LINEITEM": ["l_quantity", "l_extendedprice", "l_discount", "l_tax"],
        "ORDER":    ["o_totalprice"],
        "CUSTOMER": ["c_acctbal"],
        "SUPPLIER": ["s_acctbal"],
        "PART":     ["p_retailprice", "p_size"],
        "PARTSUPP": ["ps_availqty", "ps_supplycost"],
    }

    for table, cols in numeric_cols.items():
        csv_path = output_dir / f"{table}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        tables[table] = df[cols].dropna()

    if not tables:
        print("[mining] No CSV data — run generation first.")
        return

    mining = mine_schema_distributions(tables)
    spec = to_ground_truth_spec(mining)

    spec_path = output_dir / "mined_distributions.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"[mining] Saved to {spec_path}")
    for key, dist in spec.items():
        print(f"  {key}: {dist['family']} {dist['params']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TPC-H benchmark data")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor (SF): 1 = ~1 GB, 0.1 = ~100 MB (default: 1.0)")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    parser.add_argument("--build-dir", default=None, dest="build_dir",
                        help="Where to clone/build dbgen (default: <script_dir>/dbgen_build)")
    parser.add_argument("--mine", action="store_true",
                        help="Mine distributions after generation")
    parser.add_argument("--skip-build", action="store_true", dest="skip_build",
                        help="Skip clone+build (if dbgen binary already present)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "data"
    build_dir  = Path(args.build_dir)  if args.build_dir  else script_dir / "dbgen_build"

    if sys.platform == "win32":
        print("ERROR: dbgen compilation requires a Unix-like environment.")
        print("Run this script inside WSL2:  wsl python dataset/benchmark/tpch/generate.py")
        sys.exit(1)

    if not args.skip_build:
        dbgen_bin = clone_and_build(build_dir)
    else:
        dbgen_bin = build_dir / "dbgen"
        if not dbgen_bin.exists():
            print(f"ERROR: dbgen binary not found at {dbgen_bin}")
            sys.exit(1)

    run_dbgen(dbgen_bin, args.scale, output_dir)
    convert_tbl_to_csv(output_dir)

    if args.mine:
        mine_and_report(output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
