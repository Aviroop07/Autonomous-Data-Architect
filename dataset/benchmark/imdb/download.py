"""
Download IMDb Non-Commercial Datasets and convert to CSVs.

Source : https://datasets.imdb.com/
License: Non-commercial use only (https://www.imdb.com/interfaces/)

Files downloaded
----------------
title.basics      -> TITLE_BASIC.csv
title.ratings     -> TITLE_RATING.csv
title.episode     -> TITLE_EPISODE.csv
title.akas        -> TITLE_AKA.csv
title.principals  -> TITLE_PRINCIPAL.csv
title.crew        -> TITLE_CREW.csv
name.basics       -> NAME_BASIC.csv

Usage
-----
    python dataset/benchmark/imdb/download.py
    python dataset/benchmark/imdb/download.py --output-dir dataset/benchmark/imdb/data
    python dataset/benchmark/imdb/download.py --sample 100000   # first 100k rows per file
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

BASE_URL = "https://datasets.imdbws.com"

IMDB_FILES = [
    ("title.basics.tsv.gz",     "TITLE_BASIC.csv"),
    ("title.ratings.tsv.gz",    "TITLE_RATING.csv"),
    ("title.episode.tsv.gz",    "TITLE_EPISODE.csv"),
    ("title.akas.tsv.gz",       "TITLE_AKA.csv"),
    ("title.principals.tsv.gz", "TITLE_PRINCIPAL.csv"),
    ("title.crew.tsv.gz",       "TITLE_CREW.csv"),
    ("name.basics.tsv.gz",      "NAME_BASIC.csv"),
]

NULL_TOKEN = r"\N"


def _progress_hook(downloaded: int, chunk: int, total: int) -> None:
    if total > 0:
        pct = min(100, downloaded * chunk * 100 // total)
        bar = "#" * (pct // 5) + "." * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download_and_convert(
    gz_name: str,
    csv_name: str,
    output_dir: Path,
    sample: Optional[int] = None,
    force: bool = False,
) -> None:
    csv_path = output_dir / csv_name
    if csv_path.exists() and not force:
        print(f"  {csv_name} already exists — skipping (use --force to re-download)")
        return

    url = f"{BASE_URL}/{gz_name}"  # e.g. https://datasets.imdbws.com/title.basics.tsv.gz
    gz_path = output_dir / gz_name
    print(f"\n[{gz_name}]")

    # Download
    print(f"  Downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, gz_path, reporthook=_progress_hook)
        print()
    except Exception as e:
        print(f"\n  ERROR downloading {gz_name}: {e}")
        return

    # Decompress and convert TSV -> CSV (replace \N with empty, tab -> comma)
    print(f"  Converting to {csv_name} ...")
    rows_written = 0
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8") as gz_f, \
             csv_path.open("w", encoding="utf-8") as csv_f:
            for i, line in enumerate(gz_f):
                if sample is not None and i > sample:
                    break
                # Replace IMDb null sentinel with empty string, then convert TSV to CSV
                parts = line.rstrip("\n").split("\t")
                parts = [("" if p == NULL_TOKEN else p) for p in parts]
                # Minimal quoting: wrap fields containing commas in double quotes
                quoted = []
                for p in parts:
                    if "," in p or '"' in p or "\n" in p:
                        p = '"' + p.replace('"', '""') + '"'
                    quoted.append(p)
                csv_f.write(",".join(quoted) + "\n")
                rows_written += 1
    except Exception as e:
        print(f"  ERROR converting {gz_name}: {e}")
        return
    finally:
        gz_path.unlink(missing_ok=True)

    print(f"  Done — {rows_written:,} rows written to {csv_path}")


def mine_and_report(output_dir: Path) -> None:
    """After download, mine distributions for key numeric columns."""
    try:
        import pandas as pd
        from src.util.analysis.dist_miner import mine_schema_distributions, to_ground_truth_spec
    except ImportError:
        print("\n[mining] pandas or src.util.dist_miner not available — skipping mining step")
        return

    print("\n[mining] Loading CSVs and mining distributions...")
    tables = {}

    # title.ratings — numeric columns of interest
    ratings_path = output_dir / "TITLE_RATING.csv"
    if ratings_path.exists():
        df = pd.read_csv(ratings_path, nrows=500_000)
        df.columns = [c.lower() for c in df.columns]
        for col in ("averagerating", "numvotes"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        tables["TITLE_RATING"] = df[["averagerating", "numvotes"]].dropna()

    # title.basics — runtimeminutes
    basics_path = output_dir / "TITLE_BASIC.csv"
    if basics_path.exists():
        df = pd.read_csv(basics_path, nrows=500_000)
        df.columns = [c.lower() for c in df.columns]
        if "runtimeminutes" in df.columns:
            df["runtimeminutes"] = pd.to_numeric(df["runtimeminutes"], errors="coerce")
        tables["TITLE_BASIC"] = df[["runtimeminutes"]].dropna()

    if not tables:
        print("[mining] No CSV data found — run download first.")
        return

    mining = mine_schema_distributions(tables)
    spec = to_ground_truth_spec(mining)

    import json
    spec_path = output_dir / "mined_distributions.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"[mining] Mined distributions saved to {spec_path}")

    for key, dist in spec.items():
        print(f"  {key}: {dist['family']} {dist['params']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IMDb datasets")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Where to save CSVs (default: same dir as this script / data/)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Max rows per file (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if CSV already exists")
    parser.add_argument("--mine", action="store_true",
                        help="After download, mine distributions from numeric columns")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Subset of file stems to download (e.g. title.basics title.ratings)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = IMDB_FILES
    if args.files:
        stems = set(args.files)
        files_to_download = [(gz, csv) for gz, csv in IMDB_FILES
                             if any(gz.startswith(s) for s in stems)]

    print(f"Output directory : {output_dir}")
    print(f"Files to download: {len(files_to_download)}")
    if args.sample:
        print(f"Row limit        : {args.sample:,} per file")

    for gz_name, csv_name in files_to_download:
        download_and_convert(gz_name, csv_name, output_dir,
                             sample=args.sample, force=args.force)

    if args.mine:
        mine_and_report(output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
