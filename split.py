#!/usr/bin/env python3
"""
build_split.py  --  Stage 1: deterministic, grouped, reproducible data split.

Turns the raw ykumards/open-i mirror into the frozen foundation every later
stage reads from. Produces ONE csv (iu_split.csv), one row per usable study:

    uid, split, report_text, report_source, mesh, has_frontal

Design decisions (baked in from the Stage-0 audit + both reviews):
  * Unit of analysis = STUDY, keyed on `uid` (unique per row here, so grouping
    is trivial; frontal+lateral already live in the same row -> no view leakage).
  * report_text = findings + " " + impression, with fallbacks:
        - both present            -> concatenated        (source="findings+impression")
        - findings empty (514)    -> impression only      (source="impression_only")
        - impression empty (31)   -> findings only        (source="findings_only")
        - neither                 -> study DROPPED
  * Split is deterministic (fixed seed) at the STUDY level. NOT called "standard".
  * No patient id exists in this mirror -> we split on uid and DOCUMENT that
    patient-level separation cannot be guaranteed (write it in limitations).

This does NOT need a GPU and downloads nothing new (dataset is cached).

Usage:
    python3 build_split.py                      # 70/15/15, seed 42
    python3 build_split.py --train 0.7 --val 0.1 --test 0.2 --seed 42
    python3 build_split.py --out iu_split.csv
"""

import argparse
import csv
import random
import sys


def clean(s):
    return " ".join(str(s).split()) if isinstance(s, str) else ""


def build_report(findings, impression):
    f, i = clean(findings), clean(impression)
    if f and i:
        return f + " " + i, "findings+impression"
    if i:
        return i, "impression_only"
    if f:
        return f, "findings_only"
    return "", "none"


def main():
    ap = argparse.ArgumentParser(description="Stage 1 grouped split builder.")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="iu_split.csv")
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        sys.exit(f"ERROR: train+val+test must sum to 1.0 (got {total})")

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: pip install 'datasets>=2.0'")

    print("Loading ykumards/open-i (cached)...")
    ds = load_dataset("ykumards/open-i")["train"]

    uids = ds["uid"]
    findings = ds["findings"]
    impressions = ds["impression"]
    mesh = ds["MeSH"]
    frontal = ds["img_frontal"]

    # ---- build usable rows -------------------------------------------------
    rows = []
    source_counts = {"findings+impression": 0, "impression_only": 0,
                     "findings_only": 0, "none": 0}
    no_frontal = 0
    for idx in range(len(ds)):
        text, source = build_report(findings[idx], impressions[idx])
        source_counts[source] += 1
        if source == "none":
            continue  # drop: no usable report
        has_frontal = bool(frontal[idx])
        if not has_frontal:
            no_frontal += 1  # keep row but flag; frontal-only stages filter later
        rows.append({
            "uid": uids[idx],
            "report_text": text,
            "report_source": source,
            "mesh": clean(mesh[idx]),
            "has_frontal": int(has_frontal),
        })

    dropped = source_counts["none"]
    print(f"\nUsable studies : {len(rows):,}  (dropped {dropped} with no report text)")
    print("Report source breakdown:")
    for k in ("findings+impression", "impression_only", "findings_only"):
        print(f"    {k:<22}: {source_counts[k]:,}")
    print(f"    studies with NO frontal image (flagged): {no_frontal}")

    # ---- deterministic grouped split (on uid) ------------------------------
    # uid is unique per study here, so shuffling rows == grouping by study.
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(round(n * args.train))
    n_val = int(round(n * args.val))
    for j, r in enumerate(rows):
        r["split"] = "train" if j < n_train else \
                     "val" if j < n_train + n_val else "test"

    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        counts[r["split"]] += 1
    print(f"\nSplit (seed={args.seed}): "
          f"train={counts['train']:,}  val={counts['val']:,}  test={counts['test']:,}")

    # sanity: no uid appears in two splits (guaranteed by uniqueness, verified anyway)
    per_uid_splits = {}
    for r in rows:
        per_uid_splits.setdefault(r["uid"], set()).add(r["split"])
    leaks = [u for u, s in per_uid_splits.items() if len(s) > 1]
    if leaks:
        sys.exit(f"ERROR: {len(leaks)} uids span multiple splits — grouping broke!")
    print("Leakage check: no uid spans multiple splits. OK")

    # ---- write csv (sorted by uid for stable diffs) ------------------------
    rows.sort(key=lambda r: r["uid"])
    fields = ["uid", "split", "report_source", "has_frontal", "mesh", "report_text"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print(f"\nWrote {len(rows):,} rows -> {args.out}")
    print("This file is the frozen reference for all later stages. Commit it to git.")
    print("\nReminder for the mémoire limitations section:")
    print("  - Split is a deterministic study-level split (seed fixed), NOT an")
    print("    official/published IU split.")
    print("  - No patient id in this mirror -> patient-level separation is not")
    print("    guaranteed; possible repeated-patient leakage is acknowledged.")


if __name__ == "__main__":
    main()
