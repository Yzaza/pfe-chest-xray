#!/usr/bin/env python3
"""
label_chexbert.py  --  Stage 2: weak-label the reports with CheXbert.

Reads iu_split.csv (the frozen split), runs CheXbert over the `report_text`
column, and writes iu_labels.csv: one row per uid with the 14 CheXpert
observations as columns. Then prints a PREVALENCE table so you can see which
classes actually have enough positives to be trainable.

These are WEAK LABELS (CheXbert's automatic extraction from reports), not
verified ground truth. Every downstream stage trains against this file.

CheXbert emits 4 states per observation:
    blank(0) / positive(1) / negative(2) / uncertain(3)  [class indices]
We map to binary with the U-zeros convention (uncertain -> 0), the CheXpert
default, and record it explicitly. `--uncertain one` flips to U-ones if you
want a sensitivity check later.

Usage:
    python3 label_chexbert.py
    python3 label_chexbert.py --uncertain one --out iu_labels_u1.csv
"""

import argparse
import csv
import sys

# The 14 CheXpert observations, in CheXbert's canonical output order.
CHEXPERT_14 = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices",
]


def main():
    ap = argparse.ArgumentParser(description="CheXbert weak-labeling (Stage 2).")
    ap.add_argument("--split_csv", default="iu_split.csv")
    ap.add_argument("--out", default="iu_labels.csv")
    ap.add_argument("--uncertain", choices=["zero", "one"], default="zero",
                    help="map CheXbert 'uncertain' to 0 (default) or 1")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    try:
        import torch
        from f1chexbert import F1CheXbert
    except ImportError as e:
        sys.exit(f"ERROR: missing dep ({e}). Run: pip install f1chexbert")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- load reports from the frozen split -------------------------------
    rows = []
    with open(args.split_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    print(f"Loaded {len(rows):,} studies from {args.split_csv}")

    # ---- load CheXbert -----------------------------------------------------
    print("Loading CheXbert (downloads ~400 MB on first run)...")
    chexbert = F1CheXbert(device=device)

    # ---- label each report -------------------------------------------------
    u_val = 1 if args.uncertain == "one" else 0

    def binarize(codes):
        out = []
        for c in codes:
            if c == 1:
                out.append(1)
            elif c == 3:
                out.append(u_val)
            else:
                out.append(0)
        return out

    labeled = []
    n = len(rows)
    for i, r in enumerate(rows):
        text = r.get("report_text", "") or ""
        try:
            codes = chexbert.get_label(text)
        except Exception as e:
            print(f"  [uid {r['uid']}] labeling failed: {e} -> all zeros")
            codes = [0] * 14
        labeled.append((r["uid"], r["split"], binarize(codes)))
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  labeled {i + 1:,}/{n:,}")

    # ---- write iu_labels.csv ----------------------------------------------
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "split"] + CHEXPERT_14)
        for uid, split, vec in labeled:
            w.writerow([uid, split] + vec)
    print(f"\nWrote {len(labeled):,} rows -> {args.out}  (uncertain -> {u_val})")

    # ---- prevalence table --------------------------------------------------
    print("\n" + "=" * 62)
    print("CLASS PREVALENCE (positive counts)")
    print("=" * 62)
    splits = {"train": [], "val": [], "test": []}
    for _, split, vec in labeled:
        if split in splits:
            splits[split].append(vec)
    n_train = len(splits["train"]) or 1

    print(f"{'observation':<30}{'train+':>8}{'%':>7}{'test+':>9}")
    print("-" * 62)
    for j, name in enumerate(CHEXPERT_14):
        tr_pos = sum(v[j] for v in splits["train"])
        te_pos = sum(v[j] for v in splits["test"])
        pct = 100.0 * tr_pos / n_train
        flag = "" if te_pos >= 10 else "  <- too few in test"
        print(f"{name:<30}{tr_pos:>8}{pct:>6.1f}%{te_pos:>9}{flag}")
    print("-" * 62)
    print("Flagged classes will have noisy/meaningless AUC.")
    print("Report AUC only for supported classes; use AUPRC + CIs.")


if __name__ == "__main__":
    main()
