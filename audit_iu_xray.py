#!/usr/bin/env python3
"""
audit_iu_xray.py  --  Stage 0 dataset audit for the PFE.

Answers, for the HuggingFace mirror `ykumards/open-i`, the questions that
neither project review resolved and that determine your Stage 1 parser and
Stage 4 fusion config:

  1. How many rows are there really? (studies vs images)
  2. What splits exist? (the mirror may expose only `train`)
  3. Are reports raw XML or already-parsed columns? -> do you need a parser?
  4. What image fields exist? (frontal / lateral, decoded PIL vs paths)
  5. Are there any labels shipped, or must CheXbert produce all of them?
  6. What is the report-length distribution? -> sets ClinicalBERT max_length
  7. Is there a stable per-study id for the grouped split?

Runs on CPU on your laptop. No GPU, no training, no cost.

Usage:
    pip install "datasets>=2.0" pillow
    python3 audit_iu_xray.py                 # downloads + audits the mirror
    python3 audit_iu_xray.py --samples 3     # dump N full sample reports
    python3 audit_iu_xray.py --local DIR     # audit an already-downloaded copy
"""

import argparse
import sys
from collections import Counter


# ----------------------------------------------------------------------------- helpers
def rule(char="-", n=78):
    print(char * n)


def header(title):
    print()
    rule("=")
    print(title)
    rule("=")


def pctl(sorted_vals, p):
    """Simple percentile (linear interpolation), no numpy dependency."""
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def feature_kind(feat):
    """Best-effort classification of a datasets feature into image/text/other,
    robust across datasets versions (compares by class name, not import)."""
    name = type(feat).__name__
    if name == "Image":
        return "image"
    if name == "Value":
        dtype = getattr(feat, "dtype", "")
        if dtype == "string":
            return "text"
        return f"value:{dtype}"
    if name == "ClassLabel":
        return "label"
    if name in ("Sequence", "LargeList", "List"):
        return f"sequence[{feature_kind(getattr(feat, 'feature', None))}]"
    return name


def looks_like_xml(s):
    if not isinstance(s, str):
        return False
    s = s.strip()
    return s.startswith("<") and (">" in s) and ("</" in s or "/>" in s)


# ----------------------------------------------------------------------------- loading
def load(args):
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        sys.exit("ERROR: `datasets` not installed. Run: pip install 'datasets>=2.0' pillow")

    import datasets as _d
    print(f"datasets version: {_d.__version__}")

    if args.local:
        print(f"Loading from local disk: {args.local}")
        try:
            return load_from_disk(args.local)
        except Exception as e:  # noqa: BLE001
            print(f"load_from_disk failed ({e}); trying load_dataset(path=...)")
            return load_dataset(args.local)

    print("Loading ykumards/open-i from HuggingFace (first run downloads ~2 GB)...")
    try:
        return load_dataset("ykumards/open-i")
    except Exception as e:  # noqa: BLE001
        print(f"\nload_dataset failed: {e}")
        print("If this is an auth/gated error, run `huggingface-cli login` first.")
        print("If it needs a config, inspect the card and pass one manually.")
        sys.exit(1)


def as_splits(ds):
    """Normalise to a dict {split_name: dataset}."""
    # DatasetDict behaves like a dict; a bare Dataset does not.
    if hasattr(ds, "keys") and hasattr(ds, "items") and not hasattr(ds, "features"):
        return dict(ds.items())
    if hasattr(ds, "column_names") and hasattr(ds, "features"):
        return {"(single)": ds}
    # Fallback: DatasetDict has .items but also lacks top-level .features
    try:
        return dict(ds.items())
    except Exception:  # noqa: BLE001
        return {"(single)": ds}


# ----------------------------------------------------------------------------- audit
def audit_split(name, ds, n_samples):
    header(f"SPLIT: {name}    rows = {len(ds):,}")

    feats = ds.features
    cols = list(feats.keys())

    # --- schema
    print("\nColumns / feature schema:")
    kinds = {}
    for c in cols:
        k = feature_kind(feats[c])
        kinds[c] = k
        print(f"  - {c:<24} {k}")

    image_cols = [c for c in cols if kinds[c] == "image"]
    text_cols = [c for c in cols if kinds[c] == "text"]
    label_cols = [c for c in cols if kinds[c] in ("label",) or "label" in c.lower()]

    # --- images
    if image_cols:
        print(f"\nImage columns: {image_cols}")
        try:
            row0 = ds[0]
            for c in image_cols:
                img = row0[c]
                size = getattr(img, "size", None)
                mode = getattr(img, "mode", None)
                if size:
                    print(f"    {c}: decoded PIL image, size={size}, mode={mode}")
                elif img is None:
                    print(f"    {c}: None in row 0 (field may be sometimes-empty)")
                else:
                    print(f"    {c}: {type(img).__name__} (not a decoded image -> may be a path/str)")
        except Exception as e:  # noqa: BLE001
            print(f"    (could not decode sample images: {e})")

        # how often is each image field present?
        print("\n  Image-field population (scan of up to 500 rows):")
        scan = min(len(ds), 500)
        for c in image_cols:
            present = 0
            for i in range(scan):
                try:
                    if ds[i][c] is not None:
                        present += 1
                except Exception:  # noqa: BLE001
                    pass
            print(f"    {c}: {present}/{scan} non-null")
    else:
        print("\nNo Image-type columns found. Reports/images may be stored as paths — "
              "inspect the columns above.")

    # --- labels
    header_labels(ds, cols, kinds, label_cols)

    # --- text columns + report format
    header(f"REPORT TEXT ANALYSIS  (split: {name})")
    if not text_cols:
        print("No string columns detected. If reports are a single blob column, "
              "check the schema above for the right field.")
        return

    print(f"Text (string) columns: {text_cols}\n")

    # XML vs parsed: sample each text column
    xml_hits = {}
    for c in text_cols:
        col_vals = ds[c][: min(len(ds), 200)]  # column access avoids decoding images
        hits = sum(1 for v in col_vals if looks_like_xml(v))
        xml_hits[c] = hits

    any_xml = any(v > 0 for v in xml_hits.values())
    if any_xml:
        print("XML detected in:")
        for c, h in xml_hits.items():
            if h:
                print(f"    {c}: {h}/200 sampled values look like XML  -> parser likely needed")
    else:
        print("No XML detected in sampled text columns -> reports appear PRE-PARSED. "
              "No XML parser needed; read the columns directly.")

    # length distribution per text column (chars + whitespace word count)
    print("\nPer-column length distribution (full split):")
    length_report = {}
    for c in text_cols:
        col_vals = ds[c]
        char_lens = sorted(len(v) if isinstance(v, str) else 0 for v in col_vals)
        word_lens = sorted(len(v.split()) if isinstance(v, str) else 0 for v in col_vals)
        empty = sum(1 for v in col_vals if not (isinstance(v, str) and v.strip()))
        length_report[c] = (word_lens, empty)
        print(f"\n  [{c}]  empty/missing: {empty}/{len(col_vals)}")
        print(f"    chars  p50={pctl(char_lens,50):7.0f}  p90={pctl(char_lens,90):7.0f}  "
              f"p95={pctl(char_lens,95):7.0f}  max={char_lens[-1]:7d}")
        print(f"    words  p50={pctl(word_lens,50):7.0f}  p90={pctl(word_lens,90):7.0f}  "
              f"p95={pctl(word_lens,95):7.0f}  max={word_lens[-1]:7d}")

    # --- Findings+Impression combined length -> ClinicalBERT max_length guidance
    fi_cols = [c for c in text_cols if c.lower() in
               ("findings", "impression", "finding", "impressions")]
    if len(fi_cols) >= 1:
        combined = []
        cols_vals = {c: ds[c] for c in fi_cols}  # column access: fast, no image decode
        n = len(ds)
        for i in range(n):
            txt = " ".join(str(cols_vals[c][i]) for c in fi_cols
                           if isinstance(cols_vals[c][i], str))
            combined.append(len(txt.split()))
        combined.sort()
        p95 = pctl(combined, 95)
        p99 = pctl(combined, 99)
        rec = 128 if p95 <= 110 else (256 if p95 <= 240 else 384)
        header("CLINICALBERT max_length RECOMMENDATION")
        print(f"Findings+Impression word count (fields used: {fi_cols}):")
        print(f"    p50={pctl(combined,50):.0f}  p95={p95:.0f}  p99={p99:.0f}  max={combined[-1]}")
        print(f"    -> Note: BERT tokens > words. A safe max_length ~ {rec} "
              f"(covers p95 with WordPiece expansion). Use dynamic padding + attention mask.")

    # --- id / uniqueness for grouped split
    header("STUDY ID / GROUPED-SPLIT KEY")
    id_candidates = [c for c in cols if c.lower() in
                     ("uid", "id", "study_id", "study", "report_id", "image_id", "name")]
    if id_candidates:
        for c in id_candidates:
            try:
                vals = ds[c]
                uniq = len(set(vals))
                print(f"  {c}: {uniq:,} unique / {len(vals):,} rows "
                      f"({'UNIQUE per row' if uniq == len(vals) else 'HAS DUPLICATES -> group on this'})")
            except Exception as e:  # noqa: BLE001
                print(f"  {c}: could not evaluate ({e})")
        print("\n  -> Use a unique per-study id as the grouping key for the "
              "train/val/test split (never let two views of one study cross splits).")
    else:
        print("  No obvious id column. You'll need to synthesise a stable study key "
              "(e.g. row index or a hash of the report) and document it.")

    # --- full sample dumps
    if n_samples > 0:
        header(f"FULL SAMPLE DUMP ({n_samples} rows)")
        for i in range(min(n_samples, len(ds))):
            print(f"\n----- row {i} " + "-" * 60)
            row = ds[i]
            for c in cols:
                if kinds[c] == "image":
                    img = row[c]
                    size = getattr(img, "size", None)
                    print(f"  {c}: <image size={size}>" if size else f"  {c}: {img!r}")
                else:
                    v = row[c]
                    if isinstance(v, str) and len(v) > 600:
                        v = v[:600] + f"... [+{len(v) - 600} chars]"
                    print(f"  {c}: {v!r}")


def header_labels(ds, cols, kinds, label_cols):
    header("LABELS CHECK")
    seq_cols = [c for c in cols if kinds[c].startswith("sequence")]
    numeric_cols = [c for c in cols if kinds[c].startswith("value:")
                    and kinds[c] != "value:string"]
    mesh_cols = [c for c in cols if "mesh" in c.lower() or "mtus" in c.lower()]

    found = False
    if label_cols:
        found = True
        print(f"  ClassLabel / label-named columns: {label_cols}")
    if mesh_cols:
        found = True
        print(f"  MeSH-style annotation columns: {mesh_cols}")
        for c in mesh_cols:
            try:
                sample = ds[c][:5]
                print(f"    e.g. {c}[:5] = {sample}")
            except Exception:  # noqa: BLE001
                pass
    if seq_cols:
        found = True
        print(f"  Sequence columns (possible multi-label vectors or tags): {seq_cols}")
    if numeric_cols:
        found = True
        print(f"  Numeric columns (possible label matrix): {numeric_cols}")

    if not found:
        print("  No shipped classification labels detected.")
        print("  -> Expected: IU X-ray on this mirror carries reports + (possibly) MeSH")
        print("     tags, NOT the 14 CheXpert observations. You must generate the 14")
        print("     weak labels yourself with CheXbert (Stage 1).")
    else:
        print("\n  NOTE: even if MeSH/tag columns exist, they are NOT the 14 CheXpert")
        print("  observations. Your task still needs CheXbert-derived weak labels.")


# ----------------------------------------------------------------------------- verdict
def verdict(splits):
    header("=  BOTTOM LINE  =")
    total = sum(len(d) for d in splits.values())
    print(f"Total rows across splits: {total:,}")
    print(f"Splits present: {list(splits.keys())}")
    if len(splits) == 1:
        print("  ⚠  Only one split — there is NO official train/val/test partition.")
        print("     You must make your own DETERMINISTIC GROUPED split (fixed seed,")
        print("     saved CSV) and must NOT call it the 'standard' split in the mémoire.")
    print("\nNext actions after reading the output above:")
    print("  1. Confirm reports are pre-parsed (columns) — skip the XML parser if so.")
    print("  2. Confirm frontal/lateral fields — decide frontal-only unit of analysis.")
    print("  3. Confirm no 14-class labels shipped — plan CheXbert weak-labeling.")
    print("  4. Use the max_length recommendation for ClinicalBERT.")
    print("  5. Then, and only then, spin up the A40 for Stage 2/3.")
    rule("=")


def main():
    ap = argparse.ArgumentParser(description="Audit the IU X-ray HF mirror (Stage 0).")
    ap.add_argument("--samples", type=int, default=2,
                    help="number of full sample rows to dump (default 2)")
    ap.add_argument("--local", type=str, default=None,
                    help="path to an already-downloaded dataset (load_from_disk)")
    args = ap.parse_args()

    ds = load(args)
    splits = as_splits(ds)

    for name, split_ds in splits.items():
        audit_split(name, split_ds, args.samples)

    verdict(splits)


if __name__ == "__main__":
    main()
