#!/usr/bin/env python3
"""
cache_reports.py  --  Stage 2b: pre-generate + cache synthetic reports.

Runs the FROZEN baseline generator over every frontal image in iu_split.csv
and writes iu_generated.csv: uid, split, generated_report. Generated once,
then frozen — training never re-runs the generator.
"""

import argparse
import csv
import io
import sys

MODEL_ID = "IAMJB/chexpert-mimic-cxr-findings-baseline"


def decode_blob(blob):
    from PIL import Image
    if not blob:
        return None
    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="iu_split.csv")
    ap.add_argument("--out", default="iu_generated.csv")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=2)
    args = ap.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from transformers import (VisionEncoderDecoderModel, ViTImageProcessor,
                                  BertTokenizer, GenerationConfig)
    except ImportError as e:
        sys.exit(f"ERROR: missing dep ({e})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    want = {}
    with open(args.split_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            want[int(r["uid"])] = r["split"]
    print(f"Need reports for {len(want):,} studies")

    print("Loading dataset (cached)...")
    ds = load_dataset("ykumards/open-i")["train"]
    uid_to_idx = {u: i for i, u in enumerate(ds["uid"])}

    print(f"Loading generator {MODEL_ID} ...")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).eval().to(device)
    proc = ViTImageProcessor.from_pretrained(MODEL_ID)
    tok = BertTokenizer.from_pretrained(MODEL_ID)

    c = model.config
    dec = getattr(c, "decoder", c)

    def pick(attr, fb):
        v = getattr(c, attr, None) or getattr(dec, attr, None)
        return v if v is not None else fb

    gen_cfg = GenerationConfig(
        bos_token_id=pick("bos_token_id", tok.cls_token_id),
        eos_token_id=pick("eos_token_id", tok.sep_token_id),
        pad_token_id=pick("pad_token_id", tok.pad_token_id),
        decoder_start_token_id=pick("decoder_start_token_id", tok.cls_token_id),
        num_beams=args.num_beams, max_length=args.max_length, use_cache=True,
    )

    uids = sorted(want.keys())
    results = {}
    no_image = 0
    batch_imgs, batch_uids = [], []

    def flush():
        if not batch_imgs:
            return
        pv = proc(batch_imgs, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, generation_config=gen_cfg)
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        for u, t in zip(batch_uids, texts):
            results[u] = t.strip()
        batch_imgs.clear()
        batch_uids.clear()

    done = 0
    for uid in uids:
        idx = uid_to_idx.get(uid)
        img = decode_blob(ds[idx]["img_frontal"]) if idx is not None else None
        if img is None:
            results[uid] = ""
            no_image += 1
            done += 1
            continue
        batch_imgs.append(img)
        batch_uids.append(uid)
        if len(batch_imgs) >= args.batch:
            flush()
            done += args.batch
            print(f"  generated {min(done, len(uids)):,}/{len(uids):,}")
    flush()
    print(f"  generated {len(uids):,}/{len(uids):,} (done)")

    empty = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "split", "generated_report"])
        for uid in uids:
            txt = results.get(uid, "")
            if not txt:
                empty += 1
            w.writerow([uid, want[uid], txt])

    print(f"\nWrote {len(uids):,} rows -> {args.out}")
    print(f"  no frontal image (empty report): {no_image}")
    print(f"  total empty reports: {empty}")
    nonempty = [t for t in results.values() if t]
    uniq = len(set(nonempty))
    if nonempty:
        print(f"  unique reports: {uniq}/{len(nonempty)} "
              f"({100.0*uniq/len(nonempty):.1f}% distinct)")


if __name__ == "__main__":
    main()
