#!/usr/bin/env python3
"""
generator_compare.py  --  side-by-side: baseline vs a STRONGER generator.

Runs two pre-trained report generators on the same IU studies and prints
their outputs next to the real report, so you can see whether a bigger model
catches the pathology cases the baseline flattened into a normal template.

  baseline      : IAMJB/chexpert-mimic-cxr-findings-baseline   (~60M, ViT+BERT)
  chexpert_plus : StanfordAIMI/chexpert-plus-srrg_findings      (SwinV2 + 2-layer
                  BERT, trained on 223K CheXpert Plus reports - genuinely stronger)

Both load with the identical VisionEncoderDecoder API, so this is a drop-in
comparison, not a re-architecture. CPU-friendly on Apple Silicon.

This produces a MEMOIRE FIGURE, not a pipeline change. Remember the ceiling:
a better report is still G(image), so E4~=E1 is unaffected. The value here is
evidence for your 'stronger generator = future work' claim.

Default targets are the cases your baseline pilot got wrong (+ controls):
  3930 COPD | 517 granuloma | 2696 cardiomegaly | 3550 tortuous aorta
  2102, 1317 (partial hits) | 2698 (normal control)

Usage:
    python3 generator_compare.py
    python3 generator_compare.py --uids 3930 517 2696 3550
"""

import argparse
import io
import sys


MODELS = {
    "baseline": "IAMJB/chexpert-mimic-cxr-findings-baseline",
    "chexpert_plus": "StanfordAIMI/chexpert-plus-srrg_findings",
}

PATHO_KW = {
    "cardiomegaly", "effusion", "effusions", "consolidation", "edema",
    "opacity", "opacities", "pneumothorax", "atelectasis", "infiltrate",
    "infiltrates", "nodule", "nodules", "mass", "masses", "fracture",
    "emphysema", "fibrosis", "pneumonia", "congestion", "thickening",
    "enlarged", "hyperinflation", "hyperexpand", "hyperexpanded", "granuloma",
    "calcified", "calcification", "scarring", "tortuous", "tortuosity",
    "ectasia", "eventration", "hernia", "degenerative", "prominence",
}


def words(s):
    return "".join(c.lower() if (c.isalnum() or c.isspace()) else " "
                   for c in str(s)).split()


def patho_hits(text):
    return sum(1 for w in set(words(text)) if w in PATHO_KW)


def trunc(s, n=300):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n}]"


def load_one(model_id, device):
    import torch
    from transformers import (VisionEncoderDecoderModel, ViTImageProcessor,
                              BertTokenizer, GenerationConfig)
    print(f"  loading {model_id} ...")
    model = VisionEncoderDecoderModel.from_pretrained(model_id).eval().to(device)
    tok = BertTokenizer.from_pretrained(model_id)
    proc = ViTImageProcessor.from_pretrained(model_id)

    c = model.config
    # On VisionEncoderDecoderConfig, bos/eos live on the decoder sub-config,
    # not the top level - pull them defensively, fall back to the tokenizer.
    dec = getattr(c, "decoder", c)

    def pick(attr, fallback):
        v = getattr(c, attr, None)
        if v is None:
            v = getattr(dec, attr, None)
        return v if v is not None else fallback

    gen_cfg = GenerationConfig(
        bos_token_id=pick("bos_token_id", tok.cls_token_id),
        eos_token_id=pick("eos_token_id", tok.sep_token_id),
        pad_token_id=pick("pad_token_id", tok.pad_token_id),
        decoder_start_token_id=pick("decoder_start_token_id", tok.cls_token_id),
        num_beams=2, max_length=128, use_cache=True,
    )
    return {"model": model, "tok": tok, "proc": proc, "cfg": gen_cfg, "torch": torch}


def gen(bundle, img):
    m, proc, tok, cfg, torch = (bundle["model"], bundle["proc"], bundle["tok"],
                                bundle["cfg"], bundle["torch"])
    pv = proc(img, return_tensors="pt").pixel_values.to(m.device)
    with torch.no_grad():
        ids = m.generate(pv, generation_config=cfg)
    return tok.decode(ids[0], skip_special_tokens=True).strip()


def decode_blob(blob):
    from PIL import Image
    if not blob:
        return None
    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", type=int, nargs="+",
                    default=[3930, 517, 2696, 3550, 2102, 1317, 2698])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: pip install 'datasets>=2.0' torch transformers pillow")

    print("Loading dataset (cached)...")
    ds = load_dataset("ykumards/open-i")["train"]
    uid_to_idx = {u: i for i, u in enumerate(ds["uid"])}

    print("Loading models (chexpert_plus downloads on first run):")
    bundles = {name: load_one(mid, args.device) for name, mid in MODELS.items()}

    print("\n" + "=" * 78)
    print("BASELINE  vs  CHEXPERT-PLUS  vs  REAL REPORT")
    print("=" * 78)

    tally = {name: 0 for name in MODELS}
    seen = 0
    for uid in args.uids:
        if uid not in uid_to_idx:
            print(f"\n[uid {uid}] not in dataset - skipped")
            continue
        row = ds[uid_to_idx[uid]]
        img = decode_blob(row["img_frontal"])
        if img is None:
            print(f"\n[uid {uid}] no frontal image - skipped")
            continue
        seen += 1

        real = " ".join(p for p in (row.get("findings") or "",
                                    row.get("impression") or "") if p.strip())
        print(f"\n----- uid {uid} " + "-" * 60)
        print(f"  MeSH (manual)  : {trunc(row.get('MeSH'), 140)}")
        print(f"  REAL report    : {trunc(real)}")
        for name in MODELS:
            try:
                txt = gen(bundles[name], img)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<14}: GENERATION FAILED: {e}")
                continue
            h = patho_hits(txt)
            tally[name] += h
            print(f"  {name:<14}: {trunc(txt)}")
            print(f"  {'':<14}    (finding-words: {h})")

    print("\n" + "=" * 78)
    print("ROUGH TALLY (negation-naive - read the text, this is only a nudge)")
    print("=" * 78)
    for name in MODELS:
        print(f"  {name:<14}: total finding-words across {seen} studies = {tally[name]}")
    print("\nWhat to look for: on the pathology uids, does chexpert_plus NAME the")
    print("condition the baseline missed (COPD/hyperexpansion, granuloma/calcified")
    print("nodule, cardiomegaly, aortic tortuosity)? If yes -> that's your figure:")
    print("'a stronger generator recovers cases the baseline collapsed', which")
    print("justifies the future-work direction WITHOUT changing your 5-day plan.")
    print("On uid 2698 (normal), a good model should still read normal - check it")
    print("doesn't hallucinate disease.")
    print("=" * 78)


if __name__ == "__main__":
    main()
