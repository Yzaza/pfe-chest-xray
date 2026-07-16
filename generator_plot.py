#!/usr/bin/env python3
"""
generator_pilot.py  --  Stage 0 feasibility GATE for the PFE.

Runs the pre-trained report generator (Stage 1 of your pipeline) on a
deliberately chosen mix of IU X-ray studies and answers the single question
that decides whether the whole project is viable as designed:

    Does IAMJB/chexpert-mimic-cxr-findings-baseline, trained on MIMIC/CheXpert,
    still produce SENSIBLE, INPUT-DEPENDENT findings on IU/OpenI images?

The failure mode that kills the project is NOT "bad grammar" — it is
"the model emits the same generic 'lungs are clear' text no matter what
the image shows." A degenerate generator that ignores its input adds zero
signal to Stage 2, and you need to know that now, locally, for $0 — not on
a running A40.

To catch that, the pilot samples two groups by the manual MeSH annotation:
    - NORMAL  studies (MeSH == 'normal')      -> should read clear/normal
    - PATHOLOGY studies (MeSH names disease)  -> should read DIFFERENTLY

and reports whether the generator actually separates them.

Runs on CPU (Apple Silicon fine). ~12 images, a couple of minutes.
Reuses the datasets cache from the audit run (no 2 GB re-download).

Usage:
    # in your existing venv:
    pip install torch transformers          # torch is a chunky download
    python3 generator_pilot.py              # 6 normal + 6 pathology
    python3 generator_pilot.py --n-per-group 8 --seed 1
    python3 generator_pilot.py --device mps # try Apple GPU (CPU is the safe default)
"""

import argparse
import io
import random
import sys
from itertools import combinations


MODEL_ID = "IAMJB/chexpert-mimic-cxr-findings-baseline"

# CheXpert-flavoured finding words: their presence in generated text signals
# the model is describing an abnormality rather than boilerplate normality.
PATHO_KW = {
    "cardiomegaly", "effusion", "effusions", "consolidation", "edema",
    "opacity", "opacities", "pneumothorax", "atelectasis", "infiltrate",
    "infiltrates", "nodule", "nodules", "mass", "masses", "fracture",
    "emphysema", "fibrosis", "pneumonia", "congestion", "thickening",
    "enlarged", "hyperinflation", "calcification", "granuloma", "scarring",
    "hernia", "tortuous", "prominence", "degenerative", "abnormal",
}
NORMAL_MARKERS = ("normal", "clear", "unremarkable", "no acute",
                  "within normal limits", "no evidence", "no focal")


# ----------------------------------------------------------------------------- utils
def rule(c="-", n=78):
    print(c * n)


def head(t):
    print()
    rule("=")
    print(t)
    rule("=")


def words(s):
    return [w for w in "".join(
        ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in s
    ).split()]


def word_jaccard(a, b):
    sa, sb = set(words(a)), set(words(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def patho_hits(text):
    return sum(1 for w in set(words(text)) if w in PATHO_KW)


def has_normal_marker(text):
    t = text.lower()
    return any(m in t for m in NORMAL_MARKERS)


def truncate(s, n=280):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + f" …[+{len(s) - n}]"


# ----------------------------------------------------------------------------- load
def load_generator(device):
    try:
        import torch
        from transformers import (VisionEncoderDecoderModel,
                               ViTImageProcessor, BertTokenizer)
    except ImportError:
        sys.exit("ERROR: need torch + transformers. Run: pip install torch transformers")

    print(f"Loading generator '{MODEL_ID}' (first run downloads ~260 MB)…")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)

    # Defensive: ensure the decoder has the tokens generate() needs.
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.sep_token_id

    model.to(device)
    model.eval()
    print(f"  loaded. device={device}, "
          f"tokenizer={type(tokenizer).__name__}, "
          f"image_size={getattr(processor, 'size', '?')}")
    return model, processor, tokenizer, torch


def decode_blob(blob):
    from PIL import Image
    if blob is None or (isinstance(blob, (bytes, bytearray)) and len(blob) == 0):
        return None
    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")  # 3-ch for ViT
    except Exception:  # noqa: BLE001
        return None


def generate(model, processor, tokenizer, torch, img, max_length, num_beams):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(model.device)
    with torch.no_grad():
        ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------- data
def pick_studies(ds, n_per_group, seed):
    mesh = ds["MeSH"]
    frontal = ds["img_frontal"]

    def usable(i):
        return frontal[i] is not None and len(frontal[i]) > 0

    normal_idx = [i for i, m in enumerate(mesh)
                  if isinstance(m, str) and m.strip().lower() == "normal" and usable(i)]
    patho_idx = [i for i, m in enumerate(mesh)
                 if isinstance(m, str) and m.strip().lower() not in ("normal", "") and usable(i)]

    rng = random.Random(seed)
    rng.shuffle(normal_idx)
    rng.shuffle(patho_idx)
    chosen = ([("NORMAL", i) for i in normal_idx[:n_per_group]] +
              [("PATHOLOGY", i) for i in patho_idx[:n_per_group]])
    rng.shuffle(chosen)  # interleave so you can't game the read
    return chosen


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generator feasibility gate (Stage 0).")
    ap.add_argument("--n-per-group", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: need datasets. Run: pip install 'datasets>=2.0'")

    print("Loading ykumards/open-i (cached from the audit run)…")
    ds = load_dataset("ykumards/open-i")["train"]

    model, processor, tokenizer, torch = load_generator(args.device)
    studies = pick_studies(ds, args.n_per_group, args.seed)
    if not studies:
        sys.exit("No usable studies with a frontal image were found. Check img_frontal.")

    head("SIDE-BY-SIDE: generated findings vs real report")
    results = []  # (group, uid, generated_text, real_text)
    for group, i in studies:
        row = ds[i]
        img = decode_blob(row["img_frontal"])
        if img is None:
            print(f"\n[uid {row['uid']}] frontal image failed to decode — skipped")
            continue

        real_parts = [row.get("findings") or "", row.get("impression") or ""]
        real = " ".join(p for p in real_parts if isinstance(p, str) and p.strip())

        try:
            gen = generate(model, processor, tokenizer, torch, img,
                           args.max_length, args.num_beams)
        except Exception as e:  # noqa: BLE001
            print(f"\n[uid {row['uid']}] generation FAILED: {e}")
            continue

        results.append((group, row["uid"], gen, real))
        khits = patho_hits(gen)
        flag = "normal-ish" if has_normal_marker(gen) and khits == 0 else \
               f"flags {khits} finding word(s)"
        print(f"\n----- uid {row['uid']}  [{group}]  ({flag}) " + "-" * 24)
        print(f"  MeSH (manual): {truncate(row.get('MeSH'), 120)}")
        print(f"  GENERATED    : {truncate(gen)}")
        print(f"  REAL report  : {truncate(real)}")

    if len(results) < 2:
        sys.exit("\nToo few successful generations to judge. Investigate errors above.")

    # ---- quantitative diagnostics ---------------------------------------------
    head("DIAGNOSTICS")

    gens = [g for (_, _, g, _) in results]
    n = len(gens)

    # 1) Corpus diversity: are the outputs actually different from each other?
    uniq = len(set(" ".join(words(g)) for g in gens))
    pair_sims = [word_jaccard(a, b) for a, b in combinations(gens, 2)]
    avg_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0
    print(f"Unique generated reports : {uniq}/{n}")
    print(f"Avg pairwise word-Jaccard: {avg_sim:.3f}   "
          f"(near 1.0 = boilerplate/degenerate; lower = input-dependent)")

    # 2) THE money metric: does the generator separate pathology from normal?
    norm_hits = [patho_hits(g) for (grp, _, g, _) in results if grp == "NORMAL"]
    path_hits = [patho_hits(g) for (grp, _, g, _) in results if grp == "PATHOLOGY"]
    mean_norm = sum(norm_hits) / len(norm_hits) if norm_hits else 0.0
    mean_path = sum(path_hits) / len(path_hits) if path_hits else 0.0
    print(f"\nMean finding-words in generated text:")
    print(f"    NORMAL   group: {mean_norm:.2f}   (n={len(norm_hits)})")
    print(f"    PATHOLOGY group: {mean_path:.2f}   (n={len(path_hits)})")
    sep = mean_path - mean_norm
    print(f"    separation (Δ) : {sep:+.2f}   "
          f"(positive & meaningful = generator tracks pathology)")

    # ---- advisory verdict ------------------------------------------------------
    head("VERDICT (advisory — the side-by-side text is the real evidence)")
    if avg_sim >= 0.80:
        print("🔴  DEGENERATE: outputs are near-identical regardless of input.")
        print("    The generator is emitting boilerplate. Stage 2's text branch")
        print("    would carry ~no signal. PIVOT before spending on the pod:")
        print("      • try a different generator (e.g. a CXR-report model trained")
        print("        with IU in-distribution, or one of the newer RRG baselines), or")
        print("      • reframe the thesis around real reports only (E1 vs oracle),")
        print("        dropping the self-generated arm.")
    elif sep <= 0.15:
        print("🟠  WEAK SEPARATION: text varies, but pathology cases don't read")
        print("    meaningfully differently from normals. Domain shift is degrading")
        print("    clinical content. The pipeline will run, but expect the")
        print("    self-generated arm (E4) to add little — plan the mémoire so a")
        print("    null result is a finding, and lean on the shuffled-report control.")
        print("    Consider testing one alternative generator before committing.")
    else:
        print("🟢  VIABLE: outputs are input-dependent AND pathology cases carry more")
        print("    finding-words than normals. The generator survives IU well enough")
        print("    to give the text branch real signal. Green-light the A40 for")
        print("    Stage 2 (full cached generation) and Stage 3 (training).")
    print("\nAlso eyeball the side-by-side above for hallucinations (findings named")
    print("that the real report and image don't support) — a few are expected;")
    print("pervasive hallucination is its own caveat for the Discussion.")
    rule("=")


if __name__ == "__main__":
    main()
