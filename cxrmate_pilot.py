#!/usr/bin/env python3
"""cxrmate-single-tf pilot -- passes special_token_ids for its section-based decode."""
import io

MODEL_ID = "aehrc/cxrmate-single-tf"
UIDS = [3930, 517, 2696, 3550, 2102, 1317, 2698]

def trunc(s, n=300):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + f" ...[+{len(s)-n}]"

def main():
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading dataset (cached)...")
    ds = load_dataset("ykumards/open-i")["train"]
    uid_to_idx = {u: i for i, u in enumerate(ds["uid"])}
    print(f"Loading {MODEL_ID} ...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval().to(device)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    sep = tok.sep_token_id if tok.sep_token_id is not None else eos
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    # section-ending tokens: findings ends with sep, impression ends with eos
    special_token_ids = [sep, eos]
    print(f"tokens: bos={bos} sep={sep} eos={eos} pad={pad}  special={special_token_ids}")
    gen_kwargs = dict(max_length=128, num_beams=2, decoder_start_token_id=bos,
                      bos_token_id=bos, eos_token_id=eos, pad_token_id=pad,
                      special_token_ids=special_token_ids)

    print("\n" + "=" * 78)
    print("CXRMATE-single-tf  vs  REAL REPORT")
    print("=" * 78)
    for uid in UIDS:
        if uid not in uid_to_idx:
            print(f"\n[uid {uid}] not found"); continue
        row = ds[uid_to_idx[uid]]
        blob = row["img_frontal"]
        if not blob:
            print(f"\n[uid {uid}] no frontal"); continue
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
        try:
            with torch.no_grad():
                out = model.generate(pixel_values=pv, **gen_kwargs)
        except Exception as e:
            print(f"\n[uid {uid}] generate failed: {type(e).__name__}: {e}"); continue
        # try the model's own section decoder if present, else plain decode
        try:
            secs = model.split_and_decode_sections(out, special_token_ids, tok)
            gen = " ".join(s.strip() for s in (secs if isinstance(secs, (list, tuple)) else [secs]))
        except Exception:
            gen = tok.decode(out[0], skip_special_tokens=True).strip()
        real = " ".join(p for p in (row.get("findings") or "", row.get("impression") or "") if p.strip())
        print(f"\n----- uid {uid} " + "-" * 58)
        print(f"  MeSH   : {trunc(row.get('MeSH'), 120)}")
        print(f"  CXRMATE: {trunc(gen)}")
        print(f"  REAL   : {trunc(real)}")
    print("\n" + "=" * 78)
    print("Does cxrmate catch COPD/3930, granuloma/517, cardiomegaly/2696, aorta/3550,")
    print("or collapse to normal templates like the MIMIC-trained models did?")
    print("=" * 78)

if __name__ == "__main__":
    main()
