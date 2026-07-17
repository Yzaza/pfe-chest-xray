#!/usr/bin/env python3
"""Cache cxrmate-single-tf reports -> iu_generated_cxrmate.csv (same format)."""
import csv, io
MODEL_ID = "aehrc/cxrmate-single-tf"
def main():
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    want = {}
    with open("iu_split.csv", newline="") as fh:
        for r in csv.DictReader(fh): want[int(r["uid"])] = r["split"]
    print(f"Need reports for {len(want):,} studies")
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
    gen_kwargs = dict(max_length=128, num_beams=2, decoder_start_token_id=bos,
                      bos_token_id=bos, eos_token_id=eos, pad_token_id=pad,
                      special_token_ids=[sep, eos])
    uids = sorted(want.keys()); results, no_image = {}, 0
    for n, uid in enumerate(uids):
        idx = uid_to_idx.get(uid)
        blob = ds[idx]["img_frontal"] if idx is not None else None
        if not blob:
            results[uid] = ""; no_image += 1
        else:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
            try:
                with torch.no_grad(): out = model.generate(pixel_values=pv, **gen_kwargs)
                results[uid] = tok.decode(out[0], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"  [uid {uid}] gen failed: {e}"); results[uid] = ""
        if (n + 1) % 100 == 0: print(f"  {n+1:,}/{len(uids):,}")
    print(f"  {len(uids):,}/{len(uids):,} done")
    empty = 0
    with open("iu_generated_cxrmate.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["uid","split","generated_report"])
        for uid in uids:
            t = results.get(uid, "")
            if not t: empty += 1
            w.writerow([uid, want[uid], t])
    print(f"\nWrote iu_generated_cxrmate.csv ({len(uids):,} rows)")
    print(f"  no frontal: {no_image}   empty: {empty}")
    ne = [t for t in results.values() if t]
    if ne:
        u = len(set(ne)); print(f"  unique: {u}/{len(ne)} ({100.0*u/len(ne):.1f}% distinct) vs baseline 22.9%")
if __name__ == "__main__": main()
