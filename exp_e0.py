#!/usr/bin/env python3
"""E0: label agreement between generated and real reports (CheXbert both sides)."""
import csv, json
import numpy as np
CHEXPERT_14 = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
    "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"]
def main():
    import torch
    from f1chexbert import F1CheXbert
    from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = {}
    with open("iu_labels.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"]=="test": real[int(r["uid"])]=np.array([int(r[c]) for c in CHEXPERT_14])
    gen_text = {}
    with open("iu_generated.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            uid=int(r["uid"])
            if uid in real: gen_text[uid]=r["generated_report"] or ""
    uids=[u for u in real if u in gen_text]
    print(f"Comparing {len(uids)} test studies")
    print("Loading CheXbert...")
    cb=F1CheXbert(device=device)
    b=lambda codes:[1 if c==1 else 0 for c in codes]
    gen_lab={}
    for i,u in enumerate(uids):
        try: gen_lab[u]=b(cb.get_label(gen_text[u]))
        except Exception: gen_lab[u]=[0]*14
        if (i+1)%100==0: print(f"  {i+1}/{len(uids)}")
    Yr=np.stack([real[u] for u in uids]); Yg=np.stack([gen_lab[u] for u in uids])
    print("\n"+"="*70)
    print("E0: label agreement  generated vs real report (CheXbert both sides)")
    print("="*70)
    print(f"  {'class':<28}{'F1':>7}{'prec':>7}{'rec':>7}{'kappa':>8}{'real+':>7}")
    f1s=[]
    for j,name in enumerate(CHEXPERT_14):
        yt,yp=Yr[:,j],Yg[:,j]
        if yt.sum()==0 and yp.sum()==0:
            print(f"  {name:<28}{'--':>7}{'--':>7}{'--':>7}{'--':>8}{int(yt.sum()):>7}"); continue
        f1=f1_score(yt,yp,zero_division=0);pr=precision_score(yt,yp,zero_division=0)
        rc=recall_score(yt,yp,zero_division=0)
        try: kp=cohen_kappa_score(yt,yp)
        except Exception: kp=float("nan")
        f1s.append(f1)
        print(f"  {name:<28}{f1:>7.3f}{pr:>7.3f}{rc:>7.3f}{kp:>8.3f}{int(yt.sum()):>7}")
    mf1=float(np.mean(f1s)) if f1s else 0.0
    print("-"*70); print(f"  macro-F1 = {mf1:.3f}"); print("="*70)
    json.dump({"experiment":"E0","macro_f1":mf1,"n":len(uids)}, open("results_e0.json","w"), indent=2)
    print("Saved results_e0.json")
if __name__=="__main__": main()
