#!/usr/bin/env python3
"""E1: image-only baseline (ViT-B/16 -> 14 classes). Modes: frozen | finetune."""
import argparse, csv, io, json
import numpy as np

CHEXPERT_14 = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
    "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"]
MIN_TEST_POS = 10
MODEL_ID = "IAMJB/chexpert-mimic-cxr-findings-baseline"

def load_joined(split_csv, labels_csv):
    splits = {}
    with open(split_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            splits[int(r["uid"])] = (r["split"], int(r.get("has_frontal","1")))
    joined = {}
    with open(labels_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            uid = int(r["uid"])
            if uid not in splits: continue
            split, has_frontal = splits[uid]
            if not has_frontal: continue
            vec = np.array([int(r[c]) for c in CHEXPERT_14], dtype=np.float32)
            joined[uid] = {"split": split, "labels": vec}
    return joined

def build_dataset_index():
    from datasets import load_dataset
    ds = load_dataset("ykumards/open-i")["train"]
    return ds, {u: i for i, u in enumerate(ds["uid"])}

class XrayDS:
    def __init__(self, uids, joined, ds, uid_to_idx, proc):
        self.uids=uids; self.joined=joined; self.ds=ds; self.uid_to_idx=uid_to_idx; self.proc=proc
    def __len__(self): return len(self.uids)
    def __getitem__(self, i):
        import torch
        from PIL import Image
        uid = self.uids[i]
        blob = self.ds[self.uid_to_idx[uid]]["img_frontal"]
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        pv = self.proc(img, return_tensors="pt").pixel_values[0]
        return pv, torch.from_numpy(self.joined[uid]["labels"])

def build_model(mode, unfreeze):
    import torch.nn as nn
    from transformers import VisionEncoderDecoderModel
    vit = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).encoder
    for p in vit.parameters(): p.requires_grad = False
    if mode == "finetune":
        for blk in vit.encoder.layer[-unfreeze:]:
            for p in blk.parameters(): p.requires_grad = True
    hidden = vit.config.hidden_size
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__(); self.vit=vit; self.head=nn.Linear(hidden,14)
        def forward(self, pv):
            return self.head(self.vit(pixel_values=pv).last_hidden_state[:,0])
    return Classifier()

def macro_metrics(y_true, y_prob, supported_idx):
    from sklearn.metrics import roc_auc_score, average_precision_score
    aurocs, auprcs = {}, {}
    for j in range(14):
        yt = y_true[:, j]
        if yt.sum()==0 or yt.sum()==len(yt):
            aurocs[j]=float("nan"); auprcs[j]=float("nan"); continue
        aurocs[j]=roc_auc_score(yt, y_prob[:,j]); auprcs[j]=average_precision_score(yt, y_prob[:,j])
    return (np.nanmean([aurocs[j] for j in supported_idx]),
            np.nanmean([auprcs[j] for j in supported_idx]), aurocs, auprcs)

def bootstrap_ci(y_true, y_prob, supported_idx, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); n=len(y_true); A,P=[],[]
    for _ in range(n_boot):
        idx = rng.integers(0,n,n)
        a,p,_,_ = macro_metrics(y_true[idx], y_prob[idx], supported_idx)
        if not np.isnan(a): A.append(a)
        if not np.isnan(p): P.append(p)
    ci = lambda v:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if v else (float("nan"),)*2
    return ci(A), ci(P)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen","finetune"], required=True)
    ap.add_argument("--split_csv", default="iu_split.csv")
    ap.add_argument("--labels_csv", default="iu_labels.csv")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--unfreeze", type=int, default=3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import ViTImageProcessor
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epochs or (15 if args.mode=="frozen" else 20)
    lr = args.lr or (1e-3 if args.mode=="frozen" else 2e-5)
    print(f"Mode={args.mode} epochs={epochs} lr={lr} device={device}")
    joined = load_joined(args.split_csv, args.labels_csv)
    ds, uid_to_idx = build_dataset_index()
    proc = ViTImageProcessor.from_pretrained(MODEL_ID)
    by = {"train":[], "val":[], "test":[]}
    for uid, info in joined.items():
        if info["split"] in by: by[info["split"]].append(uid)
    print(f"train={len(by['train'])} val={len(by['val'])} test={len(by['test'])}")
    test_lab = np.stack([joined[u]["labels"] for u in by["test"]])
    test_pos = test_lab.sum(0)
    supported_idx = [j for j in range(14) if test_pos[j] >= MIN_TEST_POS]
    print("Supported:", [CHEXPERT_14[j] for j in supported_idx])
    def loader(s, sh):
        return DataLoader(XrayDS(by[s], joined, ds, uid_to_idx, proc),
                          batch_size=args.batch, shuffle=sh, num_workers=4, pin_memory=True)
    tr, va, te = loader("train",True), loader("val",False), loader("test",False)
    model = build_model(args.mode, args.unfreeze).to(device)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    def evaluate(dl):
        model.eval(); ys,ps=[],[]
        with torch.no_grad():
            for pv,y in dl:
                pv=pv.to(device)
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    logits=model(pv)
                ps.append(torch.sigmoid(logits).float().cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(ys), np.concatenate(ps)
    best_auroc, best_state, bad = -1, None, 0
    for ep in range(1, epochs+1):
        model.train(); tot=0.0
        for pv,y in tr:
            pv,y = pv.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                loss = lossf(model(pv), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()*pv.size(0)
        yv,pvp = evaluate(va)
        a,p,_,_ = macro_metrics(yv,pvp,supported_idx)
        print(f"epoch {ep:2d} train_loss={tot/len(by['train']):.4f} val_AUROC={a:.4f} val_AUPRC={p:.4f}")
        if a > best_auroc:
            best_auroc=a; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=args.patience: print("early stop"); break
    if best_state: model.load_state_dict(best_state)
    yt,pt = evaluate(te)
    ma,mp,aurocs,auprcs = macro_metrics(yt,pt,supported_idx)
    (alo,ahi),(plo,phi) = bootstrap_ci(yt,pt,supported_idx,seed=args.seed)
    print("\n"+"="*68)
    print(f"E1 [{args.mode}] TEST  (macro over {len(supported_idx)} classes)")
    print("="*68)
    print(f"  macro AUROC = {ma:.4f}  (95% CI {alo:.4f}-{ahi:.4f})")
    print(f"  macro AUPRC = {mp:.4f}  (95% CI {plo:.4f}-{phi:.4f})")
    print("-"*68)
    print(f"  {'class':<28}{'AUROC':>8}{'AUPRC':>8}{'test+':>8}")
    for j,name in enumerate(CHEXPERT_14):
        mark = "" if j in supported_idx else "  (sparse)"
        print(f"  {name:<28}{aurocs[j]:>8.3f}{auprcs[j]:>8.3f}{int(test_pos[j]):>8}{mark}")
    print("="*68)
    np.savez(f"preds_E1_{args.mode}.npz", y_true=yt, y_prob=pt,
             supported_idx=np.array(supported_idx), uids=np.array(by["test"]))
    with open(f"results_E1_{args.mode}.json","w") as fh:
        json.dump({"experiment":"E1","mode":args.mode,"macro_auroc":ma,"macro_auprc":mp,
            "auroc_ci":[alo,ahi],"auprc_ci":[plo,phi],
            "supported_classes":[CHEXPERT_14[j] for j in supported_idx],
            "per_class_auroc":{CHEXPERT_14[j]:(None if np.isnan(aurocs[j]) else aurocs[j]) for j in range(14)},
            "per_class_auprc":{CHEXPERT_14[j]:(None if np.isnan(auprcs[j]) else auprcs[j]) for j in range(14)}}, fh, indent=2)
    print(f"Saved preds_E1_{args.mode}.npz + results_E1_{args.mode}.json")

if __name__ == "__main__":
    main()
