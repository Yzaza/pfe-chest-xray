#!/usr/bin/env python3
"""Fusion experiments: image + generated report via cross-attention.
Modes: e4 (proposed) | e5 (shuffled control) | e6 (real-report oracle) | concat (E3).
Frozen encoders (E1 ablation showed fine-tuning hurts)."""
import argparse, csv, io, json
import numpy as np

CHEXPERT_14 = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
    "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"]
MIN_TEST_POS = 10
IMG_MODEL = "IAMJB/chexpert-mimic-cxr-findings-baseline"
TXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

def load_all(split_csv, labels_csv, gen_csv):
    splits, labels, gens = {}, {}, {}
    with open(split_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            splits[int(r["uid"])] = (r["split"], int(r.get("has_frontal","1")), r.get("report_text",""))
    with open(labels_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels[int(r["uid"])] = np.array([int(r[c]) for c in CHEXPERT_14], dtype=np.float32)
    with open(gen_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gens[int(r["uid"])] = r.get("generated_report","")
    joined = {}
    for uid, (split, hf, real_text) in splits.items():
        if uid not in labels or not hf: continue
        joined[uid] = {"split":split, "labels":labels[uid], "gen":gens.get(uid,""), "real":real_text}
    return joined

def build_dataset_index():
    from datasets import load_dataset
    ds = load_dataset("ykumards/open-i")["train"]
    return ds, {u:i for i,u in enumerate(ds["uid"])}

class FusionDS:
    def __init__(self, uids, joined, ds, uid_to_idx, img_proc, tokz, text_key, maxlen):
        self.uids=uids; self.joined=joined; self.ds=ds; self.uid_to_idx=uid_to_idx
        self.img_proc=img_proc; self.tokz=tokz; self.text_key=text_key; self.maxlen=maxlen
    def __len__(self): return len(self.uids)
    def __getitem__(self, i):
        import torch
        from PIL import Image
        uid = self.uids[i]
        blob = self.ds[self.uid_to_idx[uid]]["img_frontal"]
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        pv = self.img_proc(img, return_tensors="pt").pixel_values[0]
        txt = self.joined[uid][self.text_key] or ""
        enc = self.tokz(txt, truncation=True, max_length=self.maxlen, padding="max_length", return_tensors="pt")
        return pv, enc["input_ids"][0], enc["attention_mask"][0], torch.from_numpy(self.joined[uid]["labels"])

def build_model(mode, device):
    import torch, torch.nn as nn
    from transformers import VisionEncoderDecoderModel, AutoModel
    img_enc = VisionEncoderDecoderModel.from_pretrained(IMG_MODEL).encoder
    txt_enc = AutoModel.from_pretrained(TXT_MODEL)
    for p in img_enc.parameters(): p.requires_grad = False
    for p in txt_enc.parameters(): p.requires_grad = False
    img_dim = img_enc.config.hidden_size; txt_dim = txt_enc.config.hidden_size; D = 512
    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_enc=img_enc; self.txt_enc=txt_enc
            self.img_proj=nn.Linear(img_dim,D); self.txt_proj=nn.Linear(txt_dim,D)
            self.use_xattn = (mode in ("e4","e5","e6"))
            if self.use_xattn:
                self.i2t=nn.MultiheadAttention(D,8,batch_first=True)
                self.t2i=nn.MultiheadAttention(D,8,batch_first=True)
                self.ln_i=nn.LayerNorm(D); self.ln_t=nn.LayerNorm(D)
            self.head=nn.Sequential(nn.Linear(2*D,D),nn.ReLU(),nn.Dropout(0.2),nn.Linear(D,14))
        def forward(self, pv, input_ids, attn_mask):
            with torch.no_grad():
                img_tokens = self.img_enc(pixel_values=pv).last_hidden_state
                txt_tokens = self.txt_enc(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            I=self.img_proj(img_tokens); T=self.txt_proj(txt_tokens)
            key_pad=(attn_mask==0)
            if self.use_xattn:
                I2,_=self.i2t(I,T,T,key_padding_mask=key_pad)
                T2,_=self.t2i(T,I,I)
                I=self.ln_i(I+I2); T=self.ln_t(T+T2)
            img_vec=I.mean(1)
            m=(~key_pad).unsqueeze(-1).float()
            txt_vec=(T*m).sum(1)/m.sum(1).clamp(min=1.0)
            return self.head(torch.cat([img_vec,txt_vec],dim=-1))
    return Fusion().to(device)

def macro_metrics(y_true, y_prob, supported_idx):
    from sklearn.metrics import roc_auc_score, average_precision_score
    A,P={},{}
    for j in range(14):
        yt=y_true[:,j]
        if yt.sum()==0 or yt.sum()==len(yt): A[j]=float("nan"); P[j]=float("nan"); continue
        A[j]=roc_auc_score(yt,y_prob[:,j]); P[j]=average_precision_score(yt,y_prob[:,j])
    return (np.nanmean([A[j] for j in supported_idx]), np.nanmean([P[j] for j in supported_idx]), A, P)

def bootstrap_ci(y_true, y_prob, supported_idx, n_boot=1000, seed=0):
    rng=np.random.default_rng(seed); n=len(y_true); AA,PP=[],[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        a,p,_,_=macro_metrics(y_true[idx],y_prob[idx],supported_idx)
        if not np.isnan(a): AA.append(a)
        if not np.isnan(p): PP.append(p)
    ci=lambda v:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if v else (float("nan"),)*2
    return ci(AA), ci(PP)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["e4","e5","e6","concat"], required=True)
    ap.add_argument("--split_csv", default="iu_split.csv")
    ap.add_argument("--labels_csv", default="iu_labels.csv")
    ap.add_argument("--gen_csv", default="iu_generated.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import ViTImageProcessor, AutoTokenizer
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    text_key="real" if args.mode=="e6" else "gen"
    print(f"Mode={args.mode} text={text_key} epochs={args.epochs} lr={args.lr} device={device}")
    joined=load_all(args.split_csv,args.labels_csv,args.gen_csv)
    ds,uid_to_idx=build_dataset_index()
    img_proc=ViTImageProcessor.from_pretrained(IMG_MODEL)
    tokz=AutoTokenizer.from_pretrained(TXT_MODEL)
    by={"train":[],"val":[],"test":[]}
    for uid,info in joined.items():
        if info["split"] in by: by[info["split"]].append(uid)
    print(f"train={len(by['train'])} val={len(by['val'])} test={len(by['test'])}")
    if args.mode=="e5":
        rng=np.random.default_rng(args.seed)
        for split in ("train","val","test"):
            gens=[joined[u]["gen"] for u in by[split]]
            perm=rng.permutation(len(gens))
            for k,u in enumerate(by[split]): joined[u]={**joined[u],"gen":gens[perm[k]]}
        print("E5: generated reports shuffled within each split (control)")
    test_lab=np.stack([joined[u]["labels"] for u in by["test"]])
    test_pos=test_lab.sum(0)
    supported_idx=[j for j in range(14) if test_pos[j]>=MIN_TEST_POS]
    print("Supported:", [CHEXPERT_14[j] for j in supported_idx])
    def loader(s,sh):
        d=FusionDS(by[s],joined,ds,uid_to_idx,img_proc,tokz,text_key,args.maxlen)
        return DataLoader(d,batch_size=args.batch,shuffle=sh,num_workers=4,pin_memory=True)
    tr,va,te=loader("train",True),loader("val",False),loader("test",False)
    model=build_model(args.mode,device)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lossf=torch.nn.BCEWithLogitsLoss()
    scaler=torch.amp.GradScaler("cuda",enabled=(device=="cuda"))
    def evaluate(dl):
        model.eval(); ys,ps=[],[]
        with torch.no_grad():
            for pv,ii,am,y in dl:
                pv,ii,am=pv.to(device),ii.to(device),am.to(device)
                with torch.amp.autocast("cuda",enabled=(device=="cuda")):
                    logits=model(pv,ii,am)
                ps.append(torch.sigmoid(logits).float().cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(ys), np.concatenate(ps)
    best,best_state,bad=-1,None,0
    for ep in range(1,args.epochs+1):
        model.train(); tot=0.0
        for pv,ii,am,y in tr:
            pv,ii,am,y=pv.to(device),ii.to(device),am.to(device),y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda",enabled=(device=="cuda")):
                loss=lossf(model(pv,ii,am),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot+=loss.item()*pv.size(0)
        yv,pvp=evaluate(va)
        a,p,_,_=macro_metrics(yv,pvp,supported_idx)
        print(f"epoch {ep:2d} train_loss={tot/len(by['train']):.4f} val_AUROC={a:.4f} val_AUPRC={p:.4f}")
        if a>best: best=a; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=args.patience: print("early stop"); break
    if best_state: model.load_state_dict(best_state)
    yt,pt=evaluate(te)
    ma,mp,A,P=macro_metrics(yt,pt,supported_idx)
    (alo,ahi),(plo,phi)=bootstrap_ci(yt,pt,supported_idx,seed=args.seed)
    print("\n"+"="*68)
    print(f"{args.mode.upper()} TEST  (macro over {len(supported_idx)} classes)")
    print("="*68)
    print(f"  macro AUROC = {ma:.4f}  (95% CI {alo:.4f}-{ahi:.4f})")
    print(f"  macro AUPRC = {mp:.4f}  (95% CI {plo:.4f}-{phi:.4f})")
    print("-"*68)
    print(f"  {'class':<28}{'AUROC':>8}{'AUPRC':>8}{'test+':>8}")
    for j,name in enumerate(CHEXPERT_14):
        mark="" if j in supported_idx else "  (sparse)"
        print(f"  {name:<28}{A[j]:>8.3f}{P[j]:>8.3f}{int(test_pos[j]):>8}{mark}")
    print("="*68)
    np.savez(f"preds_{args.mode}.npz", y_true=yt, y_prob=pt, supported_idx=np.array(supported_idx), uids=np.array(by["test"]))
    with open(f"results_{args.mode}.json","w") as fh:
        json.dump({"experiment":args.mode,"macro_auroc":ma,"macro_auprc":mp,"auroc_ci":[alo,ahi],"auprc_ci":[plo,phi],
            "supported_classes":[CHEXPERT_14[j] for j in supported_idx],
            "per_class_auroc":{CHEXPERT_14[j]:(None if np.isnan(A[j]) else A[j]) for j in range(14)},
            "per_class_auprc":{CHEXPERT_14[j]:(None if np.isnan(P[j]) else P[j]) for j in range(14)}}, fh, indent=2)
    print(f"Saved preds_{args.mode}.npz + results_{args.mode}.json")

if __name__ == "__main__":
    main()
