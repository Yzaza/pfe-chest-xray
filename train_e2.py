#!/usr/bin/env python3
"""E2: text-only classifier (generated report -> 14 classes), frozen ClinicalBERT."""
import argparse, csv, json
import numpy as np
CHEXPERT_14 = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
    "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"]
MIN_TEST_POS = 10
TXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
def load_all(split_csv, labels_csv, gen_csv):
    splits, labels, gens = {}, {}, {}
    with open(split_csv, newline="") as fh:
        for r in csv.DictReader(fh): splits[int(r["uid"])]=(r["split"],int(r.get("has_frontal","1")))
    with open(labels_csv, newline="") as fh:
        for r in csv.DictReader(fh): labels[int(r["uid"])]=np.array([int(r[c]) for c in CHEXPERT_14],dtype=np.float32)
    with open(gen_csv, newline="") as fh:
        for r in csv.DictReader(fh): gens[int(r["uid"])]=r.get("generated_report","")
    joined={}
    for uid,(split,hf) in splits.items():
        if uid not in labels or not hf: continue
        joined[uid]={"split":split,"labels":labels[uid],"gen":gens.get(uid,"")}
    return joined
class TxtDS:
    def __init__(self,uids,joined,tokz,maxlen): self.uids=uids;self.joined=joined;self.tokz=tokz;self.maxlen=maxlen
    def __len__(self): return len(self.uids)
    def __getitem__(self,i):
        import torch
        uid=self.uids[i]; txt=self.joined[uid]["gen"] or ""
        enc=self.tokz(txt,truncation=True,max_length=self.maxlen,padding="max_length",return_tensors="pt")
        return enc["input_ids"][0],enc["attention_mask"][0],torch.from_numpy(self.joined[uid]["labels"])
def build_model():
    import torch.nn as nn
    from transformers import AutoModel
    txt=AutoModel.from_pretrained(TXT_MODEL)
    for p in txt.parameters(): p.requires_grad=False
    dim=txt.config.hidden_size
    class M(nn.Module):
        def __init__(self):
            super().__init__(); self.txt=txt
            self.head=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,14))
        def forward(self,ii,am):
            import torch
            with torch.no_grad(): h=self.txt(input_ids=ii,attention_mask=am).last_hidden_state
            m=am.unsqueeze(-1).float(); pooled=(h*m).sum(1)/m.sum(1).clamp(min=1.0)
            return self.head(pooled)
    return M()
def macro_metrics(y_true,y_prob,sup):
    from sklearn.metrics import roc_auc_score, average_precision_score
    A,P={},{}
    for j in range(14):
        yt=y_true[:,j]
        if yt.sum()==0 or yt.sum()==len(yt): A[j]=float("nan");P[j]=float("nan");continue
        A[j]=roc_auc_score(yt,y_prob[:,j]); P[j]=average_precision_score(yt,y_prob[:,j])
    return np.nanmean([A[j] for j in sup]),np.nanmean([P[j] for j in sup]),A,P
def bootstrap_ci(y_true,y_prob,sup,n_boot=1000,seed=0):
    rng=np.random.default_rng(seed);n=len(y_true);AA,PP=[],[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n); a,p,_,_=macro_metrics(y_true[idx],y_prob[idx],sup)
        if not np.isnan(a): AA.append(a)
        if not np.isnan(p): PP.append(p)
    ci=lambda v:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if v else (float("nan"),)*2
    return ci(AA),ci(PP)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--maxlen",type=int,default=128)
    ap.add_argument("--patience",type=int,default=5); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"E2 text-only epochs={args.epochs} lr={args.lr} device={device}")
    joined=load_all("iu_split.csv","iu_labels.csv","iu_generated.csv")
    tokz=AutoTokenizer.from_pretrained(TXT_MODEL)
    by={"train":[],"val":[],"test":[]}
    for uid,info in joined.items():
        if info["split"] in by: by[info["split"]].append(uid)
    print(f"train={len(by['train'])} val={len(by['val'])} test={len(by['test'])}")
    test_lab=np.stack([joined[u]["labels"] for u in by["test"]]); test_pos=test_lab.sum(0)
    sup=[j for j in range(14) if test_pos[j]>=MIN_TEST_POS]
    def loader(s,sh): return DataLoader(TxtDS(by[s],joined,tokz,args.maxlen),batch_size=args.batch,shuffle=sh,num_workers=4,pin_memory=True)
    tr,va,te=loader("train",True),loader("val",False),loader("test",False)
    model=build_model().to(device)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    lossf=torch.nn.BCEWithLogitsLoss(); scaler=torch.amp.GradScaler("cuda",enabled=(device=="cuda"))
    def evaluate(dl):
        model.eval(); ys,ps=[],[]
        with torch.no_grad():
            for ii,am,y in dl:
                ii,am=ii.to(device),am.to(device)
                with torch.amp.autocast("cuda",enabled=(device=="cuda")): logits=model(ii,am)
                ps.append(torch.sigmoid(logits).float().cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(ys),np.concatenate(ps)
    best,bs,bad=-1,None,0
    for ep in range(1,args.epochs+1):
        model.train(); tot=0.0
        for ii,am,y in tr:
            ii,am,y=ii.to(device),am.to(device),y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda",enabled=(device=="cuda")): loss=lossf(model(ii,am),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot+=loss.item()*ii.size(0)
        yv,pv=evaluate(va); a,p,_,_=macro_metrics(yv,pv,sup)
        print(f"epoch {ep:2d} train_loss={tot/len(by['train']):.4f} val_AUROC={a:.4f} val_AUPRC={p:.4f}")
        if a>best: best=a; bs={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=args.patience: print("early stop"); break
    if bs: model.load_state_dict(bs)
    yt,pt=evaluate(te); ma,mp,A,P=macro_metrics(yt,pt,sup)
    (alo,ahi),(plo,phi)=bootstrap_ci(yt,pt,sup,seed=args.seed)
    print("\n"+"="*68); print(f"E2 TEST (macro over {len(sup)} classes)"); print("="*68)
    print(f"  macro AUROC = {ma:.4f}  (95% CI {alo:.4f}-{ahi:.4f})")
    print(f"  macro AUPRC = {mp:.4f}  (95% CI {plo:.4f}-{phi:.4f})")
    print("-"*68)
    for j,name in enumerate(CHEXPERT_14):
        mark="" if j in sup else "  (sparse)"
        print(f"  {name:<28}{A[j]:>8.3f}{P[j]:>8.3f}{int(test_pos[j]):>8}{mark}")
    print("="*68)
    np.savez("preds_e2.npz",y_true=yt,y_prob=pt,supported_idx=np.array(sup),uids=np.array(by["test"]))
    json.dump({"experiment":"E2","macro_auroc":ma,"macro_auprc":mp,"auroc_ci":[alo,ahi],"auprc_ci":[plo,phi],
        "per_class_auroc":{CHEXPERT_14[j]:(None if np.isnan(A[j]) else A[j]) for j in range(14)},
        "per_class_auprc":{CHEXPERT_14[j]:(None if np.isnan(P[j]) else P[j]) for j in range(14)}}, open("results_e2.json","w"), indent=2)
    print("Saved preds_e2.npz + results_e2.json")
if __name__=="__main__": main()
