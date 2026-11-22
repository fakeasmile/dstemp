# -*- coding: utf-8 -*-
"""微调 chinese-macbert-base 于 ToxiCN 官方 split，测编码器天花板。
用法：python finetune_ceiling.py [--epochs 3] [--maxlen 128]
"""
import json, os, sys, time, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

R = r"D:/dstemp"
MODEL = r"D:/models/chinese-macbert-base"  # 可被 --model 覆盖

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--maxlen", type=int, default=128)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--model", type=str, default=None)
ap.add_argument("--tag", type=str, default="macbert")
args = ap.parse_args()
if args.model: MODEL = args.model

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)

tr = json.load(open(os.path.join(R, "data/raw/TOXICN/train.json"), encoding="utf-8"))
te = json.load(open(os.path.join(R, "data/raw/TOXICN/test.json"), encoding="utf-8"))

Xtr = [r["content"] for r in tr]; ytr = np.array([r["toxic"] for r in tr])
Xte = [r["content"] for r in te]; yte = np.array([r["toxic"] for r in te])

# 90/10 划分：dev 只用于选 epoch 与阈值
itr, iva = train_test_split(np.arange(len(Xtr)), test_size=0.1, stratify=ytr, random_state=42)
print(f"train {len(itr)}  dev {len(iva)}  test {len(Xte)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)

class DS(Dataset):
    def __init__(self, texts, labels, maxlen):
        self.enc = tok(texts, truncation=True, max_length=maxlen, padding=False)
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.enc.items()}, torch.tensor(self.y[i])

def collate(batch):
    feats, ys = zip(*batch)
    out = tok.pad([f for f in feats], padding=True, return_tensors="pt")
    return out, torch.stack(ys)

dl_tr = DataLoader(DS([Xtr[i] for i in itr], ytr[itr], args.maxlen), batch_size=args.bs,
                   shuffle=True, collate_fn=collate)
dl_va = DataLoader(DS([Xtr[i] for i in iva], ytr[iva], args.maxlen), batch_size=32, collate_fn=collate)
dl_te = DataLoader(DS(Xte, yte, args.maxlen), batch_size=32, collate_fn=collate)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = len(dl_tr) * args.epochs
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

def predict(dl):
    model.eval(); P = []
    with torch.no_grad():
        for batch, _ in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            P.append(torch.softmax(model(**batch).logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(P)

def best_thr(y, s):
    c = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    b, t = -1, 0.5
    for x in c:
        f = f1_score(y, (s >= x).astype(int), average="macro")
        if f > b: b, t = f, x
    return t

best = (-1, None, None)
for ep in range(args.epochs):
    t0 = time.time(); model.train(); tot = 0
    for batch, y in dl_tr:
        batch = {k: v.to(dev) for k, v in batch.items()}; y = y.to(dev)
        out = model(**batch, labels=y)
        loss = out.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tot += loss.item()
    pv = predict(dl_va)
    thr = best_thr(ytr[iva], pv)
    f = f1_score(ytr[iva], (pv >= thr).astype(int), average="macro")
    print(f"epoch {ep+1}  loss {tot/len(dl_tr):.4f}  dev MacroF1 {f:.4f}  AUC {roc_auc_score(ytr[iva],pv):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if f > best[0]:
        best = (f, {k: v.clone() for k, v in model.state_dict().items()}, thr)

print("载入最佳 epoch", flush=True)
model.load_state_dict(best[1])
pt = predict(dl_te)
ft = f1_score(yte, (pt >= best[2]).astype(int), average="macro")
print(f"== MacBERT 天花板 ==  test MacroF1 {ft:.4f}  AUC {roc_auc_score(yte, pt):.4f}  thr {best[2]:.3f}  devF1 {best[0]:.4f}", flush=True)
np.save(os.path.join(R, f"results/{args.tag}_test_prob.npy"), pt)
