# -*- coding: utf-8 -*-
"""
Learned Concept Measurement (LCM)
把 PLM 放在「测量端」而不是「修正端」：

  LLM 路线（原方案）: x --LLM--> ẑ --g--> z̃ --clf--> ŷ      受 DPI 限制
  LCM  路线（本脚本）: x --BERT--> e --h_θ--> z --clf_lin--> ŷ   不受 DPI 限制

  L = BCE(ŷ, y) + λ · MSE(z, ẑ)
    λ=∞  → 复刻 LLM 概念（可解释，性能≈0.79）
    λ=0  → 自由瓶颈（性能↑，概念漂移↑）

同时扫瓶颈宽度 K，得到「概念粒度—性能—保真度」三维曲线。
"""
import json, os, time
import numpy as np
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

R = r"D:/dstemp"; MODEL = r"D:/models/chinese-macbert-base"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

tr = json.load(open(f"{R}/data/raw/TOXICN/train.json", encoding="utf-8"))
te = json.load(open(f"{R}/data/raw/TOXICN/test.json", encoding="utf-8"))
Xtr = [r["content"] for r in tr]; ytr = np.array([r["toxic"] for r in tr])
Xte = [r["content"] for r in te]; yte = np.array([r["toxic"] for r in te])

def concepts(split, n):
    o = json.load(open(f"{R}/data/processed/TOXICN/v8_glm-4-9b-chat/items_v8_glm-4-9b-chat_{split}.json", encoding="utf-8"))
    it = [d["item_id"] for d in o["meta"]["item_definitions"]]
    ids = [f"TOXICN_{split}_{i:04d}" for i in range(n)]
    return np.array([[o["matrix"][k][j] for j in it] for k in ids], dtype=np.float32), it

Ztr, items = concepts("train", len(tr)); Zte, _ = concepts("test", len(te))
K0 = Ztr.shape[1]

# ---------- 冻结 BERT 抽 [CLS] ----------
tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL).to(dev).eval()

@torch.no_grad()
def embed(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], truncation=True, max_length=160, padding=True, return_tensors="pt").to(dev)
        out.append(enc(**b).last_hidden_state[:, 0].cpu().numpy())
    return np.concatenate(out)

t0 = time.time()
Etr = embed(Xtr); Ete = embed(Xte)
print(f"[cls] train {Etr.shape} test {Ete.shape}  ({time.time()-t0:.0f}s)", flush=True)

itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=42)

def best_thr(y, s):
    c = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    b, t = -1, 0.5
    for x in c:
        f = f1_score(y, (s >= x).astype(int), average="macro")
        if f > b: b, t = f, x
    return t

class LCM(nn.Module):
    """e -> 线性 -> z(sigmoid, K维) -> 线性 -> logit。刻意低容量，避免记忆。"""
    def __init__(self, d, K):
        super().__init__()
        self.to_z = nn.Linear(d, K)
        self.clf = nn.Linear(K, 1)
        nn.init.zeros_(self.clf.bias)
    def forward(self, e):
        z = torch.sigmoid(self.to_z(e))
        return z, self.clf(z).squeeze(-1)

def run(K, lam, epochs=40, lr=3e-3, seed=0, patience=6):
    torch.manual_seed(seed)
    mu, sd = Etr[itr].mean(0), Etr[itr].std(0) + 1e-6
    E = torch.tensor((Etr - mu) / sd, device=dev); Y = torch.tensor(ytr, device=dev).float()
    Z = torch.tensor(Ztr, device=dev)
    m = LCM(Etr.shape[1], K).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-2)
    idx = itr.copy()   # 必须是全局索引；不能用 arange(len(itr))
    best = (-1.0, None, 0.5, -1)   # (devF1, state, thr, epoch)
    for ep in range(epochs):
        m.train(); np.random.shuffle(idx)
        for j in range(0, len(idx), 256):
            b = torch.tensor(idx[j:j+256], device=dev)
            z, logit = m(E[b])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, Y[b])
            if lam > 0 and K == K0:
                loss = loss + lam * nn.functional.mse_loss(z, Z[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(torch.tensor((Etr[iva] - mu) / sd, device=dev))[1]).cpu().numpy()
        thr = best_thr(ytr[iva], pv)
        f1v = f1_score(ytr[iva], (pv >= thr).astype(int), average="macro")
        if f1v > best[0]:
            best = (f1v, {k: v.clone() for k, v in m.state_dict().items()}, thr, ep)
        elif ep - best[3] >= patience:
            break
    m.load_state_dict(best[1]); m.eval()
    with torch.no_grad():
        Et = torch.tensor((Ete - mu) / sd, device=dev)
        zt_t, lt = m(Et)
        pt = torch.sigmoid(lt).cpu().numpy(); zt = zt_t.cpu().numpy()
    f1t = f1_score(yte, (pt >= best[2]).astype(int), average="macro")
    auc = roc_auc_score(yte, pt)
    fid = np.nan
    if K == K0:
        fid = float(np.mean([np.corrcoef(zt[:, i], Zte[:, i])[0, 1] for i in range(K0)]))
    return dict(K=K, lam=lam, dev_f1=best[0], test_f1=f1t, auc=auc, fid=fid, thr=best[2], ep=best[3])

print("\n=== 参考：LLM 概念 ẑ 直接接线性分类器 ===", flush=True)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=3000, C=0.3).fit(Ztr[itr], ytr[itr])
pv = lr.predict_proba(Ztr[iva])[:, 1]; pt = lr.predict_proba(Zte)[:, 1]
thr = best_thr(ytr[iva], pv)
print(f"  test F1={f1_score(yte,(pt>=thr).astype(int),average='macro'):.4f}  AUC={roc_auc_score(yte,pt):.4f}", flush=True)

print("\n=== LCM：冻结 BERT 表示 → K 维概念瓶颈 → 线性分类 ===", flush=True)
print(f"{'K':>5} {'λ':>6} {'devF1':>8} {'testF1':>8} {'AUC':>8} {'保真度':>8}", flush=True)
rows = []
for K in (26, 64, 200, 512):
    for lam in ((0.0, 0.5, 2.0, 10.0) if K == 26 else (0.0,)):
        r = run(K, lam)
        rows.append(r)
        print(f"{K:>5} {lam:>6.1f} {r['dev_f1']:>8.4f} {r['test_f1']:>8.4f} {r['auc']:>8.4f} {r['fid']:>8.3f}", flush=True)

import pandas as pd
pd.DataFrame(rows).to_csv(f"{R}/results/lcm_sweep.csv", index=False)
print("\n已写入 results/lcm_sweep.csv", flush=True)
