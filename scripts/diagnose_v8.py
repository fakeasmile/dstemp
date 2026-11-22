# -*- coding: utf-8 -*-
"""
v8 概念向量的决定性诊断：
  A. 信息天花板：ẑ → y 的最优可达性能（多种假设类搜索）
  B. 违反设计的诊断上界：逐概念用标签做单调重标定
  C. 噪声鲁棒性：独立噪声是否能被 26 维平均自然抑制
  D. 全向量错误 vs 局部错误：纠错器的真实作用空间
  E. 错误结构：按 expression / topic / 长度 / 平台 分解
输出：results/v8_diagnosis.txt
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

ROOT = r"D:/dstemp"
PROC = os.path.join(ROOT, "data/processed/TOXICN/v8_glm-4-9b-chat")
RAW = os.path.join(ROOT, "data/raw/TOXICN")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

def load_split(split):
    raw = json.load(open(os.path.join(RAW, f"{split}.json"), encoding="utf-8"))
    obj = json.load(open(os.path.join(PROC, f"items_v8_glm-4-9b-chat_{split}.json"), encoding="utf-8"))
    mat, meta = obj["matrix"], obj["meta"]
    items = [d["item_id"] for d in meta["item_definitions"]]
    ids = [f"TOXICN_{split}_{i:04d}" for i in range(len(raw))]
    X = np.array([[mat[s][it] for it in items] for s in ids], dtype=np.float64)
    y = np.array([r["toxic"] for r in raw], dtype=int)
    return X, y, items, raw

Xtr, ytr, items, raw_tr = load_split("train")
Xte, yte, _, raw_te = load_split("test")
K = len(items)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def best_thr(y, s):
    cand = np.unique(np.quantile(np.sort(s), np.linspace(0.01, 0.99, 200)))
    best, bt = -1, 0.5
    for t in cand:
        f = f1_score(y, (s >= t).astype(int), average="macro")
        if f > best: best, bt = f, t
    return bt

def cv_oof(make, X, y, scale=False, seed=42):
    skf_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y)); thrs = []
    for tr, va in skf_.split(X, y):
        Xa, Xb = X[tr], X[va]
        if scale:
            sc = StandardScaler().fit(Xa); Xa, Xb = sc.transform(Xa), sc.transform(Xb)
        m = make().fit(Xa, y[tr])
        oof[va] = m.predict_proba(Xb)[:, 1]
        thrs.append(best_thr(y[tr], m.predict_proba(Xa)[:, 1]))
    return oof, float(np.mean(thrs))

def report(tag, oof, thr):
    p(f"{tag:42s} F1@折内阈值={f1_score(ytr,(oof>=thr).astype(int),average='macro'):.4f} "
      f" AUC={roc_auc_score(ytr,oof):.4f}")

p("=" * 80); p("A. 信息天花板：任何 ẑ→y 的函数最多能到多少"); p("=" * 80)
cands = {
    "LR(L2)": (lambda: LogisticRegression(max_iter=3000, C=0.3), True),
    "MLP(64)": (lambda: MLPClassifier((64,), max_iter=3000, random_state=0, early_stopping=True), True),
    "MLP(256,128)": (lambda: MLPClassifier((256, 128), max_iter=3000, random_state=0,
                                           early_stopping=True, alpha=1e-4), True),
    "XGB(400x4)": (lambda: xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                             subsample=0.9, colsample_bytree=0.9,
                                             eval_metric="logloss", random_state=0, n_jobs=4), False),
    "XGB(1200x6, 更容量)": (lambda: xgb.XGBClassifier(n_estimators=1200, max_depth=6, learning_rate=0.03,
                                                     subsample=0.8, colsample_bytree=0.7,
                                                     reg_lambda=2.0, eval_metric="logloss",
                                                     random_state=0, n_jobs=4), False),
    "HistGB(强非线性)": (lambda: HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06,
                                                              max_leaf_nodes=31, random_state=0), False),
}
oofs = {}
for tag, (mk, sc) in cands.items():
    o, t = cv_oof(mk, Xtr, ytr, scale=sc)
    oofs[tag] = (o, t); report(tag, o, t)

# 多种子确认最优者不是运气
p("")
p("多种子（5 个划分种子）稳健性，检验最优假设类：")
for tag in ["XGB(400x4)", "XGB(1200x6, 更容量)"]:
    mk, sc = cands[tag]
    fs = []
    for sd in (1, 2, 3, 4, 5):
        o, t = cv_oof(mk, Xtr, ytr, scale=sc, seed=sd)
        fs.append(f1_score(ytr, (o >= t).astype(int), average="macro"))
    p(f"  {tag:24s} 均值 {np.mean(fs):.4f}  标准差 {np.std(fs):.4f}  范围 [{min(fs):.4f},{max(fs):.4f}]")
    p(f"     各划分: {['%.4f'%v for v in fs]}")

p("")
p("=" * 80); p("B. 违规诊断上界：逐概念用 gold 标签做单调重标定（仅作上界，不是可用方法）"); p("=" * 80)
# 折内拟合 isotonic: z_j -> y，再训分类器。表示"每个概念单独能榨出的全部信息"
oof_iso = np.zeros((len(ytr), K)); thr_l = []
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in skf2.split(Xtr, ytr):
    for j in range(K):
        ir = IsotonicRegression(out_of_bounds="clip").fit(Xtr[tr, j], ytr[tr])
        oof_iso[va, j] = ir.predict(Xtr[va, j])
    sc = StandardScaler().fit(oof_iso[tr])
    m = LogisticRegression(max_iter=3000, C=0.3).fit(sc.transform(oof_iso[tr]), ytr[tr])
    thr_l.append(best_thr(ytr[tr], m.predict_proba(sc.transform(oof_iso[tr]))[:, 1]))
    pr = m.predict_proba(sc.transform(oof_iso[va]))[:, 1]
    oof_iso[va, -1] = 1  # 占位，避免误用
    globals().setdefault("_tmp", []).append(0)

# 重新跑一遍干净版本
oof_iso_pr = np.zeros(len(ytr)); thr_l = []
for tr, va in skf2.split(Xtr, ytr):
    Ztr = np.zeros((len(tr), K)); Zva = np.zeros((len(va), K))
    for j in range(K):
        ir = IsotonicRegression(out_of_bounds="clip").fit(Xtr[tr, j], ytr[tr])
        Ztr[:, j] = ir.predict(Xtr[tr, j]); Zva[:, j] = ir.predict(Xtr[va, j])
    sc = StandardScaler().fit(Ztr)
    m = LogisticRegression(max_iter=3000, C=0.3).fit(sc.transform(Ztr), ytr[tr])
    thr_l.append(best_thr(ytr[tr], m.predict_proba(sc.transform(Ztr))[:, 1]))
    oof_iso_pr[va] = m.predict_proba(sc.transform(Zva))[:, 1]
p(f"{'逐概念 isotonic(用标签拟合) + LR':42s} "
  f"F1@折内阈值={f1_score(ytr,(oof_iso_pr>=np.mean(thr_l)).astype(int),average='macro'):.4f} "
  f" AUC={roc_auc_score(ytr,oof_iso_pr):.4f}")
p("说明：这是违规上界（用了 gold toxic 标签当概念监督）。若它也不比原始高多少，")
p("      说明 ẑ 的逐概念信息已被下游分类器榨干，任何单调重标定都无增量。")

p("")
p("=" * 80); p("C. 噪声鲁棒性：独立噪声能否被 26 维平均自然抑制"); p("=" * 80)
best_tag = "XGB(400x4)"
base_o, base_t = oofs[best_tag]
for sigma in (0.05, 0.10, 0.20, 0.30):
    rng = np.random.default_rng(0)
    Xn = np.clip(Xte + rng.normal(0, sigma, Xte.shape), 0, 1)
    m = cands[best_tag][0]().fit(Xtr, ytr)
    pr = m.predict_proba(Xn)[:, 1]
    p(f"  加 N(0,{sigma:.2f}) 独立噪声 → test MacroF1={f1_score(yte,(pr>=base_t).astype(int),average='macro'):.4f} "
      f"AUC={roc_auc_score(yte,pr):.4f}")
m = cands[best_tag][0]().fit(Xtr, ytr)
pr0 = m.predict_proba(Xte)[:, 1]
p(f"  无噪声基线                  → test MacroF1={f1_score(yte,(pr0>=base_t).astype(int),average='macro'):.4f} "
  f"AUC={roc_auc_score(yte,pr0):.4f}")
# 相关噪声（模拟"整条文本被误判"的系统性误差）
for rho in (0.5, 0.8):
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.15, (Xte.shape[0], 1))
    indiv = rng.normal(0, 0.15, Xte.shape)
    Xn = np.clip(Xte + np.sqrt(rho) * common + np.sqrt(1 - rho) * indiv, 0, 1)
    pr = m.predict_proba(Xn)[:, 1]
    p(f"  相关噪声(共同因子 ρ={rho})  → test MacroF1={f1_score(yte,(pr>=base_t).astype(int),average='macro'):.4f} "
      f"AUC={roc_auc_score(yte,pr):.4f}")

p("")
p("=" * 80); p("D. 全向量错误 vs 局部错误：纠错器真正的作用空间"); p("=" * 80)
mte = m.predict_proba(Xte)[:, 1]
pred = (mte >= base_t).astype(int)
mean_z = Xte.mean(1)
fn = (yte == 1) & (pred == 0); fp = (yte == 0) & (pred == 1)
p(f"test: n={len(yte)}  FN={fn.sum()}  FP={fp.sum()}  总错误={fn.sum()+fp.sum()}")
p(f"  mean(P_yes) 分布: y=1 → {mean_z[yte==1].mean():.3f}±{mean_z[yte==1].std():.3f} ; "
  f"y=0 → {mean_z[yte==0].mean():.3f}±{mean_z[yte==0].std():.3f}")
p(f"  FN 的 mean(P_yes) = {mean_z[fn].mean():.3f}   FP 的 mean(P_yes) = {mean_z[fp].mean():.3f}")
unanimous_fn = fn & (Xte.max(1) < 0.5)
unanimous_fp = fp & (Xte.min(1) > 0.5)
p(f"  「全向量一致错」（所有概念都 <0.5 但 y=1）: {unanimous_fn.sum()}")
p(f"  「全向量一致错」（所有概念都 >0.5 但 y=0）: {unanimous_fp.sum()}")
p(f"  即：FN 中 {unanimous_fn.sum()/max(fn.sum(),1)*100:.1f}% 是整个向量都低，"
  f"FP 中 {unanimous_fp.sum()/max(fp.sum(),1)*100:.1f}% 是整个向量都高")
# 局部错误：少数维度异常
spread_fn = (Xte[fn].std(1) if fn.sum() else np.array([0]))
p(f"  FN 样本内概念横向标准差均值 = {spread_fn.mean():.3f}（越小=整条一起低，越大=局部错）")

p("")
p("=" * 80); p("E. 错误结构分解（test, XGBoost 全训练集拟合）"); p("=" * 80)
df = pd.DataFrame({
    "y": yte, "pred": pred, "prob": mte,
    "expr": [r["expression"] for r in raw_te],
    "topic": [r["topic"] for r in raw_te],
    "plat": [r["platform"] for r in raw_te],
    "len": [r["length"] for r in raw_te],
    "ttype": [r["toxic_type"] for r in raw_te],
})
p("按 expression（0=不适用/非目标类, 1=显式, 2=隐式, 3=转述）:")
for v, g in df.groupby("expr"):
    f1 = f1_score(g.y, g.pred, average="macro") if g.y.nunique() > 1 else float("nan")
    p(f"   expr={v}  n={len(g):4d}  正例率={g.y.mean():.2f}  MacroF1={f1:.4f}  错误率={(g.y!=g.pred).mean():.3f}")
p("按 topic:")
for v, g in df.groupby("topic"):
    p(f"   {v:8s} n={len(g):4d}  正例率={g.y.mean():.2f}  MacroF1={f1_score(g.y,g.pred,average='macro'):.4f}  "
      f"错误率={(g.y!=g.pred).mean():.3f}")
p("按平台:")
for v, g in df.groupby("plat"):
    p(f"   {v:8s} n={len(g):4d}  错误率={(g.y!=g.pred).mean():.3f}")
df["lenbin"] = pd.qcut(df["len"], 4, labels=["Q1短", "Q2", "Q3", "Q4长"], duplicates="drop")
p("按文本长度四分位:")
for v, g in df.groupby("lenbin", observed=True):
    p(f"   {str(v):8s} n={len(g):4d}  错误率={(g.y!=g.pred).mean():.3f}  mean(P_yes)={Xte[g.index].mean(1).mean():.3f}")

p("")
p("=" * 80); p("F. 概念对「隐式表达」的敏感度：隐式样本上概念是否整体偏低"); p("=" * 80)
for j, it in enumerate(items):
    e1 = Xte[(yte == 1) & (np.array([r["expression"] for r in raw_te]) == 1), j].mean()
    e2 = Xte[(yte == 1) & (np.array([r["expression"] for r in raw_te]) == 2), j].mean()
    e3 = Xte[(yte == 1) & (np.array([r["expression"] for r in raw_te]) == 3), j].mean()
    p(f"   {it}: 显式 {e1:.3f}  隐式 {e2:.3f}  转述 {e3:.3f}   隐式-显式 {e2-e1:+.3f}")

open(os.path.join(OUT, "v8_diagnosis.txt"), "w", encoding="utf-8").write("\n".join(lines))
p("\n已写入 results/v8_diagnosis.txt")
