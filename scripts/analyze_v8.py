# -*- coding: utf-8 -*-
"""
核对 v8 概念向量（TOXICN / glm-4-9b-chat）的基本性质，
并复算研究方案 v3 中依赖 v7 180 维矩阵报告过的各项统计量。
输出：results/v8_audit_report.txt  + 若干 csv
"""
import json, os, sys, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb

ROOT = r"D:/dstemp"
PROC = os.path.join(ROOT, "data/processed/TOXICN/v8_glm-4-9b-chat")
RAW = os.path.join(ROOT, "data/raw/TOXICN")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

lines = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)

# ---------- 载入 ----------
def load_split(split):
    raw = json.load(open(os.path.join(RAW, f"{split}.json"), encoding="utf-8"))
    obj = json.load(open(os.path.join(PROC, f"items_v8_glm-4-9b-chat_{split}.json"), encoding="utf-8"))
    mat = obj["matrix"]
    meta = obj["meta"]
    items = [d["item_id"] for d in meta["item_definitions"]]
    ids = [f"TOXICN_{split}_{i:04d}" for i in range(len(raw))]
    assert set(ids) == set(mat.keys()), f"{split}: id 不匹配 {len(ids)} vs {len(mat)}"
    X = np.array([[mat[s][it] for it in items] for s in ids], dtype=np.float64)
    y = np.array([r["toxic"] for r in raw], dtype=int)
    return X, y, items, raw, meta

Xtr, ytr, items, raw_tr, meta_tr = load_split("train")
Xte, yte, _, raw_te, _ = load_split("test")

p("=" * 78)
p("0. 规模与标签分布")
p("=" * 78)
p(f"train X {Xtr.shape}  test X {Xte.shape}  概念数 K={len(items)}")
p(f"train toxic 比例 {ytr.mean():.4f}  (n+={ytr.sum()}, n-={(1-ytr).sum()})")
p(f"test  toxic 比例 {yte.mean():.4f}  (n+={yte.sum()}, n-={(1-yte).sum()})")
p(f"特征值域 [{Xtr.min():.6f}, {Xtr.max():.6f}]   NaN={np.isnan(Xtr).sum()}")
p(f"随机猜测 Macro-F1 参考 = {f1_score(yte, np.zeros_like(yte), average='macro'):.4f} (全预测0)")

# ---------- 概念层分布 ----------
p("")
p("=" * 78)
p("1. 概念层分布：是否为退化/饱和维度")
p("=" * 78)
rows = []
for j, it in enumerate(items):
    c = Xtr[:, j]
    rows.append(dict(item=it, mean=c.mean(), std=c.std(),
                     p_lt_01=(c < 0.1).mean(), p_gt_09=(c > 0.9).mean(),
                     auc=roc_auc_score(ytr, c), ap=average_precision_score(ytr, c)))
df_c = pd.DataFrame(rows).sort_values("auc", ascending=False)
p(df_c.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
p("")
p(f"单概念 AUC 最高 {df_c.auc.max():.4f}（{df_c.iloc[0]['item']}），最低 {df_c.auc.min():.4f}（{df_c.iloc[-1]['item']}）")
p(f"AUC<0.55 的概念数 = {(df_c.auc<0.55).sum()}   AUC>0.75 的概念数 = {(df_c.auc>0.75).sum()}")
df_c.to_csv(os.path.join(OUT, "v8_concept_stats.csv"), index=False)

# ---------- 冗余 / 有效秩 ----------
p("")
p("=" * 78)
p("2. 冗余结构与有效秩")
p("=" * 78)
C = np.corrcoef(Xtr, rowvar=False)
iu = np.triu_indices_from(C, k=1)
absr = np.abs(C[iu])
p(f"平均 |Pearson| = {absr.mean():.4f}   中位数 = {np.median(absr):.4f}")
p(f"|r|>0.8 的概念对 = {(absr>0.8).sum()} / {len(absr)}   最大 |r| = {absr.max():.4f}")
pair = [(items[i], items[j], C[i, j]) for i, j in zip(*iu)]
pair.sort(key=lambda t: -abs(t[2]))
p("最强相关前 8 对：")
for a, b, r in pair[:8]:
    p(f"   {a}-{b}  r={r:+.4f}")

# 基率控制：稀疏/低基率概念会压低 Pearson，另报 Spearman + 标准化后的相关
from scipy.stats import spearmanr, rankdata
Xs = np.apply_along_axis(rankdata, 0, Xtr)
Cs = np.corrcoef(Xs, rowvar=False)
absrs = np.abs(Cs[iu])
p(f"[基率控制] 平均 |Spearman| = {absrs.mean():.4f}   |r|>0.8 对 = {(absrs>0.8).sum()}")

# 有效秩
def eff_rank_mats(M):
    M = M - M.mean(0)
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-12]
    pr = s / s.sum()
    ent = -np.sum(pr * np.log(pr))
    return dict(entropic=np.exp(ent), stable=(s.sum() ** 2) / (s ** 2).sum(),
                n_ge1=int((s > 1).sum()), kmax=len(s))

st = StandardScaler().fit(Xtr)
Z = st.transform(Xtr)
er_raw = eff_rank_mats(Xtr)
er_z = eff_rank_mats(Z)
p(f"有效秩（原始尺度，熵定义） = {er_raw['entropic']:.2f} / K={len(items)}")
p(f"有效秩（标准化后，熵定义）  = {er_z['entropic']:.2f}")
p(f"stable rank（标准化后）      = {er_z['stable']:.2f}")
p(f"奇异值 >1 的个数            = {er_z['n_ge1']}")
pca = PCA(n_components=len(items)).fit(Z)
ev = pca.explained_variance_ratio_
p(f"PC1 解释方差 {ev[0]*100:.1f}%  PC1+2 {ev[:2].sum()*100:.1f}%  PC1-5 {ev[:5].sum()*100:.1f}%")
for target in (0.90, 0.95, 0.99):
    k = int(np.searchsorted(np.cumsum(ev), target) + 1)
    p(f"累计 {target*100:.0f}% 方差需要 {k} 个主成分")
np.save(os.path.join(OUT, "v8_pca_evr.npy"), ev)
dfC = pd.DataFrame(C, index=items, columns=items)
dfC.to_csv(os.path.join(OUT, "v8_concept_corr.csv"))

# ---------- 题目敏感性：同题不同值 / 同值不同题 ----------
p("")
p("=" * 78)
p("3. 题目是否真的在影响首 token 概率（值碰撞诊断）")
p("=" * 78)
n = Xtr.shape[0]
dup_frac = []
for i, j in itertools.combinations(range(len(items)), 2):
    eq = (np.abs(Xtr[:, i] - Xtr[:, j]) < 1e-9).mean()
    dup_frac.append((items[i], items[j], eq))
dup_frac.sort(key=lambda t: -t[2])
p("完全相等的样本占比最高的一批概念对（<1e-9 判等，值精度 1e-6）：")
for a, b, f in dup_frac[:10]:
    p(f"   {a}-{b}  相等比例 {f*100:.2f}%")
p(f"超过 5% 样本完全相等的概念对数量：{sum(1 for _,_,f in dup_frac if f>0.05)} / {len(dup_frac)}")

per_row_distinct = np.array([len(np.unique(np.round(r, 6))) for r in Xtr[:2000]])
p(f"每条文本的 26 个概念中不同取值个数：均值 {per_row_distinct.mean():.2f} 中位 {np.median(per_row_distinct):.0f} 最小 {per_row_distinct.min()}")

# ---------- 下游 CBM：复现基线 ----------
p("")
p("=" * 78)
p("4. 下游 CBM 基线（5 折 CV，阈值在验证折内选择）")
p("=" * 78)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def best_thr(y, s):
    cand = np.unique(np.round(np.sort(s), 4))
    if len(cand) > 200:
        cand = np.quantile(cand, np.linspace(0.01, 0.99, 200))
    best, bt = -1, 0.5
    for t in cand:
        f = f1_score(y, (s >= t).astype(int), average="macro")
        if f > best:
            best, bt = f, t
    return bt

def run_cv(make_model, X, y, tag, scale=False):
    oof = np.zeros(len(y))
    oof05 = np.zeros(len(y))
    thrs = []
    for tr, va in skf.split(X, y):
        Xa, Xb = X[tr], X[va]
        if scale:
            sc = StandardScaler().fit(Xa); Xa, Xb = sc.transform(Xa), sc.transform(Xb)
        m = make_model().fit(Xa, y[tr])
        oof[va] = m.predict_proba(Xb)[:, 1]
        t = best_thr(y[tr], m.predict_proba(Xa)[:, 1])
        thrs.append(t)
    f1_05 = f1_score(y, (oof >= 0.5).astype(int), average="macro")
    f1_bt = f1_score(y, (oof >= np.mean(thrs)).astype(int), average="macro")
    auc = roc_auc_score(y, oof)
    p(f"{tag:34s} MacroF1@0.5={f1_05:.4f}  MacroF1@折内阈值={f1_bt:.4f}  AUC={auc:.4f}")
    return oof, dict(tag=tag, f1_05=f1_05, f1_bt=f1_bt, auc=auc)

res = []
res.append(run_cv(lambda: LogisticRegression(max_iter=2000, C=1.0), Xtr, ytr, "LR (L2, C=1)"))
res.append(run_cv(lambda: LogisticRegression(max_iter=3000, C=0.1, penalty="l2"), Xtr, ytr, "LR (L2, C=0.1)"))
res.append(run_cv(lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                        random_state=0, early_stopping=True), Xtr, ytr, "MLP (64)", scale=True))
res.append(run_cv(lambda: xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                            subsample=0.9, colsample_bytree=0.9,
                                            eval_metric="logloss", random_state=0,
                                            n_jobs=4), Xtr, ytr, "XGBoost (400x4)"))

# ---------- 极简基线：概念压缩到 1 维 ----------
p("")
p("-" * 78)
p("极简基线：把 26 维概念压成 1 维，看还剩多少信息")
p("-" * 78)
agg_mean = Xtr.mean(1, keepdims=True)
run_cv(lambda: LogisticRegression(max_iter=1000), agg_mean, ytr, "LR on mean(P_yes)", scale=True)

top1 = str(df_c.iloc[0]["item"])
j1 = items.index(top1)
run_cv(lambda: LogisticRegression(max_iter=1000), Xtr[:, [j1]], ytr, f"LR on 单概念 {top1}", scale=True)

for k in (2, 3, 5, 10):
    Zk = PCA(n_components=k, random_state=0).fit_transform(Z)
    run_cv(lambda: LogisticRegression(max_iter=2000), Zk, ytr, f"LR on PCA-{k}", scale=True)

# 概念分组消融：只用某一构念
p("")
p("-" * 78)
p("按构念分组消融（只用该组概念）")
p("-" * 78)
groups = {}
for it in items:
    groups.setdefault(it[0], []).append(items.index(it))
for g, idx in groups.items():
    run_cv(lambda: LogisticRegression(max_iter=2000), Xtr[:, idx], ytr,
           f"LR 仅构念 {g} ({len(idx)}维)", scale=True)

# 标签相关维度：L1 选择 + 置换重要性
p("")
p("-" * 78)
p("标签相关维度：到底有多少概念在真正贡献预测")
p("-" * 78)
sc_all = StandardScaler().fit(Xtr)
for Cval in (0.03, 0.1, 0.3, 1.0):
    m1 = LogisticRegression(penalty="l1", solver="liblinear", C=Cval, max_iter=5000)
    m1.fit(sc_all.transform(Xtr), ytr)
    nz = int((np.abs(m1.coef_[0]) > 1e-8).sum())
    p(f"L1(C={Cval}) 非零概念 {nz}/{len(items)}")

from sklearn.inspection import permutation_importance
xgb_full = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric="logloss", random_state=0, n_jobs=4).fit(Xtr, ytr)
pi = permutation_importance(xgb_full, Xte, yte, n_repeats=10, random_state=0,
                            scoring="roc_auc", n_jobs=4)
imp = pd.DataFrame({"item": items, "dAUC": pi.importances_mean}).sort_values("dAUC", ascending=False)
p("置换重要性（test, AUC 下降）：")
for _, r in imp.iterrows():
    p(f"   {r['item']}  {r['dAUC']:+.4f}")
p(f"重要性为正的概念数 = {(imp.dAUC>0).sum()} / {len(items)}")
imp.to_csv(os.path.join(OUT, "v8_perm_importance.csv"), index=False)

# ---------- 全训练集训练 → 测试集 ----------
p("")
p("=" * 78)
p("5. 训练集全量拟合 → 官方测试集（对外可比口径）")
p("=" * 78)
for tag, mk, sc in [
    ("LR(L2)", lambda: LogisticRegression(max_iter=3000, C=0.1), True),
    ("XGBoost", lambda: xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                          subsample=0.9, colsample_bytree=0.9,
                                          eval_metric="logloss", random_state=0, n_jobs=4), False),
]:
    Xa, Xb = Xtr, Xte
    if sc:
        s = StandardScaler().fit(Xa); Xa, Xb = s.transform(Xa), s.transform(Xb)
    m = mk().fit(Xa, ytr)
    pr = m.predict_proba(Xb)[:, 1]
    # 阈值来自训练集折内
    t = best_thr(ytr, m.predict_proba(Xa)[:, 1])
    p(f"{tag:10s} test MacroF1@0.5={f1_score(yte,(pr>=0.5).astype(int),average='macro'):.4f} "
      f"@thr({t:.3f})={f1_score(yte,(pr>=t).astype(int),average='macro'):.4f} "
      f"AUC={roc_auc_score(yte,pr):.4f}  Acc={(pr>=t).mean() if False else ((pr>=t).astype(int)==yte).mean():.4f}")

# ---------- 与标签的耦合：概念是否只是标签的影子 ----------
p("")
p("=" * 78)
p("6. 概念值与 gold 子字段（toxic_type / expression / target）的一致性")
p("=" * 78)
tt = np.array([r["toxic_type"] for r in raw_tr])
ex = np.array([r["expression"] for r in raw_tr])
m = ytr == 1
p(f"toxic=1 子集 n={m.sum()}；toxic_type 分布 {np.bincount(tt[m])}")
for j, it in enumerate(items):
    c = Xtr[m, j]
    p(f"   {it}: 子字段A均值 {c[tt[m]==0].mean():.3f} vs {c[tt[m]==1].mean():.3f} | "
      f"expression0 {c[ex[m]==0].mean():.3f} vs 1 {c[ex[m]==1].mean():.3f}")

# ---------- 结论摘要 ----------
p("")
p("=" * 78)
p("摘要")
p("=" * 78)
p(f"K={len(items)}；有效秩(标准化)={er_z['entropic']:.2f}；PC1={ev[0]*100:.1f}%")
p(f"方案v3 报告（v7, K=180）：平均|r|=0.4415, |r|>0.8 对=440, 有效秩=15.17, F1=0.8018")
p(f"本次实测（v8, K={len(items)}）：见上。")

open(os.path.join(OUT, "v8_audit_report.txt"), "w", encoding="utf-8").write("\n".join(lines))
p("\n报告已写入 results/v8_audit_report.txt")
