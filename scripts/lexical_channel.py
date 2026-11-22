# -*- coding: utf-8 -*-
"""
概念瓶颈 + 词法通道：可复现实现与协议检验

三件事：
  1. 记录 0.8119 那次的原始配置（protocol="legacy"）
  2. 用更严格协议重测（protocol="clean"）：vectorizer 折内拟合、阈值取 OOF
  3. 诊断：训练/测试重复、特征块范数失衡、C 敏感性

用法：python lexical_channel.py [legacy|clean|diag]
"""
import json, os, sys, hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix

ROOT = r"D:/dstemp"
PROC = os.path.join(ROOT, "data/processed/TOXICN/v8_glm-4-9b-chat")
RAW = os.path.join(ROOT, "data/raw/TOXICN")

# ---------------- 数据 ----------------
def load_raw(split):
    return json.load(open(os.path.join(RAW, f"{split}.json"), encoding="utf-8"))

def load_concepts(split, n_raw):
    obj = json.load(open(os.path.join(PROC, f"items_v8_glm-4-9b-chat_{split}.json"), encoding="utf-8"))
    items = [d["item_id"] for d in obj["meta"]["item_definitions"]]
    ids = [f"TOXICN_{split}_{i:04d}" for i in range(n_raw)]
    X = np.array([[obj["matrix"][k][it] for it in items] for k in ids], dtype=np.float64)
    return X, items

tr, te = load_raw("train"), load_raw("test")
ytr = np.array([r["toxic"] for r in tr]); yte = np.array([r["toxic"] for r in te])
ctr = [r["content"] for r in tr]; cte = [r["content"] for r in te]
Ztr, items = load_concepts("train", len(tr))
Zte, _ = load_concepts("test", len(te))

VEC_KW = dict(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
              max_features=200000, sublinear_tf=True, norm="l2")

def build(vec, tr_texts, va_texts, tr_z, va_z, scale_blocks=None):
    """scale_blocks: None | 'equal' 把两个特征块的 L2 范数拉平"""
    A = vec.transform(tr_texts); B = vec.transform(va_texts)
    if scale_blocks == "equal":
        # 让词法块与概念块在每行上贡献相同的平方范数
        na = np.sqrt(A.multiply(A).sum(1)).A.ravel() + 1e-12
        nb = np.sqrt((tr_z ** 2).sum(1)) + 1e-12
        # 概念块缩放到与词法块同范数（按训练集平均比例）
        r = np.mean(na) / np.mean(nb)
        tr_z = tr_z * r; va_z = va_z * r
    return hstack([A, csr_matrix(tr_z)]).tocsr(), hstack([B, csr_matrix(va_z)]).tocsr()

def best_thr(y, s):
    c = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    b, t = -1, 0.5
    for x in c:
        f = f1_score(y, (s >= x).astype(int), average="macro")
        if f > b: b, t = f, x
    return t

# ---------------- 两种协议 ----------------
def run_legacy(C=4.0):
    """原始实现：vectorizer 全训练集拟合；阈值取训练折 in-sample 预测"""
    v = TfidfVectorizer(**VEC_KW).fit(ctr)
    A, B = build(v, ctr, cte, Ztr, Zte)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(ytr)); ths = []
    for a, b in skf.split(np.arange(len(ytr)), ytr):
        m = LogisticRegression(max_iter=3000, C=C).fit(A[a], ytr[a])
        oof[b] = m.predict_proba(A[b])[:, 1]
        ths.append(best_thr(ytr[a], m.predict_proba(A[a])[:, 1]))   # in-sample，有偏
    m = LogisticRegression(max_iter=3000, C=C).fit(A, ytr)
    pr = m.predict_proba(B)[:, 1]
    thr = np.mean(ths)
    return dict(cv=f1_score(ytr, (oof >= thr).astype(int), average="macro"),
                test=f1_score(yte, (pr >= thr).astype(int), average="macro"),
                auc=roc_auc_score(yte, pr), thr=thr,
                dnnz=A.nnz, nfeat=A.shape[1])

def run_clean(C=4.0, scale_blocks=None, seed=42):
    """严格协议：vectorizer 折内拟合；阈值取 OOF（不碰测试集）"""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(ytr))
    for a, b in skf.split(np.arange(len(ytr)), ytr):
        v = TfidfVectorizer(**VEC_KW).fit([ctr[i] for i in a])
        Aa, Ab = build(v, [ctr[i] for i in a], [ctr[i] for i in b], Ztr[a], Ztr[b],
                       scale_blocks=scale_blocks)
        m = LogisticRegression(max_iter=3000, C=C).fit(Aa, ytr[a])
        oof[b] = m.predict_proba(Ab)[:, 1]
    thr = best_thr(ytr, oof)                      # 阈值来自 OOF，干净
    v = TfidfVectorizer(**VEC_KW).fit(ctr)
    A, B = build(v, ctr, cte, Ztr, Zte, scale_blocks=scale_blocks)
    m = LogisticRegression(max_iter=3000, C=C).fit(A, ytr)
    pr = m.predict_proba(B)[:, 1]
    return dict(cv=f1_score(ytr, (oof >= thr).astype(int), average="macro"),
                test=f1_score(yte, (pr >= thr).astype(int), average="macro"),
                auc=roc_auc_score(yte, pr), thr=thr,
                oof_auc=roc_auc_score(ytr, oof))

def run_concepts_only(C=4.0, model="lr"):
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(ytr))
    for a, b in skf.split(np.arange(len(ytr)), ytr):
        s = StandardScaler().fit(Ztr[a])
        if model == "lr":
            m = LogisticRegression(max_iter=3000, C=0.3).fit(s.transform(Ztr[a]), ytr[a])
        else:
            m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9,
                                  eval_metric="logloss", random_state=0, n_jobs=4).fit(Ztr[a], ytr[a])
        oof[b] = m.predict_proba(s.transform(Ztr[b]) if model == "lr" else Ztr[b])[:, 1]
    thr = best_thr(ytr, oof)
    s = StandardScaler().fit(Ztr)
    if model == "lr":
        m = LogisticRegression(max_iter=3000, C=0.3).fit(s.transform(Ztr), ytr)
        pr = m.predict_proba(s.transform(Zte))[:, 1]
    else:
        m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9,
                              eval_metric="logloss", random_state=0, n_jobs=4).fit(Ztr, ytr)
        pr = m.predict_proba(Zte)[:, 1]
    return dict(cv=f1_score(ytr, (oof >= thr).astype(int), average="macro"),
                test=f1_score(yte, (pr >= thr).astype(int), average="macro"),
                auc=roc_auc_score(yte, pr), thr=thr,
                oof_auc=roc_auc_score(ytr, oof))

# ---------------- 诊断 ----------------
def diag():
    print("=== A. 训练/测试重复检查 ===")
    str_tr = set(ctr); str_te = set(cte)
    print(f"  train 唯一文本 {len(str_tr)}/{len(ctr)}   test 唯一 {len(str_te)}/{len(cte)}")
    print(f"  test 文本出现在 train 中的条数: {sum(1 for c in cte if c in str_tr)}")
    # 前缀近似重复（前 20 字）
    pre_tr = {}
    for c in ctr: pre_tr.setdefault(c[:20], 0); pre_tr[c[:20]] += 1
    print(f"  test 前20字在 train 出现过的条数: {sum(1 for c in cte if c[:20] in pre_tr)}")

    print()
    print("=== B. 特征块范数失衡 ===")
    v = TfidfVectorizer(**VEC_KW).fit(ctr)
    A = v.transform(ctr)
    na = np.sqrt(A.multiply(A).sum(1)).A.ravel()
    nb = np.sqrt((Ztr ** 2).sum(1))
    print(f"  词法块行 L2 范数: 均值 {na.mean():.4f} (TfidfVectorizer norm='l2' → 恒为 1)")
    print(f"  概念块行 L2 范数: 均值 {nb.mean():.4f}")
    print(f"  → 概念块权重约为词法块的 {nb.mean()/na.mean():.2f} 倍（隐式加权）")

    print()
    print("=== C. C 敏感性（clean 协议）===")
    for C in (0.25, 1.0, 4.0, 16.0):
        r = run_clean(C=C)
        print(f"  C={C:5.2f}  CV={r['cv']:.4f}  test={r['test']:.4f}  AUC={r['auc']:.4f}  thr={r['thr']:.3f}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "clean"
    print(f"数据集 TOXICN  train={len(tr)} test={len(te)}  概念={Ztr.shape[1]}")
    if mode == "legacy":
        print("--- 原始实现（vectorizer 全训练集、阈值 in-sample）---")
        r = run_legacy(); print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    elif mode == "clean":
        print("--- 严格协议（vectorizer 折内、阈值 OOF）---")
        r = run_clean(); print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print("--- 特征块等范数 ---")
        r2 = run_clean(scale_blocks="equal")
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r2.items()})
        print("--- 仅概念 LR（参照）---")
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in run_concepts_only().items()})
        print("--- 仅概念 XGB（参照）---")
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in run_concepts_only(model="xgb").items()})
    else:
        diag()
