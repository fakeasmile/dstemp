# -*- coding: utf-8 -*-
"""
irt_concepts.py —— 概念引出作为题反矩阵：IRT 校准 + Person-fit 误差识别
====================================================================
理论：LLM 对每条文本、每个概念输出 P(是)，构成 9600×26 的"题反矩阵"。
把它当作心理测量问题，拟合 2PL 项目反应模型：

    P_ij = σ( a_j · θ_i + c_j )          (c_j = -a_j·b_j)

其中 θ_i 是文本 i 的潜在"有害程度"，a_j 是概念 j 的区分度，b_j 是难度。

产出（全部用已有数据，不需要人工标注，不需要重跑 LLM）：
  1. θ̂ —— 校准后的潜在分数（相对 mean(ẑ) 的改进幅度是第一个可检验点）
  2. 题目参数 a_j / b_j —— 概念质量审计（哪些概念区分度高、哪些太容易/太难）
  3. Person-fit（infit/outfit）—— 无监督的测量误差检测器（回答你的 RQ1）
  4. 下游对照：mean(ẑ) vs θ̂ vs 全向量 vs 区分度加权 vs person-fit 降权

用法：
  /d/miniconda3/envs/toxicnew/python.exe scripts/irt_concepts.py

输出文件（results/）：
  irt_theta.npz            train/test 的 θ̂
  irt_item_params.csv      item_id, a, b, infit 相关统计
  irt_personfit.csv        每条文本的 outfit / infit（test）
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

R = r"D:/dstemp"
RAW = os.path.join(R, "data/raw/TOXICN")
PROC = os.path.join(R, "data/processed/TOXICN/v8_glm-4-9b-chat")
RES = os.path.join(R, "results")
os.makedirs(RES, exist_ok=True)

# ---------------- 载入 ----------------
def load(split):
    raw = json.load(open(os.path.join(RAW, f"{split}.json"), encoding="utf-8"))
    obj = json.load(open(os.path.join(PROC, f"items_v8_glm-4-9b-chat_{split}.json"), encoding="utf-8"))
    items = [d["item_id"] for d in obj["meta"]["item_definitions"]]
    ids = [f"TOXICN_{split}_{i:04d}" for i in range(len(raw))]
    Z = np.array([[obj["matrix"][k][it] for it in items] for k in ids], dtype=np.float64)
    y = np.array([int(r["toxic"]) for r in raw])
    return Z, y, items

Ztr, ytr, items = load("train")
Zte, yte, _ = load("test")
K = Ztr.shape[1]
print(f"[data] train {Ztr.shape} test {Zte.shape} K={K}")

# ---------------- logit 变换（题目反应在 logit 空间近似线性） ----------------
def to_logit(Z):
    Z = np.clip(Z, 1e-4, 1 - 1e-4)
    return np.log(Z / (1 - Z))

Ltr, Lte = to_logit(Ztr), to_logit(Zte)

# ---------------- θ：用 PCA 第一主成分作初始潜在分数 ----------------
sc = StandardScaler().fit(Ltr)
pca = PCA(n_components=1, random_state=0).fit(sc.transform(Ltr))
raw_tr = pca.transform(sc.transform(Ltr))[:, 0]
raw_te = pca.transform(sc.transform(Lte))[:, 0]
# 统一符号：让 θ 与 mean(logit) 正相关，保证语义一致
if np.corrcoef(raw_tr, Ltr.mean(1))[0, 1] < 0:
    raw_tr = -raw_tr
    raw_te = -raw_te
# 用 train 的均值/标准差同时标准化两片，保证尺度一致
mu, sd = float(raw_tr.mean()), float(raw_tr.std())
theta_tr = (raw_tr - mu) / sd
theta_te = (raw_te - mu) / sd

# ---------------- 题目参数：每列对 θ 做逻辑回归 ----------------
rows = []
P_hat_tr = np.zeros_like(Ztr)
P_hat_te = np.zeros_like(Zte)
for j in range(K):
    lr = LogisticRegression(max_iter=2000, C=1e6)   # 几乎无正则，纯拟合
    lr.fit(theta_tr.reshape(-1, 1), (Ztr[:, j] >= 0.5).astype(int))
    a = float(lr.coef_[0, 0])
    c = float(lr.intercept_[0])
    b = -c / a if abs(a) > 1e-9 else float("nan")
    # 用连续概率也能算拟合并计算期望
    P_hat_tr[:, j] = 1 / (1 + np.exp(-(a * theta_tr + c)))
    P_hat_te[:, j] = 1 / (1 + np.exp(-(a * theta_te + c)))
    rows.append(dict(item=items[j], a=a, b=b, c=c,
                     mean=float(Ztr[:, j].mean()), auc=roc_auc_score(ytr, Ztr[:, j])))

df_item = pd.DataFrame(rows)
df_item.to_csv(os.path.join(RES, "irt_item_params.csv"), index=False)
print("\n=== 题目参数（按区分度 a 排序）===")
print(df_item.sort_values("a", ascending=False).round(3).to_string(index=False))

# ---------------- Person-fit：标准化残差 + outfit/infit ----------------
def outfit(Z, P):
    # outfit = mean( (z - P)^2 / (P(1-P)) )，z 用连续概率，P 是模型期望
    v = P * (1 - P) + 1e-6
    return ((Z - P) ** 2 / v).mean(1)

def infit(Z, P):
    v = P * (1 - P) + 1e-6
    return ((Z - P) ** 2).sum(1) / v.sum(1)

outfit_tr, outfit_te = outfit(Ztr, P_hat_tr), outfit(Zte, P_hat_te)
infit_tr, infit_te = infit(Ztr, P_hat_tr), infit(Zte, P_hat_te)

df_pf = pd.DataFrame({"outfit": outfit_te, "infit": infit_te})
df_pf.to_csv(os.path.join(RES, "irt_personfit.csv"), index=False)
print(f"\n=== Person-fit（test）===")
print(f"  outfit 均值 {outfit_te.mean():.3f}  中位 {np.median(outfit_te):.3f}  >2 的文本 {int((outfit_te>2).sum())} 条")
print(f"  infit  均值 {infit_te.mean():.3f}  >1.5 的文本 {int((infit_te>1.5).sum())} 条")

# 保存 θ
np.savez(os.path.join(RES, "irt_theta.npz"), theta_tr=theta_tr, theta_te=theta_te)
print("\n已写 irt_theta.npz / irt_item_params.csv / irt_personfit.csv")

# ---------------- 下游对照：不同标量汇总 vs 全向量 ----------------
print("\n=== 下游对照（5 折 CV，阈值折内选择）===")
skf = StratifiedKFold(5, shuffle=True, random_state=42)

def best_thr(y, s):
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    bf, bt = -1.0, 0.5
    for t in cand:
        f = f1_score(y, (s >= t).astype(int), average="macro")
        if f > bf:
            bf, bt = f, t
    return bt

def cv_eval(X, tag):
    oof = np.zeros(len(ytr)); ths = []
    for a, b in skf.split(X, ytr):
        m = LogisticRegression(max_iter=3000, C=0.3).fit(X[a], ytr[a])
        oof[b] = m.predict_proba(X[b])[:, 1]
        ths.append(best_thr(ytr[a], m.predict_proba(X[a])[:, 1]))
    thr = np.mean(ths)
    f = f1_score(ytr, (oof >= thr).astype(int), average="macro")
    print(f"  {tag:36s} CV-F1={f:.4f}  AUC={roc_auc_score(ytr, oof):.4f}")

mean_tr = Ztr.mean(1, keepdims=True)
cv_eval(mean_tr, "mean(ẑ)")
cv_eval(theta_tr.reshape(-1, 1), "θ̂ (PCA+逻辑校准)")
a_all = df_item["a"].values
# Fisher 加权（区分度加权 logit 和）
fish_tr = (Ltr * a_all[None, :]).sum(1, keepdims=True) / (a_all ** 2).sum()
cv_eval(fish_tr, "区分度加权 Σa·logit(z)")
cv_eval(Ztr, "全向量 ẑ (26 维 LR)")

# Person-fit 降权：把 outfit 高的文本在训练时降权（用样本权重模拟"更可信的测量"）
w = np.clip(1.0 / (outfit_tr + 1e-6), 0, 1.0)
w = w / w.mean()
oof = np.zeros(len(ytr)); ths = []
for a, b in skf.split(Ztr, ytr):
    m = LogisticRegression(max_iter=3000, C=0.3).fit(Ztr[a], ytr[a], sample_weight=w[a])
    oof[b] = m.predict_proba(Ztr[b])[:, 1]
    ths.append(best_thr(ytr[a], m.predict_proba(Ztr[a])[:, 1]))
thr = np.mean(ths)
print(f"  {'person-fit 降权全向量':36s} CV-F1={f1_score(ytr,(oof>=thr).astype(int),average='macro'):.4f} "
      f" AUC={roc_auc_score(ytr, oof):.4f}")
