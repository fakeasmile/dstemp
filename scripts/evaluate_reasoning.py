# -*- coding: utf-8 -*-
"""
evaluate_reasoning.py —— 对比 v8（孤立 yes/no）与 v9（推理优先）的概念测量质量
=============================================================================
检验的核心预测（可证伪）：
  1. 推理优先后，为隐式攻击设计的 C 族概念（C01–C04）与目标属性族（T01–T04）
     的区分度 a 应显著上升——因为模型现在有"分析"作依据，不再答非所问。
  2. 隐式文本子集（expression=2）上的 AUC / 错误率应改善。
  3. 整体下游 AUC / F1 应不降、最好上升。

用法：
  /d/miniconda3/envs/toxicnew/python.exe scripts/evaluate_reasoning.py \
      --v9_dir data/processed/TOXICN/v9_reasoning_glm-4-9b-chat

输出：控制台对比表；results/reasoning_comparison.csv
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

R = r"D:/dstemp"
V8_DIR = os.path.join(R, "data/processed/TOXICN/v8_glm-4-9b-chat")
RES = os.path.join(R, "results")
os.makedirs(RES, exist_ok=True)


def load_matrix(base_dir, split):
    obj = json.load(open(os.path.join(base_dir, f"items_v8_glm-4-9b-chat_{split}.json"),
                         encoding="utf-8")) if "v8" in base_dir else \
          json.load(open(os.path.join(base_dir, f"items_v9_glm-4-9b-chat_{split}.json"),
                         encoding="utf-8"))
    items = [d["item_id"] for d in obj["meta"]["item_definitions"]]
    raw = json.load(open(os.path.join(R, f"data/raw/TOXICN/{split}.json"), encoding="utf-8"))
    ids = [f"TOXICN_{split}_{i:04d}" for i in range(len(raw))]
    Z = np.array([[obj["matrix"][k].get(it, np.nan) for it in items] for k in ids], dtype=np.float64)
    y = np.array([int(r["toxic"]) for r in raw])
    expr = np.array([r.get("expression", 0) for r in raw])
    return Z, y, items, expr


def to_logit(Z):
    Z = np.clip(np.nan_to_num(Z, nan=0.5), 1e-4, 1 - 1e-4)
    return np.log(Z / (1 - Z))


def discrimination(Z, y):
    """每概念区分度：单概念 AUC 作为稳健版（也顺带报逻辑回归斜率）。"""
    aucs = []
    for j in range(Z.shape[1]):
        col = np.nan_to_num(Z[:, j], nan=0.5)
        if col.max() - col.min() < 1e-9:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(y, col))
    return np.array(aucs)


def cv_eval(Z, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(y)); ths = []
    for a, b in skf.split(Z, y):
        sc = StandardScaler().fit(Z[a])
        m = LogisticRegression(max_iter=3000, C=0.3).fit(sc.transform(Z[a]), y[a])
        oof[b] = m.predict_proba(sc.transform(Z[b]))[:, 1]
        cand = np.unique(np.quantile(m.predict_proba(sc.transform(Z[a]))[:, 1], np.linspace(.01, .99, 200)))
        bt, bf = 0.5, -1
        for t in cand:
            f = f1_score(y[a], (m.predict_proba(sc.transform(Z[a]))[:, 1] >= t).astype(int), average="macro")
            if f > bf:
                bf, bt = f, t
        ths.append(bt)
    thr = np.mean(ths)
    return f1_score(y, (oof >= thr).astype(int), average="macro"), roc_auc_score(y, oof), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9_dir", required=True)
    args = ap.parse_args()
    v9_dir = args.v9_dir

    Ztr8, ytr8, items8, _ = load_matrix(V8_DIR, "train")
    Zte8, yte8, _, expr8 = load_matrix(V8_DIR, "test")

    # v9 文件名前缀兼容 v9_reasoning_<model>/items_v9_<model>_<split>.json
    import glob
    v9_train = glob.glob(os.path.join(v9_dir, "items_v9_*_train.json"))
    v9_test = glob.glob(os.path.join(v9_dir, "items_v9_*_test.json"))
    if not v9_train or not v9_test:
        raise FileNotFoundError(
            f"v9 输出目录找不到 train/test JSON：{v9_dir}\n"
            f"请确认 run_items_v9_reasoning.py 已对 train 和 test 各跑一遍。"
        )
    obj_tr = json.load(open(v9_train[0], encoding="utf-8"))
    obj_te = json.load(open(v9_test[0], encoding="utf-8"))
    items9 = [d["item_id"] for d in obj_tr["meta"]["item_definitions"]]
    assert items8 == items9, "v8/v9 条目顺序不一致"

    raw_tr = json.load(open(os.path.join(R, "data/raw/TOXICN/train.json"), encoding="utf-8"))
    raw_te = json.load(open(os.path.join(R, "data/raw/TOXICN/test.json"), encoding="utf-8"))
    ids_tr = [f"TOXICN_train_{i:04d}" for i in range(len(raw_tr))]
    ids_te = [f"TOXICN_test_{i:04d}" for i in range(len(raw_te))]
    Ztr9 = np.array([[obj_tr["matrix"][k].get(it, np.nan) for it in items9] for k in ids_tr], dtype=np.float64)
    Zte9 = np.array([[obj_te["matrix"][k].get(it, np.nan) for it in items9] for k in ids_te], dtype=np.float64)

    # ---- 1. 整体下游 ----
    f1_8, auc_8, _ = cv_eval(Ztr8, ytr8)
    f1_9, auc_9, _ = cv_eval(Ztr9, ytr8)
    print("=" * 72)
    print("整体下游（5 折 CV，LR）")
    print(f"  v8 孤立 yes/no : CV-F1={f1_8:.4f}  AUC={auc_8:.4f}")
    print(f"  v9 推理优先     : CV-F1={f1_9:.4f}  AUC={auc_9:.4f}")
    print(f"  Δ               : F1 {f1_9-f1_8:+.4f}   AUC {auc_9-auc_8:+.4f}")

    # ---- 2. 概念区分度 ----
    a8 = discrimination(Ztr8, ytr8)
    a9 = discrimination(Ztr9, ytr8)
    df = pd.DataFrame({"item": items8, "auc_v8": a8, "auc_v9": a9})
    df["delta"] = df.auc_v9 - df.auc_v8
    print("\n" + "=" * 72)
    print("概念区分度（单概念 AUC）变化，聚焦 C/T 族")
    focus = [i for i in items8 if i.startswith(("C", "T")) or i == "E03"]
    print(df[df.item.isin(focus)].sort_values("delta").round(4).to_string(index=False))
    print(f"\n  全部概念 delta 均值 {df.delta.mean():+.4f}；C/T 族 delta 均值 "
          f"{df[df.item.isin(focus)].delta.mean():+.4f}")

    # ---- 3. 隐式子集 ----
    # 用全量 train 拟合，test 上按 expression 分组算 AUC
    def subset_auc(Ztr, Zte, ytr, yte, expr_te, mask):
        sc = StandardScaler().fit(Ztr)
        m = LogisticRegression(max_iter=3000, C=0.3).fit(sc.transform(Ztr), ytr)
        p = m.predict_proba(sc.transform(Zte))[:, 1]
        sub = mask
        if sub.sum() == 0 or len(set(yte[sub])) < 2:
            return float("nan"), int(sub.sum())
        return roc_auc_score(yte[sub], p[sub]), int(sub.sum())

    print("\n" + "=" * 72)
    print("隐式/显式文本子集上的 AUC（test，全量训练）")
    for tag, mask in [("显式 expr=1", expr8 == 1), ("隐式 expr=2", expr8 == 2),
                      ("转述 expr=3", expr8 == 3)]:
        a8s, n8 = subset_auc(Ztr8, Zte8, ytr8, yte8, expr8, mask)
        a9s, n9 = subset_auc(Ztr9, Zte9, ytr8, yte8, expr8, mask)
        print(f"  {tag:12s} n={n8:4d}  v8 AUC={a8s:.4f}  v9 AUC={a9s:.4f}  Δ={a9s-a8s:+.4f}")

    df.to_csv(os.path.join(RES, "reasoning_comparison.csv"), index=False)
    print("\n已写 results/reasoning_comparison.csv")


if __name__ == "__main__":
    main()
