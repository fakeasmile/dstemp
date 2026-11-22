# -*- coding: utf-8 -*-
"""生成 v8 诊断三联图：信息饱和 / 隐式表达的系统性下移 / 误差相关性代价"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

ROOT = r"D:/dstemp"; OUT = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data/processed/TOXICN/v8_glm-4-9b-chat")
RAW = os.path.join(ROOT, "data/raw/TOXICN")

def load(s):
    raw = json.load(open(os.path.join(RAW, s + ".json"), encoding="utf-8"))
    o = json.load(open(os.path.join(PROC, "items_v8_glm-4-9b-chat_" + s + ".json"), encoding="utf-8"))
    it = [d["item_id"] for d in o["meta"]["item_definitions"]]
    ids = [f"TOXICN_{s}_{i:04d}" for i in range(len(raw))]
    X = np.array([[o["matrix"][k][j] for j in it] for k in ids]); y = np.array([r["toxic"] for r in raw])
    return X, y, it, raw

Xte, yte, items, raw_te = load("test")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

# ---- A: 信息饱和 ----
ax = axes[0]
labels = ["Best single\nconcept (B01)", "mean(P_yes)\n1-dim", "PCA-5", "LR\n26-dim",
          "MLP\n26-dim", "XGBoost\n26-dim", "XGBoost\n(oversized)"]
vals = [0.7788, 0.7810, 0.7837, 0.7902, 0.7954, 0.7988, 0.7937]
cols = ["#9AA5B1", "#9AA5B1", "#9AA5B1", "#5B8FF9", "#5B8FF9", "#2F6FD0", "#C6CDD5"]
b = ax.barh(range(len(vals)), vals, color=cols, height=0.62)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0.74, 0.81); ax.set_xlabel("Macro-F1 (5-fold CV)")
for i, v in enumerate(vals):
    ax.text(v + 0.0008, i, f"{v:.4f}", va="center", fontsize=7.5)
ax.axvline(0.7988, color="#2F6FD0", ls=":", lw=0.9)
ax.set_title("A. Effectively 1-dimensional:\n1-dim score 0.781 vs 26-dim 0.799",
             fontsize=9.5, loc="left")

# ---- B: 隐式 vs 显式 ----
ax = axes[1]
e = np.array([r["expression"] for r in raw_te])
m = yte == 1
deltas = []
for j, it in enumerate(items):
    e1 = Xte[m & (e == 1), j].mean(); e2 = Xte[m & (e == 2), j].mean()
    deltas.append((it, e2 - e1))
deltas.sort(key=lambda t: t[1])
names = [d[0] for d in deltas]; dvals = [d[1] for d in deltas]
cc = ["#D64545" if n in ("C02", "C01", "C03", "C04") else "#5B8FF9" for n in names]
ax.bar(range(len(dvals)), dvals, color=cc, width=0.72)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=6.5)
ax.axhline(0, color="#444", lw=0.9)
ax.set_ylabel("mean P(yes): implicit - explicit")
ax.set_ylim(-0.31, 0.02)
ax.set_title("B. Every concept drops on implicit hate\n(red = the 'expression' concepts meant to fix it)",
             fontsize=9.5, loc="left")
ax.text(0.02, 0.06, f"all 26 negative; mean {np.mean(dvals):+.3f}",
        transform=ax.transAxes, fontsize=7.5, color="#333")

# ---- C: 误差相关性代价 ----
ax = axes[2]
df = pd.read_csv(os.path.join(OUT, "v8_noise_curve.csv"))
for sig, col, mk in [(0.15, "#5B8FF9", "o"), (0.30, "#D64545", "s")]:
    d = df[df.sigma == sig].sort_values("rho")
    ax.errorbar(d.rho, d.f1, yerr=d.sd, marker=mk, ms=4.5, lw=1.6, color=col,
                capsize=2.5, elinewidth=0.8, label=f"$\\sigma$={sig}")
ax.axhline(0.7901, color="#2F6FD0", ls="--", lw=1.0, label="clean baseline")
ax.set_xlabel(r"error correlation $\rho$  (0 = independent, 1 = fully shared)")
ax.set_ylabel("Macro-F1 on test")
ax.set_ylim(0.60, 0.81)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
ax.set_title("C. Shared error hurts far more than\nindependent noise of the same size",
             fontsize=9.5, loc="left")

plt.tight_layout()
png = os.path.join(OUT, "fig_v8_diagnosis.png")
plt.savefig(png, dpi=200, bbox_inches="tight")
print("saved", png)
