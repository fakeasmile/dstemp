# -*- coding: utf-8 -*-
"""
run_scbm.py —— 充分性导向的概念瓶颈（S-CBM）
=============================================
核心问题：把 PLM 放在「测量端」（z = h(x)）而不是「修正端」（z̃ = g(ẑ)），
能否在保持严格概念瓶颈（y 只读 z）的前提下把性能推上去。

变体（--variant）：
  d1  黑箱上限    encoder -> 1 logit                      （非 CBM，作上界）
  d2  严格瓶颈    encoder -> z(K,sigmoid) -> linear(1)     （无锚定）
  d3  锚定瓶颈    d2 + lam * BCE(z, ẑ)                     （锚到 LLM 测量）
  d4  锚定+词法   d3，线性层读 [z(K), lex(L)]               （词法概念也是概念）
  d5  后验纠错对照组：A 线性读出 / B 非线性读出 / C 残差纠错（三者都只是 ẑ 的函数）

输出：dev 选阈值、test 只评一次、test 概率存 results/prob_test_<variant>.npy

运行（Git Bash）：
  export KMP_DUPLICATE_LIB_OK=TRUE
  PY=/d/miniconda3/envs/toxicnew/python.exe
  $PY scripts/build_external.py
  $PY scripts/run_scbm.py --variant d5
  $PY scripts/run_scbm.py --variant d1 --model D:/models/chinese-macbert-base
  $PY scripts/run_scbm.py --variant d3 --model D:/models/chinese-macbert-base --lam 0.5
  $PY scripts/run_scbm.py --variant d4 --model D:/models/chinese-macbert-base --lam 0.5
  # d4 词法来源可选：--lex_source both|swsr|chi2（默认 both，chi2 词条上限 --lex_chi2 200）
  $PY scripts/run_scbm.py --variant d1 --model D:/models/chinese-macbert-base --external
"""
import argparse, json, os, time, random, re
import numpy as np

try:
    from tqdm import tqdm
except ImportError:                      # 兜底：环境里没有 tqdm 也不会崩
    class tqdm:
        def __init__(self, iterable=None, **kw):
            self._it = iterable if iterable is not None else []
        def __iter__(self):
            return iter(self._it)
        def __len__(self):
            return len(self._it)
        def set_postfix(self, *a, **k):
            pass
        def update(self, *a, **k):
            pass
        def close(self):
            pass

R = r"D:/dstemp"
RAW = os.path.join(R, "data/raw")
PROC = os.path.join(R, "data/processed")
RES = os.path.join(R, "results")
os.makedirs(RES, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["d1", "d2", "d3", "d4", "d5"])
ap.add_argument("--model", default=r"D:/models/chinese-macbert-base")
ap.add_argument("--external", action="store_true")
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--maxlen", type=int, default=160)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lex_topk", type=int, default=0, help="0=不截断")
ap.add_argument("--lex_source", default="both", choices=["swsr", "chi2", "both"],
                help="词法概念来源：SWSR 人工词表 / 数据驱动 chi2 / 并集")
ap.add_argument("--lex_chi2", type=int, default=200, help="chi2 词条上限")
ap.add_argument("--lex_ablate", action="store_true",
                help="d4 推理时把词法维置零：检验 z 是否真的在承担决策（不做则跳过）")
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed)

# ============================== 数据 ==============================
def load_split(split):
    raw = json.load(open(os.path.join(RAW, f"TOXICN/{split}.json"), encoding="utf-8"))
    o = json.load(open(os.path.join(
        PROC, f"TOXICN/v8_glm-4-9b-chat/items_v8_glm-4-9b-chat_{split}.json"), encoding="utf-8"))
    items = [d["item_id"] for d in o["meta"]["item_definitions"]]
    ids = [f"TOXICN_{split}_{i:04d}" for i in range(len(raw))]
    Z = np.array([[o["matrix"][k][it] for it in items] for k in ids], dtype=np.float32)
    X = [r["content"] if isinstance(r.get("content"), str) else "" for r in raw]
    y = np.array([int(r["toxic"]) for r in raw], dtype=np.int64)
    return X, y, Z, items

Xtr, ytr, Ztr, ITEMS = load_split("train")
Xte, yte, Zte, _ = load_split("test")
K = len(ITEMS); N = len(Xtr)

from sklearn.model_selection import train_test_split
itr, iva = train_test_split(np.arange(N), test_size=0.1, stratify=ytr, random_state=42)
print(f"[cfg] variant={args.variant} model={os.path.basename(args.model)} "
      f"lam={args.lam} external={args.external} epochs={args.epochs} bs={args.bs} lr={args.lr}")
print(f"[data] train={len(itr)} dev={len(iva)} test={len(Xte)} K={K}")
print(f"[data] 正例率 train={ytr[itr].mean():.3f} dev={ytr[iva].mean():.3f} test={yte.mean():.3f}")

# ============================== 指标 ==============================
from sklearn.metrics import f1_score, roc_auc_score

def best_thr(y, s):
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    bf, bt = -1.0, 0.5
    for t in cand:
        f = f1_score(y, (s >= t).astype(int), average="macro")
        if f > bf:
            bf, bt = f, t
    return bt

def report(tag, p_dev, p_te, extra="", fname=None):
    thr = best_thr(ytr[iva], p_dev)
    f_dev = f1_score(ytr[iva], (p_dev >= thr).astype(int), average="macro")
    f_te = f1_score(yte, (p_te >= thr).astype(int), average="macro")
    auc = roc_auc_score(yte, p_te)
    print("\n" + "=" * 72)
    print(f"  {tag}")
    print(f"  dev  MacroF1 {f_dev:.4f}  thr {thr:.3f}")
    print(f"  TEST MacroF1 {f_te:.4f}  AUC {auc:.4f}")
    if extra:
        print(f"  {extra}")
    print("=" * 72)
    np.save(os.path.join(RES, f"prob_test_{fname or args.variant}.npy"), p_te)

# ============================== d5 后验纠错 ==============================
# 三个对照，全部是 ẑ 的函数（因此都受信息处理不等式约束），差别只在函数类：
#   A  线性读出            LR(ẑ)
#   B  非线性读出          MLP(ẑ)
#   C  残差纠错            z̃ = σ(logit(ẑ) + Δ_MLP(ẑ))，再由线性层读出
# 若 C 不显著优于 A/B，则"后验纠错"这条路在信息论与实证上都站不住。
if args.variant == "d5":
    import torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    def to_logit(Z):
        Z = np.clip(np.asarray(Z, dtype=np.float64), 1e-4, 1 - 1e-4)
        return torch.tensor(np.log(Z / (1 - Z)), dtype=torch.float32)

    La, Ld, Lt = to_logit(Ztr[itr]), to_logit(Ztr[iva]), to_logit(Zte)
    ya_t = torch.tensor(ytr[itr], dtype=torch.float32)

    # ---- A：线性读出 ----
    sc = StandardScaler().fit(Ztr[itr])
    base = LogisticRegression(max_iter=3000, C=0.3).fit(sc.transform(Ztr[itr]), ytr[itr])
    report("d5-A  线性 ẑ -> LR",
           base.predict_proba(sc.transform(Ztr[iva]))[:, 1],
           base.predict_proba(sc.transform(Zte))[:, 1], fname="d5_linear")

    # ---- B / C 的训练框架 ----
    def train(bottleneck, epochs=300, lr=1e-3, patience=40):
        torch.manual_seed(args.seed)
        if bottleneck:
            net = nn.Sequential(nn.Linear(K, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, K))
            head = nn.Linear(K, 1)
            params = list(net.parameters()) + list(head.parameters())
        else:
            net = nn.Sequential(nn.Linear(K, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
            head = None
            params = list(net.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-3)

        def forward(L):
            if bottleneck:
                zt = torch.sigmoid(L + net(L))        # 残差在 logit 空间，z̃ 是"修正后的概念"
                return torch.sigmoid(head(zt).squeeze(-1)), zt
            return torch.sigmoid(net(L).squeeze(-1)), None

        best = (-1.0, None, None, 0.5)
        since = 0
        for ep in range(epochs):
            net.train()
            if head is not None:
                head.train()
            p, _ = forward(La)
            loss = nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), ya_t)
            opt.zero_grad(); loss.backward(); opt.step()

            if (ep + 1) % 5 == 0:
                net.eval()
                if head is not None:
                    head.eval()
                with torch.no_grad():
                    pd, _ = forward(Ld)
                pd = pd.numpy()
                thr = best_thr(ytr[iva], pd)
                f = f1_score(ytr[iva], (pd >= thr).astype(int), average="macro")
                if f > best[0]:
                    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    hd = ({k: v.detach().clone() for k, v in head.state_dict().items()}
                          if head is not None else None)
                    best = (f, sd, hd, thr)
                    since = 0
                else:
                    since += 5
                    if since >= patience:
                        break

        net.load_state_dict(best[1])
        if head is not None:
            head.load_state_dict(best[2])
        net.eval()
        if head is not None:
            head.eval()
        with torch.no_grad():
            pd, _ = forward(Ld)
            pt, zt = forward(Lt)
        return pd.numpy(), pt.numpy(), (zt.numpy() if zt is not None else None), best[0]

    # ---- B：非线性读出（无瓶颈） ----
    pd_b, pt_b, _, _ = train(bottleneck=False)
    report("d5-B  非线性 ẑ -> MLP（无瓶颈）", pd_b, pt_b, fname="d5_mlp")

    # ---- C：残差纠错 ----
    pd_c, pt_c, zt_c, devf = train(bottleneck=True)
    mag = float(np.mean(np.abs(zt_c - Zte))) if zt_c is not None else float("nan")
    report("d5-C  残差纠错 z̃=σ(logit(ẑ)+Δ(ẑ))", pd_c, pt_c,
           extra=f"平均修改幅度 |z̃-ẑ|={mag:.4f}", fname="d5_corrected")
    raise SystemExit(0)

# ============================== d1–d4 ==============================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

dv = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
print(f"[env] device={dv}")

tok = AutoTokenizer.from_pretrained(args.model)

# 词法概念（d4）
LEX = []
if args.variant == "d4":
    terms = []

    # 来源 1：SWSR 人工辱称词表（外部知识，覆盖性别/性向向）
    if args.lex_source in ("swsr", "both"):
        lp = os.path.join(PROC, "external_lexicon.txt")
        if not os.path.exists(lp):
            raise FileNotFoundError("缺少 external_lexicon.txt，先运行 scripts/build_external.py")
        sw = [t.strip() for t in open(lp, encoding="utf-8").read().splitlines() if t.strip()]
        terms += sw
        print(f"[lex] SWSR 词表 {len(sw)} 个")

    # 来源 2：训练折内的数据驱动词条（覆盖 SWSR 词表没有的种族/地域辱称）
    if args.lex_source in ("chi2", "both"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import chi2 as chi2_score
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=5)
        A = vec.fit_transform([Xtr[i] for i in itr])      # 只用训练折，dev 不参与
        st, _ = chi2_score(A, ytr[itr])
        names = vec.get_feature_names_out()
        picked = []
        for j in np.argsort(-st):
            t = names[j]
            if re.search(r"[^\w\u4e00-\u9fff]", t):       # 剔除含标点/空格的脏词条
                continue
            picked.append(t)
            if len(picked) >= args.lex_chi2:
                break
        terms += picked
        print(f"[lex] chi2 词条 {len(picked)} 个（仅训练折选择）")

    seen = set()
    for t in terms:
        if t not in seen:
            seen.add(t); LEX.append(t)
    if args.lex_topk and args.lex_topk < len(LEX):
        # 注意：Xtr 是 list，不能用 numpy 数组直接索引
        freq = np.array([sum(1 for d in [Xtr[i] for i in itr] if t in d) for t in LEX])
        LEX = [LEX[i] for i in np.argsort(-freq)[:args.lex_topk]]
    print(f"[lex] 合计词法概念 {len(LEX)} 个   示例: {LEX[:10]}")
NL = len(LEX)

def lex_block(texts):
    M = np.zeros((len(texts), NL), dtype=np.float32)
    if NL:
        for i, t in enumerate(tqdm(texts, desc="lex block", leave=False)):
            for j, w in enumerate(LEX):
                if w in t:
                    M[i, j] = 1.0
    return M

Ltr, Lte = lex_block(Xtr), lex_block(Xte)

class DS(Dataset):
    def __init__(self, texts, labels, zhat, lex):
        self.enc = tok(texts, truncation=True, max_length=args.maxlen, padding=False)
        self.y, self.z, self.lex = labels, zhat, lex
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}      # 保留 python list，交给 tok.pad
        d["y"] = torch.tensor(int(self.y[i]))
        d["zhat"] = torch.tensor(self.z[i])
        d["lex"] = torch.tensor(self.lex[i])
        return d

TXT_KEYS = ("input_ids", "attention_mask", "token_type_ids")

def collate(items):
    out = tok.pad([{k: it[k] for k in TXT_KEYS if k in it} for it in items],
                  padding=True, return_tensors="pt")
    out["y"] = torch.stack([it["y"] for it in items])
    out["zhat"] = torch.stack([it["zhat"] for it in items])
    out["lex"] = torch.stack([it["lex"] for it in items])
    return out

def txt_of(batch):
    return {k: v.to(dv) for k, v in batch.items() if k in TXT_KEYS}

class Net(nn.Module):
    def __init__(self, path, k, n_lex):
        super().__init__()
        self.enc = AutoModel.from_pretrained(path)
        h = self.enc.config.hidden_size
        self.k, self.n_lex = k, n_lex
        if k:
            self.to_z = nn.Linear(h, k)
            self.clf = nn.Linear(k + n_lex, 1)
        else:
            self.head = nn.Linear(h, 1)
    def forward(self, txt):
        h = self.enc(**txt).last_hidden_state[:, 0]
        if self.k:
            return torch.sigmoid(self.to_z(h))
        return self.head(h).squeeze(-1)      # 标量 logit，形状 (B,)

def make_loader(texts, labels, zhat, lex, shuffle):
    return DataLoader(DS(texts, labels, zhat, lex), batch_size=args.bs,
                      shuffle=shuffle, collate_fn=collate, drop_last=False)

dl_tr = make_loader([Xtr[i] for i in itr], ytr[itr], Ztr[itr], Ltr[itr], True)
dl_va = make_loader([Xtr[i] for i in iva], ytr[iva], Ztr[iva], Ltr[iva], False)
dl_te = make_loader(Xte, yte, Zte, Lte, False)

def logits_of(model, batch, return_z=False, ablate_lex=False):
    """一次前向，同时拿到 logit 与 z，避免 dropout 造成两次采样不一致。
    ablate_lex=True 时把词法维整体置零，用于检验决策是否依赖 z。"""
    txt = txt_of(batch)
    if args.variant == "d1":
        logit = model(txt)
        return (logit, None) if return_z else logit
    z = model(txt)
    if NL:
        lf = batch["lex"].to(dv)
        if ablate_lex:
            lf = torch.zeros_like(lf)
        feat = torch.cat([z, lf], dim=1)
    else:
        feat = z
    logit = model.clf(feat).squeeze(-1)
    return (logit, z) if return_z else logit

def infer(model, dl, desc="infer", ablate_lex=False):
    model.eval(); P = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=desc, leave=False):
            P.append(torch.sigmoid(logits_of(model, batch, ablate_lex=ablate_lex)).cpu().numpy())
    return np.concatenate(P)

model = Net(args.model, 0 if args.variant == "d1" else K, NL).to(dv)
print(f"[net] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
      f"bottleneck={'no' if args.variant=='d1' else f'{K}+{NL}'}")

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total = len(dl_tr) * args.epochs
sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

# 外部语料热启动
if args.external:
    cp = os.path.join(PROC, "external_corpus.jsonl")
    if not os.path.exists(cp):
        raise FileNotFoundError("缺少 external_corpus.jsonl，先运行 scripts/build_external.py")
    ext = [json.loads(l) for l in open(cp, encoding="utf-8")]
    print(f"[ext] 外部语料 {len(ext)} 条，预训练 1 epoch")
    dl_ext = make_loader([e["text"] for e in ext], np.array([e["label"] for e in ext]),
                         np.zeros((len(ext), K), dtype=np.float32),
                         np.zeros((len(ext), NL), dtype=np.float32), True)
    model.train(); t0 = time.time()
    pbar = tqdm(dl_ext, desc="external", leave=False)
    for batch in pbar:
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_of(model, batch), batch["y"].to(dv).float())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"[ext] loss {loss.item():.4f} ({time.time()-t0:.0f}s)")

best = (-1.0, None)
ZCHAT_CLIP = (1e-4, 1 - 1e-4)
for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot = 0.0
    pbar = tqdm(dl_tr, desc=f"ft ep{ep+1}/{args.epochs}", leave=False)
    for batch in pbar:
        y = batch["y"].to(dv).float()
        if args.variant in ("d3", "d4"):
            logit, z = logits_of(model, batch, return_z=True)
        else:
            logit, z = logits_of(model, batch), None
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        if z is not None:
            tgt = batch["zhat"].to(dv).clamp(*ZCHAT_CLIP)
            loss = loss + args.lam * nn.functional.binary_cross_entropy(z, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tot += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    pv = infer(model, dl_va)
    thr = best_thr(ytr[iva], pv)
    f = f1_score(ytr[iva], (pv >= thr).astype(int), average="macro")
    print(f"[ft] epoch {ep+1} loss {tot/len(dl_tr):.4f} devF1 {f:.4f} ({time.time()-t0:.0f}s)")
    if f > best[0]:
        best = (f, {k: v.detach().clone() for k, v in model.state_dict().items()})

model.load_state_dict(best[1])
pv, pt = infer(model, dl_va), infer(model, dl_te)

# 概念保真度：d2/d3/d4 的 z 都是 26 维，与 ẑ 逐维对齐才有意义
extra = ""
if args.variant in ("d2", "d3", "d4"):
    Zs = []
    model.eval()
    with torch.no_grad():
        for batch in dl_te:
            Zs.append(model(txt_of(batch)).cpu().numpy())
    Zout = np.concatenate(Zs)
    fid = np.mean([np.corrcoef(Zout[:, i], Zte[:, i])[0, 1] for i in range(K)])
    extra = f"概念保真度 mean corr(z, ẑ) = {fid:.3f}"

report(f"{args.variant}  {os.path.basename(args.model)}  ext={args.external}  lam={args.lam}",
       pv, pt, extra)

# ---- d4 专用诊断：决策到底靠 z 还是靠词法 ----
if args.variant == "d4" and NL:
    w = model.clf.weight.detach()[0]
    nz, nl_ = float(w[:K].norm()), float(w[K:].norm())
    print(f"[diag] ||w_z||={nz:.4f}  ||w_lex||={nl_:.4f}  "
          f"词法权重占比 {nl_/(nz+nl_+1e-9)*100:.1f}%")

    if args.lex_ablate:
        pv0 = infer(model, dl_va, desc="infer(lex=0)", ablate_lex=True)
        pt0 = infer(model, dl_te, desc="infer(lex=0)", ablate_lex=True)
        report("d4 词法消融（推理时 lex=0，只留 z）", pv0, pt0, fname="d4_no_lex")
