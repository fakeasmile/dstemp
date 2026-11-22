# -*- coding: utf-8 -*-
"""
build_external.py
构建跨数据集外部语料，用于概念编码器的预训练/热启动。

数据源（均在 data/raw/ 下）：
  COLD/train.json   32,157 条  {content, toxic(0/1), topic}
  CHSD/train.json   13,588 条  {content, toxic(0/1)}
  SWSR/SexComment.csv         {comment_text, label, category, target}

输出：
  data/processed/external_corpus.jsonl   {"text","label","source"}
  data/processed/external_lexicon.txt    统一词表（SWSR 人工辱称词表 + 频次过滤）

注意：原始数据存在 NaN / 非字符串 content（COLD 已确认），本脚本统一清洗，
      并逐项打印丢弃原因与数量。

用法：
  D:/miniconda3/envs/toxicnew/python.exe scripts/build_external.py
"""
import json, os
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
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

R = r"D:/dstemp"
RAW = os.path.join(R, "data/raw")
OUT = os.path.join(R, "data/processed")
os.makedirs(OUT, exist_ok=True)

# ---------------- 清洗工具 ----------------
def is_missing(v):
    """None / NaN / pd.NA 统一判为缺失。"""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False

def clean_text(v):
    if is_missing(v):
        return ""
    if not isinstance(v, str):
        return ""      # 非字符串一律丢弃，避免 "0.0" 这类脏数据
    return v.strip()

def clean_label(v):
    if is_missing(v):
        return None
    try:
        y = int(v)
    except Exception:
        return None
    return y if y in (0, 1) else None

# ---------------- ToxiCN 测试集（去重基准） ----------------
tox_test = json.load(open(os.path.join(RAW, "TOXICN/test.json"), encoding="utf-8"))
TEST_TEXTS = set(clean_text(r.get("content")) for r in tox_test)

rows = []
seen = {}                 # text -> label，用于去重与冲突检测
stat = {"empty": 0, "non_str": 0, "test_overlap": 0, "bad_label": 0,
        "dup_same": 0, "dup_conflict": 0, "kept": 0}
conflicts = []

def add(text, label, source):
    if text is not None and not isinstance(text, str) and not is_missing(text):
        stat["non_str"] += 1
        return
    t = clean_text(text)
    if len(t) < 2:
        stat["empty"] += 1
        return
    if t in TEST_TEXTS:
        stat["test_overlap"] += 1
        return
    y = clean_label(label)
    if y is None:
        stat["bad_label"] += 1
        return
    if t in seen:
        if seen[t] == y:
            stat["dup_same"] += 1
        else:
            stat["dup_conflict"] += 1
            if len(conflicts) < 5:
                conflicts.append((t[:40], seen[t], y, source))
        return
    seen[t] = y
    rows.append({"text": t, "label": y, "source": source})
    stat["kept"] += 1

def load_json_list(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

src_stat = {}

# ---------------- COLD ----------------
p = os.path.join(RAW, "COLD/train.json")
if os.path.exists(p):
    n0 = stat["kept"]
    data = load_json_list(p)
    bad_type = 0
    for r in tqdm(data, desc="COLD", leave=False):
        if not isinstance(r.get("content"), str):
            bad_type += 1
        add(r.get("content"), r.get("toxic", 0), "cold")
    src_stat["cold"] = dict(raw=len(data), kept=stat["kept"] - n0, non_str_content=bad_type)
else:
    src_stat["cold"] = "missing"

# ---------------- CHSD ----------------
p = os.path.join(RAW, "CHSD/train.json")
if os.path.exists(p):
    n0 = stat["kept"]
    data = load_json_list(p)
    for r in tqdm(data, desc="CHSD", leave=False):
        add(r.get("content"), r.get("toxic", 0), "chsd")
    src_stat["chsd"] = dict(raw=len(data), kept=stat["kept"] - n0)
else:
    src_stat["chsd"] = "missing"

# ---------------- SWSR ----------------
p = os.path.join(RAW, "SWSR/SexComment.csv")
if os.path.exists(p):
    n0 = stat["kept"]
    df = pd.read_csv(p)
    col_t = "comment_text" if "comment_text" in df.columns else df.columns[1]
    if "label" not in df.columns:
        raise ValueError(f"SWSR 缺少 label 列，实际列名: {list(df.columns)}")
    n_raw = len(df)
    df = df.dropna(subset=[col_t, "label"])
    for t, y in tqdm(zip(df[col_t], df["label"]), total=len(df), desc="SWSR", leave=False):
        add(t, y, "swsr")
    src_stat["swsr"] = dict(raw=n_raw, after_dropna=len(df), kept=stat["kept"] - n0)
else:
    src_stat["swsr"] = "missing"

# ---------------- 写出语料 ----------------
out_jsonl = os.path.join(OUT, "external_corpus.jsonl")
with open(out_jsonl, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

n_pos = sum(r["label"] for r in rows)
print("\n=== 外部语料 ===")
print(f"  保留 {len(rows)} 条   正例 {n_pos} ({n_pos/max(len(rows),1)*100:.1f}%)")
for k, v in src_stat.items():
    print(f"  {k:6s}: {v}")
print(f"  丢弃：空/过短 {stat['empty']}  非字符串 {stat['non_str']}  "
      f"与测试集重复 {stat['test_overlap']}  标签非法 {stat['bad_label']}")
print(f"  去重：完全重复 {stat['dup_same']}  同文异标 {stat['dup_conflict']}")
if conflicts:
    print("  同文异标示例（文本前40字, 已有标签, 冲突标签, 来源）：")
    for c in conflicts:
        print(f"    {c}")
print(f"  写出 {out_jsonl}")

# ---------------- 词表 ----------------
lex = []
p = os.path.join(RAW, "SWSR/SexHateLex.txt")
if os.path.exists(p):
    raw_terms = open(p, encoding="utf-8", errors="ignore").read().splitlines()
    seen_t = set()
    for t in raw_terms:
        t = t.strip()
        if t and t not in seen_t:
            seen_t.add(t)
            lex.append(t)
    print(f"\n=== 词表 ===\n  SWSR 人工辱称词 {len(lex)} 条（去重后）  示例: {lex[:8]}")
else:
    print("\n[警告] 未找到 SWSR/SexHateLex.txt，词表为空")

tox_train = json.load(open(os.path.join(RAW, "TOXICN/train.json"), encoding="utf-8"))
docs = [clean_text(r.get("content")) for r in tox_train]

counts = []
for t in tqdm(lex, desc="统计词频", leave=False):
    counts.append(sum(1 for d in docs if t in d))

lex_f = [t for t, c in zip(lex, counts) if c >= 5]
out_lex = os.path.join(OUT, "external_lexicon.txt")
with open(out_lex, "w", encoding="utf-8") as f:
    f.write("\n".join(lex_f))
print(f"  在 ToxiCN 训练集出现 >=5 条的词 {len(lex_f)} 个 → {out_lex}")
print(f"  过滤掉的冷门词 {len(lex) - len(lex_f)} 个")
