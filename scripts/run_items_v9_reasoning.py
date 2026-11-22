# -*- coding: utf-8 -*-
"""
run_items_v9_reasoning.py —— 推理优先（reasoning-first）概念测量
================================================================
背景：v8 用"孤立的一 token yes/no"问每个概念。实测发现：为隐式攻击设计的
概念（C 族反讽/委婉、T 族目标属性）区分度最低、难度最高——因为模型在
还没形成"这是讽刺/影射"的判断时就被迫给出 yes/no。

本脚本在概念问询前，先让模型对整段文本自由生成一句"隐含攻击分析"，
再把该分析插入每个概念问句。这是对"测量协议"的修改，仍产出 26 维概念
概率向量（严格 CBM），仅改变测量方式。

与 v8 的关系：
- 完全复用 v8 的 vLLM 加载、受限解码、首 token 概率提取、断点续跑。
- 差异只有：每个样本多一次"分析"生成；概念问句多一个【分析】段。
- 输出目录独立：data/processed/TOXICN/v9_reasoning_<model>/，不覆盖 v8。

用法（Git Bash）：
  /d/miniconda3/envs/toxicnew/python.exe scripts/run_items_v9_reasoning.py \
    --model_path /root/autodl-tmp/glm-4-9b-chat \
    --data_path data/raw/TOXICN --split train --limit 8
  # 全量：去掉 --limit

产出（与 v8 同构，方便下游脚本最小改动读取）：
  items_v9_<model>_<split>.json        {meta, matrix, invalid}
  items_v9_<model>_<split>_matrix.csv
  items_v9_<model>_<split>_reasoning.jsonl   每样本的分析文本（审计用）
"""
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REASONING_TEMPLATE = """你是中文文本现象判定助手。下面是一段社交平台文本。
请判断它是否含有“隐含”的攻击性内容，包括但不限于：反讽/阴阳怪气、影射、
谐音/拆字/变体替换、委婉贬损、借他人之口的攻击、只在特定语境下才成立的侮辱。
注意：字面上看起来无害、甚至像夸奖的话，可能实际是攻击。

文本：{text}

请用一句话（不超过 80 字）指出这段文本可能隐含了什么攻击，以及你的判断依据。
分析："""

ITEM_TEMPLATE_REASONING = """你是严格的中文文本现象判定助手。请根据文本判断指定概念是否出现。
只依据文本中可观察到的内容判断，不推测作者未表达的意图，不补充外部事实。
对内容现象概念，文本中被引用或转述的表达也按其实际内容判断；对发言者立场概念，
只判断能够归属于当前发言者的立场，不能因为文本没有反对就推定其支持。

文本：{text}

【分析】{reasoning}

概念：{concept_name}
判定问题：{question}
概念定义：{definition}
边界说明：{boundary}

只回答“是”或“否”。回答："""

MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}

REQUIRED_COLUMNS = [
    "条目id", "构念id", "构念名", "区块", "题式", "方向", "条目问句",
    "概念定义", "边界说明", "优先级",
]
ALL_PRIORITIES = {"核心", "目标属性", "储备"}

DEFAULT_ITEMS_CANDIDATES = [
    PROJECT_ROOT / "data" / "concepts" / "cbm_concept_set_v8.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v8_items(csv_path: Path, include_reserve: bool = False) -> List[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 v8 词典：{csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(x or "").lstrip("\ufeff\x00").strip() for x in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"v8 词典缺少必需列：{missing}")
        all_items = []
        for line_no, row in enumerate(reader, start=2):
            item = {k: (row.get(k) or "").strip() for k in REQUIRED_COLUMNS}
            if not item["条目id"]:
                raise ValueError(f"词典第 {line_no} 行条目id为空")
            for col in ("构念名", "条目问句", "概念定义", "边界说明", "优先级"):
                if not item[col]:
                    raise ValueError(f"词典第 {line_no} 行 {col} 为空（条目 {item['条目id']}）")
            if item["优先级"] not in ALL_PRIORITIES:
                raise ValueError(f"词典第 {line_no} 行优先级非法：{item['优先级']}")
            all_items.append(item)
    if not all_items:
        raise ValueError(f"词典中没有概念：{csv_path}")
    selected = [x for x in all_items if include_reserve or x["优先级"] != "储备"]
    return selected


def normalize_sample(item, source_name, split, idx):
    if not isinstance(item, dict):
        raise TypeError(f"数据第 {idx + 1} 条不是 JSON object")
    if "content" not in item or item["content"] is None:
        raise ValueError(f"数据第 {idx + 1} 条缺少 content")
    content = str(item["content"])
    if not content.strip():
        raise ValueError(f"数据第 {idx + 1} 条 content 为空")
    sid = item.get("id")
    if sid is None or str(sid).strip() == "":
        sid = f"{source_name}_{split}_{idx:04d}"
    return {
        "id": str(sid),
        "source": str(item.get("source") or source_name),
        "split": split,
        "content": content,
        "toxic": item.get("toxic"),
        "topic": item.get("topic"),
        "toxic_type": item.get("toxic_type"),
        "expression": item.get("expression"),
        "target": item.get("target"),
    }


def load_dataset(data_path: Path, split: str):
    if not data_path.exists():
        raise FileNotFoundError(f"data_path 不存在：{data_path}")
    if data_path.is_file():
        json_path = data_path
        dataset_tag = data_path.parent.name or data_path.stem
        actual_split = data_path.stem
        files = [data_path.name]
    else:
        json_path = data_path / f"{split}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"{data_path} 下找不到 {split}.json")
        dataset_tag = data_path.name
        actual_split = split
        files = [json_path.name]
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"数据文件顶层必须是 JSON list：{json_path}")
    samples = [normalize_sample(x, dataset_tag, actual_split, i) for i, x in enumerate(raw)]
    ids = [x["id"] for x in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"数据样本 id 不唯一")
    return samples, files, dataset_tag, json_path


def load_vllm_model(model_path, gpu_memory_utilization, max_model_len,
                    max_num_seqs, max_num_batched_tokens):
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json：{model_path}")
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, padding_side="right", local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading local vLLM model from: {model_path}", flush=True)
    llm = LLM(
        model=str(model_path), trust_remote_code=True, dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True,
        max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens, enforce_eager=True, max_logprobs=50,
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer) -> dict:
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


def get_candidate_ids(words, tokenizer):
    ids = []
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"verbalizer {word!r} 不是单 token：{token_ids}")
        ids.extend(token_ids)
    return list(dict.fromkeys(ids))


def make_mask_processor(allowed_ids):
    allowed = sorted(set(allowed_ids))

    def _processor(*args, **kwargs):
        import torch
        logits = kwargs.get("logits")
        if logits is None:
            logits = next(a for a in args if isinstance(a, torch.Tensor))
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        return masked

    return _processor


def build_restricted_sampling(allowed_ids):
    from vllm import SamplingParams
    allowed = sorted(set(allowed_ids))
    for mode, extra in [
        ("logits_processors", {"logits_processors": [make_mask_processor(allowed)]}),
        ("allowed_token_ids", {"allowed_token_ids": allowed}),
        ("无约束回退", {}),
    ]:
        try:
            return SamplingParams(max_tokens=1, temperature=0, logprobs=50, **extra), mode
        except TypeError:
            continue
    raise RuntimeError("无法建立受限解码 SamplingParams")


def render_prompts(tokenizer, prompt_texts, chat_kwargs, model_name):
    manual_suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    rendered = []
    for text in prompt_texts:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        if manual_suffix is not None:
            prompt += manual_suffix
        elif not prompt.endswith("\n"):
            prompt += "\n"
        rendered.append(prompt)
    return rendered


def _logprob_value(value):
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def get_probability_result(request_output, tokenizer, yes_ids, no_ids):
    if not request_output.outputs:
        raise RuntimeError("vLLM 返回空 outputs")
    completion = request_output.outputs[0]
    if not completion.token_ids:
        raise RuntimeError("模型没有生成首 token")
    if not completion.logprobs or not completion.logprobs[0]:
        raise RuntimeError("没有首 token logprobs")
    token_id = int(completion.token_ids[0])
    first_token = ("是" if token_id in yes_ids else
                   "否" if token_id in no_ids else
                   tokenizer.decode([token_id], skip_special_tokens=False))
    logprob_map = completion.logprobs[0]

    def mass_for(ids):
        total = 0.0
        for tid in ids:
            entry = logprob_map.get(tid)
            if entry is not None:
                total += math.exp(_logprob_value(entry))
        return total

    yes_mass = mass_for(yes_ids)
    no_mass = mass_for(no_ids)
    cand_mass = yes_mass + no_mass
    p_yes = None if (cand_mass <= 0.0 or not math.isfinite(cand_mass)) else yes_mass / cand_mass
    return {
        "first_token": first_token,
        "first_token_id": token_id,
        "p_yes": None if p_yes is None else round(float(p_yes), 6),
        "cand_mass": round(float(cand_mass), 6),
    }


def make_reasoning_prompt(sample):
    return REASONING_TEMPLATE.format(text=sample["content"])


def make_concept_prompt(sample, item, reasoning):
    return ITEM_TEMPLATE_REASONING.format(
        text=sample["content"],
        reasoning=reasoning,
        concept_name=item["构念名"],
        question=item["条目问句"],
        definition=item["概念定义"],
        boundary=item["边界说明"],
    )


def read_checkpoint(path, config, dataset_ids, item_ids, overwrite):
    if overwrite and path.exists():
        path.unlink()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "meta", "config": config}, ensure_ascii=False) + "\n")
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    parsed, valid_end = [], 0
    for i, line in enumerate(lines):
        try:
            parsed.append(json.loads(line.decode("utf-8")))
            valid_end += len(line)
        except Exception:
            if i == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                with path.open("r+b") as f:
                    f.truncate(valid_end)
                break
            raise
    if not parsed or parsed[0].get("record_type") != "meta":
        raise ValueError(f"checkpoint 缺少 meta 首行：{path}")
    if parsed[0].get("config") != config:
        raise ValueError("checkpoint 配置不一致；请用 --overwrite 重跑")
    results = {}
    item_id_set = set(item_ids)
    for obj in parsed[1:]:
        if obj.get("record_type") != "sample":
            raise ValueError("checkpoint 记录类型非法")
        sid = obj.get("id")
        if sid not in dataset_ids or sid in results:
            raise ValueError(f"checkpoint 样本异常：{sid}")
        for k in ("p_yes", "first_token", "cand_mass", "first_token_id"):
            if set(obj.get(k, {})) != item_id_set:
                raise ValueError(f"checkpoint 样本 {sid} 条目不完整")
        results[sid] = obj
    return results


def atomic_write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_csv(path, header, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description="ToxiCN 推理优先概念测量（reasoning-first）")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--data_path", required=True, type=Path)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--items_csv", type=Path, default=DEFAULT_ITEMS_CANDIDATES[0])
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--max_num_batched_tokens", type=int, default=16384)
    p.add_argument("--batch_samples", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reasoning_max_tokens", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    items_path = args.items_csv.expanduser().resolve()

    dataset, input_files, dataset_tag, json_path = load_dataset(data_path, args.split)
    if args.limit is not None:
        dataset = dataset[:args.limit]
        print(f"技术冒烟限制：{len(dataset)} 条")

    items = load_v8_items(items_path, include_reserve=False)
    item_ids = [x["条目id"] for x in items]

    model_name = model_path.name
    split_tag = json_path.stem
    suffix = split_tag + (f"_limit{args.limit}" if args.limit is not None else "")
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  PROJECT_ROOT / "data" / "processed" / dataset_tag / f"v9_reasoning_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = output_dir / f"items_v9_{model_name}_{suffix}.json"
    matrix_out = output_dir / f"items_v9_{model_name}_{suffix}_matrix.csv"
    reasoning_out = output_dir / f"items_v9_{model_name}_{suffix}_reasoning.jsonl"
    checkpoint = output_dir / f"items_v9_{model_name}_{suffix}.checkpoint.jsonl"

    if args.overwrite:
        for p in (json_out, matrix_out, reasoning_out):
            if p.exists():
                p.unlink()

    script_path = Path(__file__).resolve()
    config = {
        "model_path": str(model_path), "model_name": model_name,
        "input_path": str(json_path), "input_sha256": sha256_file(json_path),
        "split": split_tag, "n_samples": len(dataset),
        "items_path": str(items_path), "items_sha256": sha256_file(items_path),
        "script_path": str(script_path), "script_sha256": sha256_file(script_path),
        "selected_item_ids": item_ids,
        "prompt_version": "v9.0-reasoning-first",
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "temperature": 0,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    results = read_checkpoint(checkpoint, config, {x["id"] for x in dataset}, item_ids, args.overwrite)
    todo = [x for x in dataset if x["id"] not in results]

    print("=" * 72)
    print(f"数据集 {dataset_tag}/{split_tag} 样本 {len(dataset)} 待处理 {len(todo)}")
    print(f"测量条目 {len(items)}；推理优先协议 v9；每样本 = 1 次分析 + {len(items)} 次概念问询")
    print(f"输出目录 {output_dir}")
    print("=" * 72, flush=True)

    if todo:
        tokenizer, llm = load_vllm_model(
            model_path, args.gpu_memory_utilization, args.max_model_len,
            args.max_num_seqs, args.max_num_batched_tokens
        )
        chat_kwargs = detect_chat_kwargs(tokenizer)
        yes_ids = get_candidate_ids(["是"], tokenizer)
        no_ids = get_candidate_ids(["否"], tokenizer)
        if set(yes_ids) & set(no_ids):
            raise ValueError("是/否 verbalizer token 重叠")

        from vllm import SamplingParams
        reasoning_params = SamplingParams(
            max_tokens=args.reasoning_max_tokens, temperature=0, stop=["\n"]
        )
        sampling_params, restrict_mode = build_restricted_sampling(yes_ids + no_ids)

        progress = tqdm(total=len(dataset), initial=len(results), desc="reasoning-first 测量")
        try:
            for start in range(0, len(todo), args.batch_samples):
                batch = todo[start:start + args.batch_samples]

                # ---- 阶段 1：自由生成"隐含攻击分析" ----
                reasoning_prompts = [make_reasoning_prompt(s) for s in batch]
                rendered_r = render_prompts(tokenizer, reasoning_prompts, chat_kwargs, model_name)
                outputs_r = llm.generate(rendered_r, reasoning_params, use_tqdm=False)
                reasonings = []
                for s, o in zip(batch, outputs_r):
                    txt = (o.outputs[0].text if o.outputs else "").strip()
                    reasonings.append(txt or "（无法分析）")

                # ---- 阶段 2：带分析的概念问询（受限解码） ----
                prompt_texts, request_keys = [], []
                for sample, reasoning in zip(batch, reasonings):
                    for item in items:
                        prompt_texts.append(make_concept_prompt(sample, item, reasoning))
                        request_keys.append((sample["id"], item["条目id"]))
                rendered_c = render_prompts(tokenizer, prompt_texts, chat_kwargs, model_name)
                outputs_c = llm.generate(rendered_c, sampling_params, use_tqdm=False)
                if len(outputs_c) != len(request_keys):
                    raise RuntimeError("概念问询输出数与请求数不一致")

                batch_records = {}
                for sample, reasoning in zip(batch, reasonings):
                    batch_records[sample["id"]] = {
                        "record_type": "sample", "id": sample["id"],
                        "reasoning": reasoning,
                        "p_yes": {}, "first_token": {}, "cand_mass": {}, "first_token_id": {},
                        "restricted_decoding": restrict_mode,
                    }
                for (sid, iid), output in zip(request_keys, outputs_c):
                    val = get_probability_result(output, tokenizer, yes_ids, no_ids)
                    rec = batch_records[sid]
                    rec["p_yes"][iid] = val["p_yes"]
                    rec["first_token"][iid] = val["first_token"]
                    rec["cand_mass"][iid] = val["cand_mass"]
                    rec["first_token_id"][iid] = val["first_token_id"]

                with checkpoint.open("a", encoding="utf-8") as f:
                    for sample in batch:
                        rec = batch_records[sample["id"]]
                        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                        results[sample["id"]] = rec
                    f.flush()
                    os.fsync(f.fileno())
                progress.update(len(batch))
        finally:
            progress.close()

    missing = [x["id"] for x in dataset if x["id"] not in results]
    if missing:
        raise RuntimeError(f"仍缺少 {len(missing)} 个样本")

    # ---- 汇总输出 ----
    matrix = {}
    reasoning_lines = []
    invalid = []
    for sample in dataset:
        sid = sample["id"]
        rec = results[sid]
        matrix[sid] = rec["p_yes"]
        reasoning_lines.append(json.dumps(
            {"id": sid, "reasoning": rec.get("reasoning", "")}, ensure_ascii=False
        ))
        for item in items:
            iid = item["条目id"]
            if rec["p_yes"].get(iid) is None:
                invalid.append({"id": sid, "item_id": iid})

    meta = {
        "script": "run_items_v9_reasoning.py",
        "script_version": "9.0.0",
        "prompt_version": config["prompt_version"],
        "config_sha256": config_hash,
        "config": config,
        "data_files": input_files,
        "n_samples": len(dataset),
        "n_items": len(items),
        "item_definitions": [
            {"item_id": x["条目id"], "construct_id": x["构念id"], "construct_name": x["构念名"],
             "priority": x["优先级"], "direction": x["方向"], "question": x["条目问句"],
             "definition": x["概念定义"], "boundary": x["边界说明"]}
            for x in items
        ],
        "restricted_decoding": restrict_mode if todo else results.get(dataset[0]["id"], {}).get("restricted_decoding", "unknown"),
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "n_invalid": len(invalid),
    }
    atomic_write_json(json_out, {"meta": meta, "matrix": matrix, "invalid": invalid})

    with reasoning_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(reasoning_lines) + "\n")

    matrix_rows = []
    for sample in dataset:
        sid = sample["id"]
        matrix_rows.append([
            sid, sample["toxic"], sample["content"],
            *[matrix[sid][iid] if matrix[sid][iid] is not None else "" for iid in item_ids],
        ])
    atomic_write_csv(matrix_out, ["id", "toxic", "content", *item_ids], matrix_rows)

    print("=" * 72)
    print(f"完成：样本 {len(dataset)} × 条目 {len(items)}")
    print(f"无效 p_yes：{len(invalid)}")
    print(f"JSON：{json_out}")
    print(f"分析文本：{reasoning_out}")
    print(f"矩阵 CSV：{matrix_out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
