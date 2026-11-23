# -*- coding: utf-8 -*-
"""
elicit_direct.py —— LLM 直接判断（一个独立的概念通道）
====================================================
背景：26 个概念的 mean 已近似"总体有害程度"，但那是从 26 个孤立 yes/no 间接
拼出来的。本脚本直接问模型两个问题（推理优先 + 受限解码），得到两个新概念：
  DIRECT_TOXIC    综合判断这段文本是否有害（含明显 + 隐含）
  DIRECT_IMPLICIT 是否含有"隐含"的攻击（反讽/影射/谐音/委婉贬损）

它们与 v8/v9 的 26 概念正交度更高（尤其 DIRECT_IMPLICIT 直击隐式缺口），
作为第 27/28 个概念拼进下游，属于"新增测量"，仍严格 CBM。

与 v9 复用同一套 vLLM/受限解码/断点逻辑；输出独立目录 v9_direct_<model>/。

用法（Git Bash）：
  python scripts/elicit_direct.py \
    --model_path /root/autodl-tmp/glm-4-9b-chat \
    --data_path data/raw/TOXICN --split train --limit 8
  # 全量：去掉 --limit，train / test 各跑一遍
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

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REASONING_TEMPLATE = """你是中文内容审核助手。下面是一段社交平台文本。
请判断它是否有害，包括：明显的辱骂/攻击，以及隐含的讽刺、影射、谐音/拆字变体、委婉贬损。
如果无害，请明确说明"无害"。

文本：{text}

请用一句话（不超过 80 字）说明你的判断及依据。
分析："""

DIRECT_ITEMS = [
    {
        "item_id": "DIRECT_TOXIC",
        "name": "有害综合判断",
        "question": "综合判断，这段文本是否有害（包括明显辱骂与隐含的讽刺、影射、谐音、委婉贬损）？",
    },
    {
        "item_id": "DIRECT_IMPLICIT",
        "name": "隐含攻击判断",
        "question": "这段文本是否含有隐含的攻击（反讽、影射、谐音/变体、委婉贬损），即使字面上看不出来？",
    },
]

ITEM_TEMPLATE_DIRECT = """你是严格的中文文本现象判定助手。请根据文本与已给出的分析回答判断题。
只依据文本内容判断，不补充外部事实。

文本：{text}

【分析】{reasoning}

问题：{question}

只回答“是”或“否”。回答："""

MODEL_PROMPT_SUFFIX = {"glm-4-9b-chat": "\n", "Qwen3.5-9B": "\n"}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(data_path, split):
    if not data_path.exists():
        raise FileNotFoundError(f"data_path 不存在：{data_path}")
    json_path = data_path / f"{split}.json" if data_path.is_dir() else data_path
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到 {json_path}")
    dataset_tag = data_path.name if data_path.is_dir() else data_path.parent.name
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"数据文件顶层必须是 JSON list：{json_path}")
    samples = []
    for i, item in enumerate(raw):
        content = str(item.get("content", ""))
        if not content.strip():
            raise ValueError(f"数据第 {i+1} 条 content 为空")
        sid = item.get("id") or f"{dataset_tag}_{split}_{i:04d}"
        samples.append({
            "id": str(sid), "content": content, "toxic": item.get("toxic"),
            "expression": item.get("expression"),
        })
    return samples, dataset_tag, json_path


def load_vllm_model(model_path, gpu_memory_utilization, max_model_len,
                    max_num_seqs, max_num_batched_tokens):
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


def detect_chat_kwargs(tokenizer):
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


def get_candidate_ids(words, tokenizer):
    ids = []
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"verbalizer {w!r} 不是单 token：{t}")
        ids.extend(t)
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


def _logprob_value(v):
    return float(v.logprob) if hasattr(v, "logprob") else float(v)


def get_probability_result(request_output, tokenizer, yes_ids, no_ids):
    completion = request_output.outputs[0]
    if not completion.logprobs or not completion.logprobs[0]:
        raise RuntimeError("没有首 token logprobs")
    token_id = int(completion.token_ids[0])
    first_token = ("是" if token_id in yes_ids else
                   "否" if token_id in no_ids else
                   tokenizer.decode([token_id], skip_special_tokens=False))
    logprob_map = completion.logprobs[0]

    def mass_for(ids):
        s = 0.0
        for tid in ids:
            e = logprob_map.get(tid)
            if e is not None:
                s += math.exp(_logprob_value(e))
        return s

    yes_mass = mass_for(yes_ids)
    no_mass = mass_for(no_ids)
    cm = yes_mass + no_mass
    p_yes = None if (cm <= 0 or not math.isfinite(cm)) else yes_mass / cm
    return {
        "first_token": first_token,
        "p_yes": None if p_yes is None else round(float(p_yes), 6),
        "cand_mass": round(float(cm), 6),
    }


def make_reasoning_prompt(sample):
    return REASONING_TEMPLATE.format(text=sample["content"])


def make_question_prompt(sample, reasoning, item):
    return ITEM_TEMPLATE_DIRECT.format(
        text=sample["content"], reasoning=reasoning, question=item["question"]
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
    if parsed[0].get("config") != config:
        raise ValueError("checkpoint 配置不一致；请 --overwrite 重跑")
    results = {}
    ids_set = set(item_ids)
    for obj in parsed[1:]:
        sid = obj.get("id")
        if sid not in dataset_ids or sid in results:
            raise ValueError(f"checkpoint 样本异常：{sid}")
        if set(obj.get("p_yes", {})) != ids_set:
            raise ValueError(f"checkpoint 样本 {sid} 条目不完整")
        results[sid] = obj
    return results


def parse_args():
    p = argparse.ArgumentParser(description="LLM 直接判断（推理优先）")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--data_path", required=True, type=Path)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--max_num_batched_tokens", type=int, default=16384)
    p.add_argument("--batch_samples", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reasoning_max_tokens", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()

    samples, dataset_tag, json_path = load_dataset(data_path, args.split)
    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"技术冒烟限制：{len(samples)} 条")

    item_ids = [d["item_id"] for d in DIRECT_ITEMS]
    model_name = model_path.name
    split_tag = json_path.stem
    suffix = split_tag + (f"_limit{args.limit}" if args.limit is not None else "")
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  PROJECT_ROOT / "data" / "processed" / dataset_tag / f"v9_direct_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = output_dir / f"items_direct_{model_name}_{suffix}.json"
    checkpoint = output_dir / f"items_direct_{model_name}_{suffix}.checkpoint.jsonl"

    config = {
        "model_path": str(model_path), "model_name": model_name,
        "input_path": str(json_path), "input_sha256": sha256_file(json_path),
        "split": split_tag, "n_samples": len(samples),
        "prompt_version": "v9-direct-reasoning",
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "temperature": 0, "item_ids": item_ids,
    }
    results = read_checkpoint(checkpoint, config, {x["id"] for x in samples}, item_ids, args.overwrite)
    todo = [x for x in samples if x["id"] not in results]

    print("=" * 72)
    print(f"数据集 {dataset_tag}/{split_tag} 样本 {len(samples)} 待处理 {len(todo)}")
    print(f"直接判断概念 {len(DIRECT_ITEMS)} 个；每样本 = 1 分析 + {len(DIRECT_ITEMS)} 问询")
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

        from vllm import SamplingParams
        reasoning_params = SamplingParams(max_tokens=args.reasoning_max_tokens, temperature=0, stop=["\n"])
        sampling_params, restrict_mode = build_restricted_sampling(yes_ids + no_ids)

        progress = tqdm(total=len(samples), initial=len(results), desc="直接判断测量")
        try:
            for start in range(0, len(todo), args.batch_samples):
                batch = todo[start:start + args.batch_samples]

                # 阶段 1：分析
                rp = [make_reasoning_prompt(s) for s in batch]
                ro = llm.generate(render_prompts(tokenizer, rp, chat_kwargs, model_name),
                                  reasoning_params, use_tqdm=False)
                reasonings = [(o.outputs[0].text if o.outputs else "").strip() for o in ro]
                reasonings = [r or "（无法分析）" for r in reasonings]

                # 阶段 2：两个直接判断
                prompts, keys = [], []
                for s, r in zip(batch, reasonings):
                    for it in DIRECT_ITEMS:
                        prompts.append(make_question_prompt(s, r, it))
                        keys.append((s["id"], it["item_id"]))
                co = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                                  sampling_params, use_tqdm=False)
                if len(co) != len(keys):
                    raise RuntimeError("直接判断输出数与请求数不一致")

                records = {s["id"]: {"record_type": "sample", "id": s["id"],
                                     "reasoning": r, "p_yes": {}, "first_token": {},
                                     "cand_mass": {}, "restricted_decoding": restrict_mode}
                           for s, r in zip(batch, reasonings)}
                for (sid, iid), o in zip(keys, co):
                    v = get_probability_result(o, tokenizer, yes_ids, no_ids)
                    rec = records[sid]
                    rec["p_yes"][iid] = v["p_yes"]
                    rec["first_token"][iid] = v["first_token"]
                    rec["cand_mass"][iid] = v["cand_mass"]

                with checkpoint.open("a", encoding="utf-8") as f:
                    for s in batch:
                        f.write(json.dumps(records[s["id"]], ensure_ascii=False, separators=(",", ":")) + "\n")
                        results[s["id"]] = records[s["id"]]
                    f.flush()
                    os.fsync(f.fileno())
                progress.update(len(batch))
        finally:
            progress.close()

    matrix = {}
    invalid = []
    for s in samples:
        rec = results[s["id"]]
        matrix[s["id"]] = rec["p_yes"]
        for iid in item_ids:
            if rec["p_yes"].get(iid) is None:
                invalid.append({"id": s["id"], "item_id": iid})

    meta = {
        "script": "elicit_direct.py", "prompt_version": config["prompt_version"],
        "config": config, "n_samples": len(samples), "n_items": len(DIRECT_ITEMS),
        "item_definitions": DIRECT_ITEMS,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"), "n_invalid": len(invalid),
    }
    tmp = json_out.with_suffix(json_out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "matrix": matrix, "invalid": invalid}, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, json_out)

    print("=" * 72)
    print(f"完成：样本 {len(samples)}；无效 p_yes {len(invalid)}")
    print(f"JSON：{json_out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
