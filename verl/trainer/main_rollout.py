#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal single-sample rollout runner for verl-agent, using
agent_system/multi_turn_rollout/rollout_loop_multi.py::MultiAgentTrajectoryCollector

功能：
1) 只跑单条样本（来自 --question 或 --gsm8k_jsonl 的第一条）
2) 调用 MultiAgentTrajectoryCollector 做一次（或一回合）rollout
3) 将完整轨迹以 JSONL 写入到 --output_dir 目录下

使用示例：
PYTHONPATH=. python tools/main_rollout.py \
  --model_path /p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-1.5B-Instruct \
  --output_dir /p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K_test \
  --question "John has 3 apples, buys 2 bags with 5 apples each. How many apples in total?"

如需从 GSM8K jsonl 读取第一条：
PYTHONPATH=. python tools/main_rollout.py \
  --model_path /path/to/model \
  --output_dir /p/scratch/.../GSM8K_test \
  --gsm8k_jsonl /path/to/gsm8k_train.jsonl
"""

import os
import sys
import json
import time
import argparse
import inspect
from pathlib import Path
from datetime import datetime

# --------- 安全导入 collector ---------
try:
    # 根据你的目录结构，保证当前工作目录是 repo 根：PYTHONPATH=. python tools/main_rollout.py
    from agent_system.multi_turn_rollout.rollout_loop_multi import MultiAgentTrajectoryCollector
except Exception as e:
    print("[FATAL] 无法导入 MultiAgentTrajectoryCollector：", repr(e))
    print("请确认：PYTHONPATH 指向 verl-agent 仓库根目录；文件路径存在。")
    sys.exit(1)

# --------- 可选：huggingface 模型（仅在 collector 需要时）---------
# 有的分支由 collector 内部管理模型；有的需要外部传入：
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

def load_first_gsm8k_item(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line:
            raise ValueError(f"{jsonl_path} 为空")
        obj = json.loads(line)
    # 常见字段：question / answer
    q = obj.get("question") or obj.get("prompt") or obj.get("input") or ""
    a = obj.get("answer") or obj.get("label") or obj.get("output") or ""
    return {"question": q, "answer": a}

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def build_minimal_inputs(args):
    """
    依据 MultiAgentTrajectoryCollector 的 __init__ 参数，自适应准备最小入参。
    你可以在打印出的参数清单基础上，针对你的分支，增补/改名下面的键。
    """
    sig = inspect.signature(MultiAgentTrajectoryCollector.__init__)
    print("\n[INFO] MultiAgentTrajectoryCollector.__init__ 参数：")
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        print(f"  - {name} (default={param.default})")

    kwargs = {}

    # 常见可能需要的键（按你分支实际命名调整；不在签名里的键不会传入）：
    # 1) 模型与分词器（若需要外部提供）
    if "tokenizer" in sig.parameters and args.model_path and AutoTokenizer:
        kwargs["tokenizer"] = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if "model" in sig.parameters and args.model_path and AutoModelForCausalLM:
        kwargs["model"] = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

    # 2) agent 相关（示例：agent_num / roles / debate 等）
    if "agent_num" in sig.parameters:
        kwargs["agent_num"] = args.agent_num
    if "max_turns" in sig.parameters:
        kwargs["max_turns"] = args.max_turns
    if "temperature" in sig.parameters:
        kwargs["temperature"] = args.temperature

    # 3) 环境/奖励管理（如你的分支要求）
    if "env_name" in sig.parameters:
        kwargs["env_name"] = "gsm8k"  # 仅作为标记；真实环境一般在 env 层，这里只是占位
    if "reward_manager" in sig.parameters and args.reward_manager is not None:
        kwargs["reward_manager"] = args.reward_manager  # 你可以传入自定义的 reward 管理器

    # 4) 日志与输出控制
    if "log_dir" in sig.parameters:
        kwargs["log_dir"] = args.output_dir
    if "disable_log_stats" in sig.parameters:
        kwargs["disable_log_stats"] = True

    print("\n[INFO] 传入 MultiAgentTrajectoryCollector 的 kwargs：")
    for k, v in kwargs.items():
        print(f"  - {k}: {type(v)}")

    return kwargs

def run_single_rollout(args):
    ensure_dir(args.output_dir)
    # 准备样本
    if args.question:
        sample = {"question": args.question, "answer": args.answer or ""}
    elif args.gsm8k_jsonl:
        sample = load_first_gsm8k_item(args.gsm8k_jsonl)
    else:
        # 极简内置样本
        sample = {
            "question": "If Tom has 7 marbles and buys 5 more, then gives 3 to Lily, how many marbles does Tom have?",
            "answer": "9"
        }

    # 实例化 collector
    collector_kwargs = build_minimal_inputs(args)
    collector = MultiAgentTrajectoryCollector(**collector_kwargs)

    # 单次/单轮 rollout 的调用兼容（不同分支方法名不同）
    inputs_for_collector = {
        # 常见传参：question / sample / max_steps / seed 等（按分支而定）
        "question": sample["question"],
        "reference_answer": sample.get("answer", ""),
        "max_steps": args.max_steps
    }

    print("\n[INFO] 启动单样本 rollout ...")
    t0 = time.time()
    if hasattr(collector, "collect"):
        result = collector.collect(**{k: v for k, v in inputs_for_collector.items() if k in inspect.signature(collector.collect).parameters})
    elif hasattr(collector, "run_episode"):
        result = collector.run_episode(**{k: v for k, v in inputs_for_collector.items() if k in inspect.signature(collector.run_episode).parameters})
    else:
        raise RuntimeError("MultiAgentTrajectoryCollector 既没有 collect(...) 也没有 run_episode(...) 方法，请查看你的分支实现。")
    dt = time.time() - t0

    # 组织输出
    out = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(dt, 3),
        "question": sample["question"],
        "reference_answer": sample.get("answer", ""),
        "rollout": result,  # 期望包含 messages/actions/rewards/agent_outputs 等
    }

    # 写 JSONL
    outfile = os.path.join(args.output_dir, "gsm8k_single_rollout.jsonl")
    with open(outfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\n[OK] 单次 rollout 完成，已追加写入：{outfile}")
    print("建议检查字段：rollout.messages / rollout.actions / rollout.rewards / final_answer 等（取决于分支实现）")

def main():
    parser = argparse.ArgumentParser(description="Single-sample rollout runner for verl-agent MultiAgentTrajectoryCollector")
    parser.add_argument("--model_path", type=str, default="", help="可选；若 collector 需要外部模型/分词器，则提供其 HF 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 JSONL 目录")
    parser.add_argument("--question", type=str, default="", help="直接提供一个 GSM8K 问题文本")
    parser.add_argument("--answer", type=str, default="", help="可选：标准答案（用于对比/奖励）")
    parser.add_argument("--gsm8k_jsonl", type=str, default="", help="可选；从 JSONL 读取第一条样本（含 question/answer 字段）")

    parser.add_argument("--agent_num", type=int, default=2, help="若分支支持多 agent，这里设定数量")
    parser.add_argument("--max_turns", type=int, default=1, help="一次 rollout 的最大轮数（若 collector 支持）")
    parser.add_argument("--max_steps", type=int, default=1, help="一次 rollout 的最大步数（若 collector 支持）")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度（若 collector 支持）")

    parser.add_argument("--reward_manager", type=str, default=None, help="可选：自定义 reward 管理器名或路径（按分支实现）")

    args = parser.parse_args()
    run_single_rollout(args)

if __name__ == "__main__":
    main()