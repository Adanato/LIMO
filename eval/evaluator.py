# evaluator.py
import os
import json
import pickle
import importlib.util
from math import comb
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm.envs as envs

# --- your existing utils ---
from utils.utils import set_seed
from utils.parser import parse_question, parse_ground_truth, extract_answer
from utils.data_loader import load_data
from utils.math_normalization import *  # noqa: F401
from utils.grader import check_is_correct


@dataclass
class EvalConfig:
    # Core eval
    model_name_or_path: str
    data_name: str = "math"
    dataset_config: Optional[str] = None
    split: str = "test"
    data_dir: str = "./data"
    start_idx: int = 0
    end_idx: int = -1

    # Generation
    n_sampling: int = 1
    k: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    dtype: str = "auto"
    stop: Optional[List[str]] = None

    # Prompting
    prompt_type: str = "qwen-base"
    prompt_file_path: str = "./prompts"
    surround_with_messages: bool = False
    use_few_shot: bool = False

    # Output
    output_dir: str = "./outputs"
    completions_save_dir: str = "./completions"

    # Misc
    seed: int = 0

    # Optional: database path (used by main.py)
    db_path: str = "./eval_results.db"


def _get_three_prompt(prompt_type: str, data_name: str) -> Tuple[str, str, str]:
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "system_prompt"):
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    if not hasattr(module, "few_shot_prompt"):
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    if not hasattr(module, "question_format"):
        raise AttributeError(f"'question_format' not found in {file_path}")

    return module.system_prompt, module.few_shot_prompt, module.question_format


def _get_conversation_prompt_by_messages(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def _save_completions(completions, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True
                )
    with open(filepath, "wb") as f:
        pickle.dump(completions, f)


def infer(config: EvalConfig) -> Dict[str, Any]:
    """
    GPU-only stage: run vLLM generation and return payload needed for CPU eval.
    NOTE: This does NOT compute accuracy/pass@k or write JSONL.
    """
    set_seed(config.seed)

    top_p = 1.0 if config.temperature == 0 else config.top_p
    model_name_or_path = config.model_name_or_path

    # Plan n vs epochs (preserve original behavior)
    n_sampling = config.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = max(1, n_sampling // factor)

    # Load data slice
    all_examples = load_data(config.data_name, config.split, config.data_dir)
    start = config.start_idx
    end = len(all_examples) if config.end_idx == -1 else config.end_idx
    examples = all_examples[start:end]

    # Paths
    model_name_for_path = "/".join(model_name_or_path.rstrip("/").split("/")[-3:])
    out_file_prefix = f'{config.split}_{config.prompt_type}_t{config.temperature}'
    out_dir_model = f'{config.output_dir}/{model_name_for_path}/{config.data_name}'
    os.makedirs(out_dir_model, exist_ok=True)
    out_file = f'{out_dir_model}/{out_file_prefix}_k{config.n_sampling}_s{start}_e{end}.jsonl'

    if os.path.exists(out_file):
        # Nothing else to do; CPU eval can be skipped by orchestrator.
        return {
            "skipped": True,
            "model_name": model_name_or_path,
            "data_name": config.data_name,
            "split": config.split,
            "start_idx": start,
            "end_idx": end,
            "output_file": out_file,
            "completions_dir": f'{config.completions_save_dir}/{model_name_for_path}/{config.data_name}',
            "total": len(examples),
            # fields needed by orchestrator to return DB payload
            "n_sampling": config.n_sampling,
            "k": config.k,
            "temperature": config.temperature,
            "top_p": top_p,
            "max_tokens": config.max_tokens,
            "prompt_type": config.prompt_type,
            "surround_with_messages": int(config.surround_with_messages),
            "use_few_shot": int(config.use_few_shot),
        }

    completions_dir = f'{config.completions_save_dir}/{model_name_for_path}/{config.data_name}'
    os.makedirs(completions_dir, exist_ok=True)

    # GPU env & tokenizer
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if os.environ.get("CUDA_VISIBLE_DEVICES") else []
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Build prompts once
    prompt_batch: List[str] = []
    for example in tqdm(examples, total=len(examples), desc="Preparing prompts"):
        question = parse_question(example, config.data_name)
        system_prompt, few_shot_prompt, question_format = _get_three_prompt(config.prompt_type, config.data_name)

        if config.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)

        if config.surround_with_messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = _get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)

        prompt_batch.append(cur_prompt)

    # Create LLM (GPU) and sampling params
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=len(available_gpus) if available_gpus else 1,
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        n=factor,
        top_p=top_p,
        stop=config.stop
    )

    # Generate across epochs and collect only generated text
    file_outputs: List[Dict[str, Any]] = []
    for cur_epoch in range(generation_epoch):
        completions_path = f'{completions_dir}/{out_file_prefix}_k{config.n_sampling}_s{start}_e{end}_gen_round{cur_epoch}.pkl'
        completions = llm.generate(prompt_batch, sampling_params)
        _save_completions(completions, completions_path)

        for i in range(len(examples)):
            d = examples[i]
            question = parse_question(d, config.data_name)
            outputs_i = completions[i].outputs
            generated_responses = [outputs_i[j].text for j in range(len(outputs_i))]

            if cur_epoch == 0:
                entry = {
                    "question": question,
                    "generated_responses": generated_responses,
                }
                if "id" in d:
                    entry["id"] = d["id"]
                if "source" in d:
                    entry["source"] = d["source"]
                file_outputs.append(entry)
            else:
                file_outputs[i]["generated_responses"].extend(generated_responses)

    # Return payload for CPU eval
    return {
        "skipped": False,
        "examples": examples,            # needed for ground-truth parsing
        "file_outputs": file_outputs,    # generated responses & meta
        "meta": {
            "model_name": model_name_or_path,
            "data_name": config.data_name,
            "split": config.split,
            "start_idx": start,
            "end_idx": end,
            "output_file": out_file,
            "completions_dir": completions_dir,
            "n_sampling": config.n_sampling,
            "k": config.k,
            "temperature": config.temperature,
            "top_p": top_p,
            "max_tokens": config.max_tokens,
            "prompt_type": config.prompt_type,
            "surround_with_messages": int(config.surround_with_messages),
            "use_few_shot": int(config.use_few_shot),
        },
    }


def eval(config: EvalConfig, infer_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    CPU-only stage: score generations, compute metrics, write JSONL, and return DB-ready dict.
    """
    if infer_payload.get("skipped"):
        # Mirror the DB row shape, flagged as skipped (metrics unknown)
        return {
            "model_name": infer_payload["model_name"],
            "data_name": infer_payload["data_name"],
            "split": infer_payload["split"],
            "n_sampling": infer_payload["n_sampling"],
            "k": infer_payload["k"],
            "temperature": infer_payload["temperature"],
            "top_p": infer_payload["top_p"],
            "max_tokens": infer_payload["max_tokens"],
            "prompt_type": infer_payload["prompt_type"],
            "surround_with_messages": infer_payload["surround_with_messages"],
            "use_few_shot": infer_payload["use_few_shot"],
            "start_idx": infer_payload["start_idx"],
            "end_idx": infer_payload["end_idx"],
            "total": infer_payload["total"],
            "correct": None,
            "accuracy": None,
            "pass_at_k": None,
            "output_file": infer_payload["output_file"],
            "completions_dir": infer_payload["completions_dir"],
            "skipped": True,
        }

    examples: List[Dict[str, Any]] = infer_payload["examples"]
    file_outputs: List[Dict[str, Any]] = infer_payload["file_outputs"]
    meta = infer_payload["meta"]

    out_file = meta["output_file"]
    pass_at_k_list: List[float] = []
    k = max(1, int(meta["k"]))
    correct_cnt = 0

    # Score
    for i in tqdm(range(len(examples)), desc="Scoring"):
        d = examples[i]
        _, gt_ans = parse_ground_truth(d, meta["data_name"])
        generated_responses = file_outputs[i]["generated_responses"]
        generated_answers = [extract_answer(r, meta["data_name"]) for r in generated_responses]
        is_correct_list = [check_is_correct(ans, gt_ans) for ans in generated_answers]
        is_correct = any(is_correct_list)
        correct_cnt += 1 if is_correct else 0

        file_outputs[i]["generated_answers"] = generated_answers
        file_outputs[i]["gold_answer"] = gt_ans
        file_outputs[i]["is_correct"] = is_correct
        file_outputs[i]["answers_correctness"] = is_correct_list

        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1.0
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0.0)

    # Write JSONL
    tmp_path = out_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for d in tqdm(file_outputs, desc="Writing jsonl"):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp_path, out_file)

    total = len(examples)
    acc = (correct_cnt / total) if total else 0.0
    pass_k_val = (sum(pass_at_k_list) / len(pass_at_k_list)) if pass_at_k_list else acc

    # Return DB-ready row
    return {
        "model_name": meta["model_name"],
        "data_name": meta["data_name"],
        "split": meta["split"],
        "n_sampling": meta["n_sampling"],
        "k": k,
        "temperature": meta["temperature"],
        "top_p": meta["top_p"],
        "max_tokens": meta["max_tokens"],
        "prompt_type": meta["prompt_type"],
        "surround_with_messages": meta["surround_with_messages"],
        "use_few_shot": meta["use_few_shot"],
        "start_idx": meta["start_idx"],
        "end_idx": meta["end_idx"],
        "total": total,
        "correct": correct_cnt,
        "accuracy": acc,
        "pass_at_k": pass_k_val,
        "output_file": out_file,
        "completions_dir": meta["completions_dir"],
        "skipped": False,
    }

