# main.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from copy import deepcopy
from typing import Any, Dict, List

import yaml
import ray
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

from evaluator import EvalConfig, infer, eval as eval_cpu


def _dsn_from_env() -> str:
    load_dotenv()
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")

    hf_token = os.getenv("FYK_HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _ensure_db(conn: psycopg.Connection) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS evaluations (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        model_name TEXT NOT NULL,
        data_name TEXT NOT NULL,
        split TEXT NOT NULL,
        n_sampling INTEGER NOT NULL,
        k INTEGER NOT NULL,
        temperature DOUBLE PRECISION NOT NULL,
        top_p DOUBLE PRECISION NOT NULL,
        max_tokens INTEGER NOT NULL,
        prompt_type TEXT NOT NULL,
        surround_with_messages BOOLEAN NOT NULL,
        use_few_shot BOOLEAN NOT NULL,
        start_idx INTEGER NOT NULL,
        end_idx INTEGER NOT NULL,
        total INTEGER NOT NULL,
        correct INTEGER NOT NULL,
        accuracy DOUBLE PRECISION NOT NULL,
        pass_at_k DOUBLE PRECISION NOT NULL,
        output_file TEXT NOT NULL,
        completions_dir TEXT NOT NULL,
        CONSTRAINT uq_eval UNIQUE
            (model_name, data_name, split, n_sampling, k, temperature, top_p,
             prompt_type, surround_with_messages, use_few_shot, start_idx, end_idx)
    );"""
    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()


def _insert_result(conn: psycopg.Connection, result: Dict[str, Any]) -> bool:
    insert_sql = """
    INSERT INTO evaluations (
        model_name, data_name, split, n_sampling, k, temperature, top_p, max_tokens,
        prompt_type, surround_with_messages, use_few_shot, start_idx, end_idx,
        total, correct, accuracy, pass_at_k, output_file, completions_dir
    ) VALUES (
        %(model_name)s, %(data_name)s, %(split)s, %(n_sampling)s, %(k)s, %(temperature)s, %(top_p)s, %(max_tokens)s,
        %(prompt_type)s, %(surround_with_messages)s, %(use_few_shot)s, %(start_idx)s, %(end_idx)s,
        %(total)s, %(correct)s, %(accuracy)s, %(pass_at_k)s, %(output_file)s, %(completions_dir)s
    )
    ON CONFLICT ON CONSTRAINT uq_eval DO NOTHING;"""
    payload = {
        "model_name": result.get("model_name"),
        "data_name": result.get("data_name"),
        "split": result.get("split"),
        "n_sampling": result.get("n_sampling"),
        "k": result.get("k"),
        "temperature": result.get("temperature"),
        "top_p": result.get("top_p"),
        "max_tokens": result.get("max_tokens"),
        "prompt_type": result.get("prompt_type"),
        "surround_with_messages": bool(result.get("surround_with_messages")),
        "use_few_shot": bool(result.get("use_few_shot")),
        "start_idx": result.get("start_idx"),
        "end_idx": result.get("end_idx"),
        "total": result.get("total"),
        "correct": result.get("correct"),
        "accuracy": result.get("accuracy"),
        "pass_at_k": result.get("pass_at_k"),
        "output_file": result.get("output_file"),
        "completions_dir": result.get("completions_dir"),
    }
    with conn.cursor() as cur:
        cur.execute(insert_sql, payload)
        inserted = cur.rowcount == 1
    conn.commit()
    return inserted


# -------- Orchestrator --------
@ray.remote(num_cpus=1)
def orchestrate(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ray CPU orchestrator:
      - launches infer as a 1-GPU task
      - waits for its payload
      - runs CPU eval locally
      - returns DB-ready result dict
    """
    cfg = EvalConfig(**config_dict)

    # Launch GPU infer as a separate task
    infer_remote = ray.remote(num_gpus=1)(infer)
    infer_ref = infer_remote.remote(cfg)
    infer_payload = ray.get(infer_ref)

    if infer_payload.get("skipped"):
        # Return a DB-ready 'skipped' dict using CPU eval's shortcut
        return eval_cpu(cfg, infer_payload)

    # Run CPU eval here (no Ray, stays on this CPU orchestrator)
    return eval_cpu(cfg, infer_payload)


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping (dict).")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(args.config)

    models: List[str] = raw_cfg.get("models", [])
    eval_block: Dict[str, Any] = raw_cfg.get("eval", {})
    datasets_block: List[Dict[str, Any]] = raw_cfg.get("datasets", [])

    if not models:
        raise ValueError("YAML must include a non-empty 'models' list.")
    if not isinstance(eval_block, dict):
        raise ValueError("'eval' block must be a mapping.")
    if datasets_block and not isinstance(datasets_block, list):
        raise ValueError("'datasets' must be a list if provided.")
    if not datasets_block:
        datasets_block = [{}]

    dsn = _dsn_from_env()
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        _ensure_db(conn)

        # Ray init
        ray.init()

        # schedule orchestrators (one per model×dataset)
        futures = []
        for ds_overrides in datasets_block:
            if not isinstance(ds_overrides, dict):
                raise ValueError(f"Each item in 'datasets' must be a mapping, got: {type(ds_overrides)}")
            for model in models:
                cfg_dict = deepcopy(eval_block)
                cfg_dict.update(ds_overrides)
                cfg_dict["model_name_or_path"] = model
                futures.append(orchestrate.remote(cfg_dict))
                print(f"[main] Scheduled: model={model} data={cfg_dict.get('data_name')} split={cfg_dict.get('split')}")

        # consume as they complete
        pending = list(futures)
        results_all: List[Dict[str, Any]] = []
        total = len(pending)
        finished = 0

        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            ref = done[0]
            try:
                res: Dict[str, Any] = ray.get(ref)
            except Exception as e:
                finished += 1
                print(f"[main] Orchestrator failed ({finished}/{total}): {e}")
                continue

            finished += 1
            results_all.append(res)

            if res.get("skipped"):
                print(
                    f"[main] Skipped ({finished}/{total}): "
                    f"model={res.get('model_name')} data={res.get('data_name')} split={res.get('split')} "
                    f"file={res.get('output_file')}"
                )
                continue

            ok = _insert_result(conn, res)
            if ok:
                print(
                    f"[main] Stored ({finished}/{total}): model={res['model_name']} "
                    f"data={res['data_name']} split={res['split']} "
                    f"| acc={res['accuracy']:.4f} | pass@{res['k']}={res['pass_at_k']:.4f} "
                    f"| file={res['output_file']}"
                )
            else:
                print(
                    f"[main] Duplicate (DB upsert skipped) ({finished}/{total}): "
                    f"model={res['model_name']} data={res['data_name']} split={res['split']} "
                    f"| file={res['output_file']}"
                )

        # Recap
        print("\n[main] ---- Final Results ----")
        for res in results_all:
            if res.get("skipped"):
                print(
                    f"SKIPPED  | model={res.get('model_name')} "
                    f"data={res.get('data_name')} split={res.get('split')} "
                    f"file={res.get('output_file')}"
                )
            else:
                print(
                    f"STORED   | model={res['model_name']} "
                    f"data={res['data_name']} split={res['split']} "
                    f"acc={res['accuracy']:.4f} pass@{res['k']}={res['pass_at_k']:.4f} "
                    f"file={res['output_file']}"
                )


if __name__ == "__main__":
    main()

