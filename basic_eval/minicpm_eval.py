import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from util import exact_match_score, f1_score

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x


SYSTEM_PROMPT = (
    "You answer questions using only the given context. "
    "Answer with a short phrase or yes/no. Do not add explanations."
)


def build_context(paragraphs, include_titles=True, max_paragraphs=None):
    blocks = []
    for idx, (title, sentences) in enumerate(paragraphs):
        if max_paragraphs is not None and idx >= max_paragraphs:
            break
        paragraph_text = " ".join(sentences).strip()
        if include_titles:
            block = "{}\n{}".format(title, paragraph_text)
        else:
            block = paragraph_text
        blocks.append(block.strip())
    return "\n\n".join([b for b in blocks if b])


def iter_examples(path, limit=None, ignore_context=False):
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                if ignore_context:
                    record["context"] = ""
                elif isinstance(record.get("context"), list):
                    record["context"] = build_context(record["context"], include_titles=True)
                yield record
        return

    with open(path, "r") as f:
        data = json.load(f)
    for idx, ex in enumerate(data):
        if limit is not None and idx >= limit:
            break
        context = ""
        if not ignore_context:
            context = build_context(ex.get("context", []), include_titles=True)
        yield {
            "id": ex.get("id") or ex.get("_id"),
            "question": ex.get("question", ""),
            "context": context,
            "answer": ex.get("answer"),
        }


def build_prompt(tokenizer, question, context, system_prompt, use_chat_template):
    if (context or "").strip():
        user_prompt = "Context:\n{}\n\nQuestion:\n{}\n\nAnswer:".format(
            context, question
        )
    else:
        user_prompt = "Question:\n{}\n\nAnswer:".format(question)
    if use_chat_template and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "{}\n\n{}".format(system_prompt, user_prompt)


def extract_answer(text):
    cleaned = text.strip()
    if not cleaned:
        return ""
    for tag in ["Answer:", "answer:", "A:", "a:"]:
        if tag in cleaned:
            cleaned = cleaned.split(tag, 1)[1].strip()
            if not cleaned:
                return ""
    lines = cleaned.splitlines()
    if not lines:
        return ""
    cleaned = lines[0].strip()
    for prefix in ["the answer is ", "answer is "]:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ["'", '"']:
        cleaned = cleaned[1:-1].strip()
    lowered = cleaned.lower().lstrip(" \"'([{")
    if lowered.startswith("yes") and (len(lowered) == 3 or not lowered[3].isalpha()):
        return "yes"
    if lowered.startswith("no") and (len(lowered) == 2 or not lowered[2].isalpha()):
        return "no"
    return cleaned


def load_existing_ids(pred_file):
    ids = set()
    if not os.path.exists(pred_file):
        return ids
    with open(pred_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in record:
                ids.add(str(record["id"]))
    return ids


def load_predictions(pred_file):
    preds = {}
    if not os.path.exists(pred_file):
        return preds
    with open(pred_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" in record and "prediction" in record:
                preds[str(record["id"])] = record["prediction"]
    return preds


def write_metrics_snapshot(metrics, metrics_file=None, metrics_log=None):
    if metrics_file:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if metrics_log:
        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def evaluate_predictions(data_path, pred_file, limit=None):
    preds = load_predictions(pred_file)
    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    for ex in iter_examples(data_path, limit=limit):
        gold = ex.get("answer")
        if gold is None:
            continue
        ex_id = str(ex.get("id"))
        if ex_id not in preds:
            continue
        pred = preds[ex_id]
        em_sum += 1.0 if exact_match_score(pred, gold) else 0.0
        f1_sum += f1_score(pred, gold)[0]
        total += 1
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "total": 0, "predicted": len(preds)}
    return {
        "exact_match": 100.0 * em_sum / total,
        "f1": 100.0 * f1_sum / total,
        "total": total,
        "predicted": len(preds),
    }


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return str(ROOT_DIR / path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="llm_data/hotpot_dev_distractor_v1.jsonl")
    parser.add_argument("--pred_file", default="predictions/minicpm3-4b_dev.jsonl")
    parser.add_argument("--metrics_file", default=None)
    parser.add_argument(
        "--model",
        default=os.environ.get("MINICPM_MODEL", "openbmb/MiniCPM3-4B"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_context_chars", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--metrics_log", default=None)
    parser.add_argument("--ignore_context", action="store_true")
    args = parser.parse_args()

    args.data_file = resolve_path(args.data_file)
    args.pred_file = resolve_path(args.pred_file)
    args.metrics_file = resolve_path(args.metrics_file)
    args.metrics_log = resolve_path(args.metrics_log)

    if os.path.exists(args.pred_file) and not args.resume and not args.overwrite:
        raise SystemExit(
            "pred_file exists; use --resume to continue or --overwrite to replace it"
        )

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(args.device)
    model.eval()

    os.makedirs(os.path.dirname(args.pred_file) or ".", exist_ok=True)
    existing_preds = load_predictions(args.pred_file) if args.resume else {}
    existing_ids = set(existing_preds.keys())
    mode = "a" if args.resume else "w"

    examples = list(
        iter_examples(
            args.data_file, limit=args.limit, ignore_context=args.ignore_context
        )
    )
    use_chat_template = not args.no_chat_template
    start_time = time.time()
    em_sum = 0.0
    f1_sum = 0.0
    gold_cnt = 0
    predicted_total = len(existing_ids)

    if existing_preds:
        for ex in examples:
            ex_id = str(ex.get("id"))
            if ex_id not in existing_preds:
                continue
            gold = ex.get("answer")
            if gold is None:
                continue
            pred = existing_preds[ex_id]
            em_sum += 1.0 if exact_match_score(pred, gold) else 0.0
            f1_sum += f1_score(pred, gold)[0]
            gold_cnt += 1
        metrics = {
            "exact_match": 100.0 * em_sum / gold_cnt if gold_cnt else 0.0,
            "f1": 100.0 * f1_sum / gold_cnt if gold_cnt else 0.0,
            "total": gold_cnt,
            "predicted": predicted_total,
            "seconds": time.time() - start_time,
        }
        write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)

    with open(args.pred_file, mode) as out_f, torch.inference_mode():
        new_preds = 0
        for i in tqdm(range(0, len(examples), args.batch_size)):
            batch = examples[i : i + args.batch_size]
            prompts = []
            kept = []
            for ex in batch:
                ex_id = str(ex.get("id"))
                if ex_id in existing_ids:
                    continue
                context = ex.get("context", "")
                if args.max_context_chars is not None and len(context) > args.max_context_chars:
                    context = context[: args.max_context_chars].rsplit(" ", 1)[0]
                prompt = build_prompt(
                    tokenizer,
                    ex.get("question", ""),
                    context,
                    args.system_prompt,
                    use_chat_template,
                )
                prompts.append(prompt)
                kept.append(ex)
            if not prompts:
                continue

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False
            ).to(model.device)
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "temperature": max(args.temperature, 1e-5),
                "top_p": args.top_p,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": False,
            }
            outputs = model.generate(**inputs, **gen_kwargs)
            for j, ex in enumerate(kept):
                input_len = int(inputs["attention_mask"][j].sum().item())
                generated = outputs[j][input_len:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                pred = extract_answer(text)
                gold = ex.get("answer")
                if gold is not None:
                    em_sum += 1.0 if exact_match_score(pred, gold) else 0.0
                    f1_sum += f1_score(pred, gold)[0]
                    gold_cnt += 1
                out_f.write(
                    json.dumps(
                        {"id": ex.get("id"), "prediction": pred}, ensure_ascii=False
                    )
                    + "\n"
                )
                predicted_total += 1
                new_preds += 1
            out_f.flush()
            if args.checkpoint_every > 0 and new_preds >= args.checkpoint_every:
                metrics = {
                    "exact_match": 100.0 * em_sum / gold_cnt if gold_cnt else 0.0,
                    "f1": 100.0 * f1_sum / gold_cnt if gold_cnt else 0.0,
                    "total": gold_cnt,
                    "predicted": predicted_total,
                    "seconds": time.time() - start_time,
                }
                write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)
                new_preds = 0

    elapsed = time.time() - start_time
    metrics = {
        "exact_match": 100.0 * em_sum / gold_cnt if gold_cnt else 0.0,
        "f1": 100.0 * f1_sum / gold_cnt if gold_cnt else 0.0,
        "total": gold_cnt,
        "predicted": predicted_total,
        "seconds": elapsed,
    }
    print(
        "EM {exact_match:.2f} | F1 {f1:.2f} | total {total} | preds {predicted}".format(
            **metrics
        )
    )
    write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)


if __name__ == "__main__":
    main()
