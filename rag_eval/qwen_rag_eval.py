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
from evidence_utils import build_evidence_context, select_evidence_sentences

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x


SYSTEM_PROMPT = (
    "You answer questions using only the given context. "
    "Answer with a short phrase or yes/no. Do not add explanations."
)


def load_corpus(corpus_file):
    titles = []
    texts = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            titles.append(record["title"])
            texts.append(record["text"])
    return titles, texts


def load_retrieval(retrieval_file):
    mapping = {}
    with open(retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            mapping[str(record["id"])] = {
                "top_ids": record.get("top_ids", []),
                "evidence_context": record.get("evidence_context", ""),
            }
    return mapping


def iter_examples(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def build_prompt(tokenizer, question, context, system_prompt, use_chat_template):
    user_prompt = "Context:\n{}\n\nQuestion:\n{}\n\nAnswer:".format(context, question)
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


def load_predictions(pred_file):
    preds = {}
    if not os.path.exists(pred_file):
        return preds
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" in record and "prediction" in record:
                preds[str(record["id"])] = record["prediction"]
    return preds


def write_metrics_snapshot(metrics, metrics_file=None, metrics_log=None):
    if metrics_file:
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if metrics_log:
        with open(metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def build_context_from_ids(doc_ids, titles, texts, max_chars=None):
    blocks = []
    for doc_id in doc_ids:
        if doc_id < 0 or doc_id >= len(texts):
            continue
        block = "{}\n{}".format(titles[doc_id], texts[doc_id])
        blocks.append(block.strip())
    context = "\n\n".join([b for b in blocks if b])
    if max_chars is not None and len(context) > max_chars:
        context = context[:max_chars].rsplit(" ", 1)[0]
    return context


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return str(ROOT_DIR / path)

def log_dataset_info(data_file, retrieval_file, pred_file, examples, retrieval_map):
    first_id = None
    if examples:
        first_id = examples[0].get("id")
    print("[data] data_file={}".format(data_file))
    print("[data] retrieval_file={} entries={}".format(retrieval_file, len(retrieval_map)))
    print("[data] pred_file={}".format(pred_file))
    print("[data] examples={} first_id={}".format(len(examples), first_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="llm_data/hotpot_dev_distractor_v1.jsonl")
    parser.add_argument("--retrieval_file", required=True)
    parser.add_argument("--corpus_file", default="rag_corpus/docs.jsonl")
    parser.add_argument("--pred_file", default="predictions/qwen3-4b_rag_dev.jsonl")
    parser.add_argument("--metrics_file", default=None)
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_context_chars", type=int, default=4000)
    parser.add_argument("--context_k", type=int, default=4)
    parser.add_argument("--evidence_sentences", type=int, default=0)
    parser.add_argument("--evidence_per_doc", type=int, default=0)
    parser.add_argument("--evidence_min_len", type=int, default=20)
    parser.add_argument("--use_retrieval_evidence", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--metrics_log", default=None)
    args = parser.parse_args()

    args.data_file = resolve_path(args.data_file)
    args.retrieval_file = resolve_path(args.retrieval_file)
    args.corpus_file = resolve_path(args.corpus_file)
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

    titles, texts = load_corpus(args.corpus_file)
    retrieval_map = load_retrieval(args.retrieval_file)

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

    examples = list(iter_examples(args.data_file, limit=args.limit))
    log_dataset_info(args.data_file, args.retrieval_file, args.pred_file, examples, retrieval_map)
    use_chat_template = not args.no_chat_template
    start_time = time.time()
    em_sum = 0.0
    f1_sum = 0.0
    gold_cnt = 0
    predicted_total = len(existing_ids)
    processed_total = 0
    model_seconds = 0.0
    input_tokens_total = 0
    output_tokens_total = 0

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
            "model_seconds": model_seconds,
            "input_tokens": input_tokens_total,
            "output_tokens": output_tokens_total,
            "seconds": time.time() - start_time,
        }
        write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)

    with open(args.pred_file, mode, encoding="utf-8") as out_f, torch.inference_mode():
        new_preds = 0
        for i in tqdm(range(0, len(examples), args.batch_size)):
            batch = examples[i : i + args.batch_size]
            prompts = []
            kept = []
            for ex in batch:
                ex_id = str(ex.get("id"))
                if ex_id in existing_ids:
                    continue
                retrieved = retrieval_map.get(ex_id, {})
                if isinstance(retrieved, dict):
                    doc_ids = retrieved.get("top_ids", [])[: args.context_k]
                    retrieval_evidence = (retrieved.get("evidence_context") or "").strip()
                else:
                    doc_ids = (
                        list(retrieved)[: args.context_k]
                        if isinstance(retrieved, list)
                        else []
                    )
                    retrieval_evidence = ""
                if args.use_retrieval_evidence and retrieval_evidence:
                    context = retrieval_evidence
                elif args.evidence_sentences > 0:
                    selected = select_evidence_sentences(
                        ex.get("question", ""),
                        doc_ids,
                        titles,
                        texts,
                        max_sentences=args.evidence_sentences,
                        per_doc=args.evidence_per_doc,
                        min_len=args.evidence_min_len,
                    )
                    context = build_evidence_context(
                        selected, max_chars=args.max_context_chars
                    )
                else:
                    context = build_context_from_ids(
                        doc_ids, titles, texts, max_chars=args.max_context_chars
                    )
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
            }
            model_start = time.time()
            outputs = model.generate(**inputs, **gen_kwargs)
            model_seconds += time.time() - model_start
            for j, ex in enumerate(kept):
                input_len = int(inputs["attention_mask"][j].sum().item())
                generated = outputs[j][input_len:]
                output_len = int(generated.shape[0])
                text = tokenizer.decode(generated, skip_special_tokens=True)
                pred = extract_answer(text)
                gold = ex.get("answer")
                if gold is not None:
                    em_sum += 1.0 if exact_match_score(pred, gold) else 0.0
                    f1_sum += f1_score(pred, gold)[0]
                    gold_cnt += 1
                out_f.write(
                    json.dumps(
                        {
                            "id": ex.get("id"),
                            "prediction": pred,
                            "input_tokens": input_len,
                            "output_tokens": output_len,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                predicted_total += 1
                processed_total += 1
                input_tokens_total += input_len
                output_tokens_total += output_len
                new_preds += 1
            out_f.flush()
            if args.checkpoint_every > 0 and new_preds >= args.checkpoint_every:
                avg_div = processed_total or 1
                metrics = {
                    "exact_match": 100.0 * em_sum / gold_cnt if gold_cnt else 0.0,
                    "f1": 100.0 * f1_sum / gold_cnt if gold_cnt else 0.0,
                    "total": gold_cnt,
                    "predicted": predicted_total,
                    "model_seconds": model_seconds,
                    "input_tokens": input_tokens_total,
                    "output_tokens": output_tokens_total,
                    "avg_latency_sec": (time.time() - start_time) / avg_div,
                    "avg_model_latency_sec": model_seconds / avg_div,
                    "avg_input_tokens": input_tokens_total / avg_div,
                    "avg_output_tokens": output_tokens_total / avg_div,
                    "seconds": time.time() - start_time,
                }
                write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)
                new_preds = 0

    elapsed = time.time() - start_time
    avg_div = processed_total or 1
    metrics = {
        "exact_match": 100.0 * em_sum / gold_cnt if gold_cnt else 0.0,
        "f1": 100.0 * f1_sum / gold_cnt if gold_cnt else 0.0,
        "total": gold_cnt,
        "predicted": predicted_total,
        "model_seconds": model_seconds,
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
        "avg_latency_sec": elapsed / avg_div,
        "avg_model_latency_sec": model_seconds / avg_div,
        "avg_input_tokens": input_tokens_total / avg_div,
        "avg_output_tokens": output_tokens_total / avg_div,
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
