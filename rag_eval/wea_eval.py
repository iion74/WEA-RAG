"""Selective Web Evidence Augmentation (WEA) evaluation."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
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

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "by",
    "and",
    "or",
    "with",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "at",
    "from",
    "into",
    "about",
    "than",
    "then",
    "but",
    "if",
    "so",
    "not",
    "no",
    "yes",
}

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CAPITAL_TERM_RE = re.compile(r"(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)")
TIME_SENSITIVE_RE = re.compile(
    r"\b("
    r"today|yesterday|tomorrow|currently|current|latest|recent|recently|up[- ]to[- ]date|"
    r"as of|this year|last year|next year|this month|last month|next month|"
    r"ongoing|still|now|presently"
    r")\b",
    flags=re.IGNORECASE,
)
NUMERIC_QUESTION_RE = re.compile(
    r"\b(how many|how much|what year|what date|when|how long|how far|how old|how tall|how high)\b",
    flags=re.IGNORECASE,
)
REFUSAL_PHRASES = (
    "cannot determine",
    "can't determine",
    "not enough information",
    "insufficient information",
    "cannot be determined",
    "unable to determine",
    "no answer",
    "unknown",
)
YESNO_QUESTION_RE = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|would|will|has|have|had|"
    r"should|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|can't|couldn't|"
    r"won't|wouldn't|hasn't|haven't|hadn't|shouldn't)\b",
    flags=re.IGNORECASE,
)
NUMERIC_ANSWER_HINT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    flags=re.IGNORECASE,
)
YEAR_TOKEN_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")

try:
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover
    faiss = None
    CrossEncoder = None
    SentenceTransformer = None


def tokenize(text):
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        if len(token) >= 2
    ]


def filter_stopwords(tokens):
    return {t for t in tokens if t not in STOPWORDS}


def split_sentences(text):
    if not text:
        return []
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    sentences = SENT_SPLIT_RE.split(compact)
    return [s.strip() for s in sentences if s.strip()]


def extract_capitalized_terms(text):
    if not text:
        return []
    return [term.strip() for term in CAPITAL_TERM_RE.findall(text) if term.strip()]


def is_time_sensitive_question(question):
    if not question:
        return False
    return TIME_SENSITIVE_RE.search(question) is not None


def build_web_queries(question, missing_terms, strategy="question_missing"):
    question = (question or "").strip()
    tail_terms = " ".join(sorted(missing_terms or [])[:5]).strip()
    question_missing = "{} {}".format(question, tail_terms).strip()

    if strategy == "question_only":
        queries = [question]
    elif strategy == "question_then_missing":
        queries = [question, question_missing]
    else:
        queries = [question_missing]

    deduped = []
    seen = set()
    for q in queries:
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(q)
    return deduped


def decide_web_needed(
    question,
    rag_context,
    coverage_ratio,
    missing_terms,
    gate_mode,
    coverage_threshold,
    min_missing_terms,
    force_time_sensitive,
):
    no_rag_context = not rag_context
    low_coverage = coverage_ratio < coverage_threshold
    missing_heavy = len(missing_terms) >= min_missing_terms
    time_sensitive = is_time_sensitive_question(question)

    reasons = []
    if no_rag_context:
        reasons.append("no_rag_context")
    if low_coverage:
        reasons.append("low_coverage")
    if missing_heavy:
        reasons.append("missing_terms")
    if time_sensitive:
        reasons.append("time_sensitive")

    if gate_mode == "never":
        return False, reasons
    if gate_mode == "always":
        return True, reasons
    if gate_mode == "coverage":
        return no_rag_context or low_coverage, reasons
    if gate_mode == "heuristic":
        use_web = False
        if no_rag_context:
            use_web = True
        elif low_coverage and missing_heavy:
            use_web = True
        elif force_time_sensitive and time_sensitive and (low_coverage or missing_heavy):
            use_web = True
        return use_web, reasons
    raise ValueError("unknown web gate mode: {}".format(gate_mode))


def decide_wea_refine_needed(
    question,
    rag_answer,
    rag_context,
    coverage_ratio,
    missing_terms,
    min_missing_terms,
):
    rag_answer = (rag_answer or "").strip()
    if not rag_answer:
        return True, "rag_empty"
    if contains_refusal_phrase(rag_answer):
        return True, "rag_refusal"

    rag_score, rag_detail = score_answer_candidate(question, rag_answer, rag_context)
    low_coverage = coverage_ratio < 0.80
    missing_heavy = len(missing_terms or []) >= max(1, min_missing_terms)
    weak_support = (
        float(rag_detail.get("token_support", 0.0)) < 0.20
        and not bool(rag_detail.get("phrase_hit"))
    )
    numeric_need = (
        NUMERIC_QUESTION_RE.search(question or "") is not None
        and NUMERIC_ANSWER_HINT_RE.search(rag_answer) is None
        and rag_answer.lower() not in {"yes", "no"}
    )

    if numeric_need and (low_coverage or missing_heavy):
        return True, "numeric_missing"
    if rag_score < 0.45 and (low_coverage or missing_heavy):
        return True, "rag_low_score"
    if weak_support and (low_coverage or missing_heavy):
        return True, "rag_weak_support"
    if low_coverage and missing_heavy:
        return True, "rag_missing_clues"
    return False, "rag_sufficient"


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


def build_slot_refine_prompt(
    tokenizer,
    question,
    rag_answer,
    rag_context,
    web_context,
    missing_terms,
    system_prompt,
    use_chat_template,
    wea_fusion_mode,
):
    missing = ", ".join((missing_terms or [])[:8]).strip()
    if not missing:
        missing = "N/A"
    if wea_fusion_mode == "merge":
        task_block = (
            "Task:\n"
            "- Use RAG evidence first; use web evidence only to fill missing clues.\n"
            "- Output the shortest exact answer span copied from evidence (or yes/no).\n"
            "- Do not paraphrase and do not add unsupported words.\n"
            "- If web evidence is weak or conflicting, keep the RAG draft unchanged.\n\n"
            "Final Answer:"
        )
    else:
        task_block = (
            "Task:\n"
            "- Keep the RAG draft answer unless web evidence clearly fills missing clues.\n"
            "- If changing, output the shortest exact span copied from evidence.\n"
            "- If evidence is weak/unclear, return the RAG draft answer unchanged.\n\n"
            "Final Answer:"
        )
    user_prompt = (
        "Question:\n{question}\n\n"
        "RAG Draft Answer:\n{rag_answer}\n\n"
        "RAG Evidence:\n{rag_context}\n\n"
        "Web Evidence (for missing clues):\n{web_context}\n\n"
        "Missing Clues:\n{missing}\n\n"
        "{task_block}"
    ).format(
        question=question or "",
        rag_answer=(rag_answer or "").strip() or "(empty)",
        rag_context=rag_context or "",
        web_context=web_context or "",
        missing=missing,
        task_block=task_block,
    )
    if use_chat_template and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "{}\n\n{}".format(system_prompt, user_prompt)


def build_compose_fusion_prompt(
    tokenizer,
    question,
    rag_answer,
    wea_answer,
    rag_context,
    web_context,
    missing_terms,
    system_prompt,
    use_chat_template,
):
    missing = ", ".join((missing_terms or [])[:8]).strip()
    if not missing:
        missing = "N/A"
    user_prompt = (
        "Question:\n{question}\n\n"
        "RAG Draft Answer:\n{rag_answer}\n\n"
        "WEA Draft Answer:\n{wea_answer}\n\n"
        "RAG Evidence:\n{rag_context}\n\n"
        "Web Evidence:\n{web_context}\n\n"
        "Missing Clues:\n{missing}\n\n"
        "Task:\n"
        "- Compose one final answer by using both drafts when helpful.\n"
        "- Prefer exact spans grounded in evidence over paraphrases.\n"
        "- Drop unsupported details and keep the shortest valid factoid answer.\n"
        "- Output only a short final answer phrase (or yes/no).\n\n"
        "Final Answer:"
    ).format(
        question=question or "",
        rag_answer=(rag_answer or "").strip() or "(empty)",
        wea_answer=(wea_answer or "").strip() or "(empty)",
        rag_context=rag_context or "",
        web_context=web_context or "",
        missing=missing,
    )
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


def dynamic_wea_margin(rag_score, base_margin):
    if rag_score < 0.35:
        return min(base_margin, 0.04)
    if rag_score < 0.55:
        return min(base_margin, 0.08)
    if rag_score < 0.75:
        return min(base_margin, 0.12)
    return base_margin


def _pick_context_supported_span(answer, context, max_words=7):
    answer = (answer or "").strip()
    context_lower = (context or "").lower()
    if not answer or not context_lower:
        return ""
    tokens = answer.split()
    if not tokens:
        return ""
    limit = min(max_words, len(tokens))
    fallback = ""
    for n in range(limit, 0, -1):
        for i in range(0, len(tokens) - n + 1):
            cand = " ".join(tokens[i : i + n]).strip(" ,.;:!?()[]{}\"'")
            if len(cand) < 2:
                continue
            if cand.lower() in context_lower:
                if re.search(r"[A-Z0-9]", cand):
                    return cand
                if not fallback:
                    fallback = cand
    return fallback


def normalize_factoid_answer(answer, question="", context="", max_words=7):
    answer = (answer or "").strip()
    if not answer:
        return ""

    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.strip(" \"'`")
    lowered = answer.lower().lstrip(" \"'([{")
    if lowered.startswith("yes") and (len(lowered) == 3 or not lowered[3].isalpha()):
        return "yes"
    if lowered.startswith("no") and (len(lowered) == 2 or not lowered[2].isalpha()):
        return "no"

    if NUMERIC_QUESTION_RE.search(question or ""):
        years = YEAR_TOKEN_RE.findall(answer)
        if years:
            if context:
                for y in years:
                    if y in context:
                        return y
            return years[0]

    if len(answer.split()) > max_words:
        span = _pick_context_supported_span(answer, context, max_words=max_words)
        if span:
            answer = span
        else:
            answer = re.split(r"[.;:!?]|,\s+|\s+-\s+|\s+\(", answer, 1)[0].strip()

    words = answer.split()
    if len(words) > max_words:
        answer = " ".join(words[:max_words]).strip()

    answer = answer.strip(" ,.;:!?")
    return answer


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


def score_sentence_overlap(question_tokens, sentence, question_entities):
    sent_tokens = tokenize(sentence)
    if not sent_tokens:
        return 0.0
    overlap = len(set(question_tokens).intersection(sent_tokens))
    if overlap == 0:
        return 0.0
    score = overlap / (1.0 + 0.35 * len(sent_tokens))
    sent_lower = sentence.lower()
    entity_hits = 0
    for ent in question_entities:
        if ent.lower() in sent_lower:
            entity_hits += 1
    if entity_hits:
        score += 1.5 * entity_hits
    return score


def select_evidence_sentences(
    question,
    doc_ids,
    titles,
    texts,
    max_sentences=0,
    per_doc=0,
    min_len=20,
    mode="overlap",
    reranker=None,
    rerank_prefilter=200,
):
    if max_sentences <= 0:
        return []
    question_tokens = tokenize(question)
    question_entities = extract_capitalized_terms(question)
    candidates = []
    for doc_id in doc_ids:
        if doc_id < 0 or doc_id >= len(texts):
            continue
        title = titles[doc_id]
        for sent in split_sentences(texts[doc_id]):
            if len(sent) < min_len:
                continue
            score = score_sentence_overlap(question_tokens, sent, question_entities)
            if score <= 0:
                continue
            candidates.append((score, doc_id, title, sent))
    if not candidates:
        return []
    if mode == "rerank" and reranker is not None:
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:rerank_prefilter]
        pairs = [(question, "{}\n{}".format(c[2], c[3])) for c in candidates]
        scores = reranker.predict(pairs, batch_size=16)
        rescored = []
        for (orig, score) in zip(candidates, scores):
            rescored.append((float(score), orig[1], orig[2], orig[3]))
        candidates = rescored
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    used = set()
    if per_doc > 0:
        per_doc_counts = {}
        for score, doc_id, title, sent in candidates:
            if per_doc_counts.get(doc_id, 0) >= per_doc:
                continue
            key = (doc_id, sent)
            if key in used:
                continue
            used.add(key)
            per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
            selected.append((score, doc_id, title, sent))
            if len(selected) >= max_sentences:
                break
    if len(selected) < max_sentences:
        for score, doc_id, title, sent in candidates:
            key = (doc_id, sent)
            if key in used:
                continue
            used.add(key)
            selected.append((score, doc_id, title, sent))
            if len(selected) >= max_sentences:
                break
    return selected


def build_evidence_context(selected, max_chars=None):
    if not selected:
        return ""
    blocks = []
    for _, _, title, sent in selected:
        blocks.append("{}: {}".format(title, sent))
    context = "\n".join(blocks)
    if max_chars is not None and len(context) > max_chars:
        context = context[:max_chars].rsplit(" ", 1)[0]
    return context


def load_web_cache(cache_file):
    cache = {}
    if not cache_file or not os.path.exists(cache_file):
        return cache
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            query = record.get("query")
            if query:
                cache[query] = record.get("results", [])
    return cache


def append_web_cache(cache_file, query, results):
    if not cache_file:
        return
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": query, "results": results}) + "\n")


def load_serpapi_key():
    key = os.environ.get("SERPAPI_API_KEY")
    if key:
        return key
    env_candidates = [ROOT_DIR / ".env"]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("SERPAPI_API_KEY="):
                    continue
                return line.split("=", 1)[1].strip()
    return None


def search_serpapi(query, num_results=5, timeout=30):
    key = load_serpapi_key()
    if not key:
        raise RuntimeError("SERPAPI_API_KEY is not set")
    params = {
        "engine": "google",
        "q": query,
        "api_key": key,
        "num": num_results,
    }
    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return results


def build_web_context(results, max_chars=None):
    blocks = []
    for res in results:
        title = res.get("title", "")
        snippet = res.get("snippet", "")
        block = "{}\n{}".format(title, snippet).strip()
        if block:
            blocks.append(block)
    context = "\n\n".join(blocks)
    if max_chars is not None and len(context) > max_chars:
        context = context[:max_chars].rsplit(" ", 1)[0]
    return context


def merge_doc_ids(primary, extra, max_total=None):
    seen = set()
    merged = []
    for doc_id in list(primary) + list(extra):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(doc_id)
        if max_total is not None and len(merged) >= max_total:
            break
    return merged


def dense_search(index, embedder, query, top_k):
    embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores, ids = index.search(embedding, top_k)
    return ids[0].tolist(), scores[0].tolist()


def init_tool_retriever(index_dir, embed_model, rerank_model, device, use_rerank):
    if faiss is None or SentenceTransformer is None:
        raise RuntimeError("faiss or sentence-transformers not available for tool retrieval")
    index_path = os.path.join(index_dir, "dense.index")
    index = faiss.read_index(index_path)
    embedder = SentenceTransformer(embed_model, device=device)
    reranker = None
    if use_rerank:
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder not available for tool rerank")
        reranker = CrossEncoder(rerank_model, device=device)
    return index, embedder, reranker


def tool_retrieve(
    query,
    index,
    embedder,
    reranker,
    titles,
    texts,
    dense_k=100,
    rerank_k=50,
    top_k=4,
    batch_size=16,
):
    ids, _ = dense_search(index, embedder, query, dense_k)
    candidate_ids = [idx for idx in ids if idx >= 0]
    if not candidate_ids:
        return []
    if reranker is None:
        return candidate_ids[:top_k]
    candidate_ids = candidate_ids[:rerank_k]
    pairs = [
        (query, "{}\n{}".format(titles[idx], texts[idx]))
        for idx in candidate_ids
    ]
    rerank_scores = reranker.predict(pairs, batch_size=batch_size)
    reranked = sorted(
        zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True
    )
    return [idx for idx, _ in reranked[:top_k]]


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


def compute_coverage(question, context):
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return 1.0, set(), set()
    filtered_q = filter_stopwords(q_tokens)
    if not filtered_q:
        filtered_q = q_tokens
    c_tokens = set(tokenize(context))
    matched = filtered_q.intersection(c_tokens)
    missing = filtered_q - c_tokens
    for ent in extract_capitalized_terms(question):
        ent_tokens = set(tokenize(ent))
        if not ent_tokens:
            continue
        if not ent_tokens.issubset(c_tokens):
            missing.update(ent_tokens)
    ratio = len(matched) / len(filtered_q) if filtered_q else 1.0
    return ratio, matched, missing


def contains_refusal_phrase(answer):
    lowered = (answer or "").strip().lower()
    if not lowered:
        return True
    for phrase in REFUSAL_PHRASES:
        if phrase in lowered:
            return True
    return False


def score_answer_candidate(question, answer, context):
    answer = (answer or "").strip()
    context = context or ""
    q_text = question or ""

    details = {
        "empty": False,
        "refusal": False,
        "token_support": 0.0,
        "phrase_hit": False,
        "numeric_hint": False,
        "word_count": len(answer.split()) if answer else 0,
    }
    if not answer:
        details["empty"] = True
        return -2.0, details

    score = 0.0
    answer_tokens = list(filter_stopwords(tokenize(answer)))
    context_tokens = set(tokenize(context))
    if answer_tokens:
        hit = sum(1 for tok in answer_tokens if tok in context_tokens)
        token_support = hit / max(1, len(answer_tokens))
    else:
        token_support = 0.0
    details["token_support"] = round(token_support, 4)
    score += 1.6 * token_support

    answer_lower = answer.lower()
    context_lower = context.lower()
    if len(answer_lower) >= 3 and answer_lower in context_lower:
        details["phrase_hit"] = True
        score += 0.6

    if contains_refusal_phrase(answer):
        details["refusal"] = True
        score -= 1.0

    if NUMERIC_QUESTION_RE.search(q_text):
        has_numeric_hint = NUMERIC_ANSWER_HINT_RE.search(answer) is not None
        details["numeric_hint"] = bool(has_numeric_hint)
        if has_numeric_hint:
            score += 0.25
        elif answer_lower not in {"yes", "no"}:
            score -= 0.25

    if answer_lower in {"yes", "no"} and not YESNO_QUESTION_RE.search(q_text):
        score -= 0.1

    word_count = details["word_count"]
    if word_count > 8:
        score -= min(0.7, 0.06 * (word_count - 8))

    return score, details


def detect_answer_conflicts(rag_answer, wea_answer):
    rag_answer = (rag_answer or "").strip()
    wea_answer = (wea_answer or "").strip()
    rag_years = set(YEAR_TOKEN_RE.findall(rag_answer))
    wea_years = set(YEAR_TOKEN_RE.findall(wea_answer))
    year_conflict = bool(rag_years and wea_years and rag_years != wea_years)

    rag_ents = {t.lower() for t in extract_capitalized_terms(rag_answer)}
    wea_ents = {t.lower() for t in extract_capitalized_terms(wea_answer)}
    short_answer = len(rag_answer.split()) <= 5 and len(wea_answer.split()) <= 5
    entity_conflict = bool(short_answer and rag_ents and wea_ents and rag_ents.isdisjoint(wea_ents))
    return year_conflict, entity_conflict


def choose_dual_answer(
    question,
    rag_answer,
    wea_answer,
    rag_context,
    wea_context,
    score_margin,
    wea_min_score,
    wea_min_token_support,
    require_wea_context_match,
    rag_lock_score,
):
    rag_answer = normalize_factoid_answer(rag_answer, question=question, context=rag_context)
    wea_answer = normalize_factoid_answer(wea_answer, question=question, context=wea_context)
    rag_score, rag_detail = score_answer_candidate(question, rag_answer, rag_context)
    wea_score, wea_detail = score_answer_candidate(question, wea_answer, wea_context)

    selected = "rag"
    reason = "default_rag_safe"

    if not rag_answer and wea_answer:
        selected = "wea"
        reason = "rag_empty"
    elif rag_answer and not wea_answer:
        selected = "rag"
        reason = "wea_empty"
    elif not rag_answer and not wea_answer:
        selected = "rag"
        reason = "both_empty"
    else:
        wea_token_support = float(wea_detail.get("token_support", 0.0))
        wea_phrase_hit = bool(wea_detail.get("phrase_hit"))
        wea_context_ok = (
            (wea_phrase_hit or wea_token_support >= wea_min_token_support)
            if require_wea_context_match
            else True
        )
        wea_quality_ok = (
            (not wea_detail["refusal"])
            and wea_score >= wea_min_score
            and wea_token_support >= wea_min_token_support
            and wea_context_ok
        )
        rag_locked = (
            bool(rag_answer)
            and (not rag_detail.get("empty", False))
            and (not rag_detail.get("refusal", False))
            and rag_score >= rag_lock_score
        )

        numeric_block = False
        if NUMERIC_QUESTION_RE.search(question or ""):
            rag_num = bool(NUMERIC_ANSWER_HINT_RE.search(rag_answer or ""))
            wea_num = bool(NUMERIC_ANSWER_HINT_RE.search(wea_answer or ""))
            if rag_num and (not wea_num) and rag_score >= wea_score:
                numeric_block = True

        yesno_conflict = (
            rag_answer.lower() in {"yes", "no"}
            and wea_answer.lower() in {"yes", "no"}
            and rag_answer.lower() != wea_answer.lower()
        )
        year_conflict, entity_conflict = detect_answer_conflicts(rag_answer, wea_answer)

        if wea_detail["refusal"] and not rag_detail["refusal"]:
            reason = "wea_refusal"
        elif not wea_quality_ok:
            if wea_score < wea_min_score:
                reason = "guard_wea_min_score"
            elif wea_token_support < wea_min_token_support:
                reason = "guard_wea_min_token_support"
            elif require_wea_context_match and not wea_context_ok:
                reason = "guard_wea_no_context_match"
            else:
                reason = "guard_wea_quality"
        elif wea_score < rag_score + dynamic_wea_margin(rag_score, score_margin):
            reason = "wea_no_margin"
        elif numeric_block:
            reason = "numeric_prefer_rag"
        elif yesno_conflict:
            reason = "yesno_conflict_rag_lock"
        elif year_conflict and rag_locked and rag_score >= (wea_score - 0.05):
            reason = "year_conflict_rag_lock"
        elif entity_conflict and rag_locked and rag_score >= (wea_score - 0.05):
            reason = "entity_conflict_rag_lock"
        elif rag_locked:
            reason = "guard_rag_lock"
        else:
            selected = "wea"
            reason = "wea_override_rag_weak"

    final_answer = wea_answer if selected == "wea" else rag_answer
    return {
        "answer": final_answer,
        "selected": selected,
        "reason": reason,
        "rag_score": round(rag_score, 4),
        "wea_score": round(wea_score, 4),
        "score_delta_wea_minus_rag": round(wea_score - rag_score, 4),
        "rag_detail": rag_detail,
        "wea_detail": wea_detail,
    }


def generate_batch_predictions(model, tokenizer, prompts, args):
    if not prompts:
        return [], 0.0, [], []
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    ).to(model.device)
    use_cache = "minicpm" not in args.model.lower()
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": max(args.temperature, 1e-5),
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": use_cache,
    }
    model_start = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - model_start
    prompt_len = int(inputs["input_ids"].shape[1])

    predictions = []
    input_lens = []
    output_lens = []
    for j in range(len(prompts)):
        input_len = int(inputs["attention_mask"][j].sum().item())
        generated = outputs[j][prompt_len:]
        output_len = int(generated.shape[0])
        text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(extract_answer(text))
        input_lens.append(input_len)
        output_lens.append(output_len)

    return predictions, elapsed, input_lens, output_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="llm_data/hotpot_dev_distractor_v1.jsonl")
    parser.add_argument("--retrieval_file", required=True)
    parser.add_argument("--corpus_file", default="rag_corpus/docs.jsonl")
    parser.add_argument("--pred_file", default="predictions/qwen3-4b_wea_rag_dev.jsonl")
    parser.add_argument("--metrics_file", default=None)
    parser.add_argument(
        "--model",
        default=os.environ.get("WEA_RAG_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_context_chars", type=int, default=4000)
    parser.add_argument("--max_web_chars", type=int, default=2000)
    parser.add_argument("--max_total_chars", type=int, default=6000)
    parser.add_argument("--context_k", type=int, default=4)
    parser.add_argument("--coverage_threshold", type=float, default=0.6)
    parser.add_argument("--web_top_k", type=int, default=5)
    parser.add_argument("--web_cache_file", default="rag_cache/web_search_cache.jsonl")
    parser.add_argument(
        "--max_web_api_calls",
        type=int,
        default=0,
        help="stop evaluation once this many SerpAPI calls are used (<=0 disables)",
    )
    parser.add_argument(
        "--web_gate_mode",
        default="coverage",
        choices=["coverage", "heuristic", "always", "never"],
        help="policy for deciding whether web augmentation is needed",
    )
    parser.add_argument(
        "--web_min_missing_terms",
        type=int,
        default=3,
        help="heuristic gate: minimum missing term count considered high-risk",
    )
    parser.add_argument(
        "--web_force_for_time_sensitive",
        action="store_true",
        help="heuristic gate: allow web when question looks time-sensitive",
    )
    parser.add_argument(
        "--web_query_strategy",
        default="question_missing",
        choices=["question_missing", "question_only", "question_then_missing"],
        help="how to build web cache/API query from question and missing terms",
    )
    parser.add_argument(
        "--web_try_all_queries_on_miss",
        action="store_true",
        help="when using multiple web query candidates, try next candidates if earlier result is empty",
    )
    parser.add_argument("--web_cache_only", action="store_true")
    parser.add_argument(
        "--disable_web_cache",
        action="store_true",
        help="do not read/write web cache; always hit API for uncached queries",
    )
    parser.add_argument(
        "--stop_on_web_error",
        action="store_true",
        help="stop evaluation immediately when a live web API call fails",
    )
    parser.add_argument("--disable_web", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--metrics_log", default=None)
    parser.add_argument("--evidence_sentences", type=int, default=0)
    parser.add_argument("--evidence_per_doc", type=int, default=0)
    parser.add_argument("--evidence_min_len", type=int, default=20)
    parser.add_argument("--use_retrieval_evidence", action="store_true")
    parser.add_argument(
        "--evidence_mode", default="overlap", choices=["overlap", "rerank"]
    )
    parser.add_argument(
        "--evidence_rerank_model", default="BAAI/bge-reranker-large"
    )
    parser.add_argument("--evidence_rerank_prefilter", type=int, default=200)
    parser.add_argument(
        "--tool_retrieval", default="none", choices=["none", "dense", "dense_rerank"]
    )
    parser.add_argument("--tool_index_dir", default="rag_index")
    parser.add_argument("--tool_embed_model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--tool_rerank_model", default="BAAI/bge-reranker-large")
    parser.add_argument("--tool_dense_k", type=int, default=100)
    parser.add_argument("--tool_rerank_k", type=int, default=50)
    parser.add_argument("--tool_top_k", type=int, default=4)
    parser.add_argument("--tool_batch_size", type=int, default=16)
    parser.add_argument("--tool_device", default="cuda")
    parser.add_argument("--tool_always", action="store_true")
    parser.add_argument("--no_rag", action="store_true")
    parser.add_argument(
        "--dual_answer_select",
        action="store_true",
        help="for web-used samples, generate both RAG and WEA answers and select one",
    )
    parser.add_argument(
        "--dual_select_margin",
        type=float,
        default=0.15,
        help="minimum score gap to switch from the RAG answer to the WEA answer",
    )
    parser.add_argument(
        "--dual_wea_min_score",
        type=float,
        default=0.0,
        help="minimum WEA answer score required before selecting the WEA answer",
    )
    parser.add_argument(
        "--dual_wea_min_token_support",
        type=float,
        default=0.0,
        help="minimum WEA token-support ratio required before selecting the WEA answer",
    )
    parser.add_argument(
        "--dual_require_context_match",
        action="store_true",
        help="require the WEA answer to be grounded in context",
    )
    parser.add_argument(
        "--dual_rag_lock_score",
        type=float,
        default=0.75,
        help="keep RAG answer by default when RAG score is above this threshold",
    )
    parser.add_argument(
        "--dual_slot_refine",
        action="store_true",
        help="generate the RAG answer first, then run WEA as missing-clue refinement",
    )
    parser.add_argument(
        "--wea_fusion_mode",
        default="select",
        choices=["select", "merge", "compose"],
        help=(
            "select: choose between the RAG and WEA answers with guard rules; "
            "merge: for slot-refine samples, use the WEA-refined answer as a candidate; "
            "compose: generate an extra answer from both RAG and WEA drafts"
        ),
    )
    parser.add_argument(
        "--merge_force_all",
        action="store_true",
        help=(
            "in merge+slot_refine mode, always use the WEA answer when non-empty "
            "(bypass guard-based dual selection)"
        ),
    )
    parser.add_argument(
        "--merge_force_min_wea_score",
        type=float,
        default=None,
        help="in merge_force_all mode, fall back to RAG below this WEA score",
    )
    parser.add_argument(
        "--merge_force_min_delta",
        type=float,
        default=None,
        help="in merge_force_all mode, fall back to RAG below this score difference",
    )
    parser.add_argument(
        "--merge_force_use_conflict_guards",
        action="store_true",
        help="in merge_force_all mode, enable additional conflict/weak-support guards",
    )
    args = parser.parse_args()

    args.data_file = resolve_path(args.data_file)
    args.retrieval_file = resolve_path(args.retrieval_file)
    args.corpus_file = resolve_path(args.corpus_file)
    args.pred_file = resolve_path(args.pred_file)
    args.metrics_file = resolve_path(args.metrics_file)
    args.metrics_log = resolve_path(args.metrics_log)
    args.web_cache_file = resolve_path(args.web_cache_file)
    args.tool_index_dir = resolve_path(args.tool_index_dir)

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
    web_cache = {} if args.disable_web_cache else load_web_cache(args.web_cache_file)
    evidence_reranker = None
    if args.evidence_mode == "rerank" and args.evidence_sentences > 0:
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder not available for evidence rerank")
        evidence_reranker = CrossEncoder(args.evidence_rerank_model, device=args.tool_device)

    tool_index = tool_embedder = tool_reranker = None
    if args.tool_retrieval != "none":
        use_rerank = args.tool_retrieval == "dense_rerank"
        tool_index, tool_embedder, tool_reranker = init_tool_retriever(
            args.tool_index_dir,
            args.tool_embed_model,
            args.tool_rerank_model,
            args.tool_device,
            use_rerank,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    web_used_total = 0
    web_query_total = 0
    web_api_calls = 0
    web_cache_hits = 0
    web_gate_true_total = 0
    time_sensitive_questions = 0
    web_seconds = 0.0
    tool_used_total = 0
    tool_calls = 0
    tool_seconds = 0.0
    model_seconds = 0.0
    input_tokens_total = 0
    output_tokens_total = 0
    api_limit_reached = False
    web_error_reached = False
    web_error_count = 0
    web_last_error = ""
    web_last_error_query = ""
    dual_select_total = 0
    dual_select_wea = 0
    dual_select_rag = 0
    dual_select_fused = 0
    dual_switch_to_wea = 0
    dual_switch_to_rag = 0
    dual_switch_to_fused = 0
    dual_model_seconds = 0.0
    dual_input_tokens_total = 0
    dual_output_tokens_total = 0

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
            "max_web_api_calls": args.max_web_api_calls,
            "stopped_by_web_api_limit": api_limit_reached,
            "web_used": web_used_total,
            "web_queries": web_query_total,
            "web_api_calls": web_api_calls,
            "web_cache_hits": web_cache_hits,
            "web_cache_only": args.web_cache_only,
            "disable_web_cache": args.disable_web_cache,
            "stop_on_web_error": args.stop_on_web_error,
            "stopped_by_web_error": web_error_reached,
            "web_error_count": web_error_count,
            "web_last_error": web_last_error,
            "web_last_error_query": web_last_error_query,
            "web_gate_mode": args.web_gate_mode,
            "web_query_strategy": args.web_query_strategy,
            "web_try_all_queries_on_miss": args.web_try_all_queries_on_miss,
            "web_gate_true": web_gate_true_total,
            "time_sensitive_questions": time_sensitive_questions,
            "web_seconds": web_seconds,
            "tool_used": tool_used_total,
            "tool_calls": tool_calls,
            "tool_seconds": tool_seconds,
            "model_seconds": model_seconds,
            "dual_answer_select": args.dual_answer_select,
            "dual_slot_refine": args.dual_slot_refine,
            "wea_fusion_mode": args.wea_fusion_mode,
            "merge_force_all": args.merge_force_all,
            "merge_force_min_wea_score": args.merge_force_min_wea_score,
            "merge_force_min_delta": args.merge_force_min_delta,
            "merge_force_use_conflict_guards": args.merge_force_use_conflict_guards,
            "dual_select_margin": args.dual_select_margin,
            "dual_wea_min_score": args.dual_wea_min_score,
            "dual_wea_min_token_support": args.dual_wea_min_token_support,
            "dual_require_context_match": args.dual_require_context_match,
            "dual_rag_lock_score": args.dual_rag_lock_score,
            "dual_select_total": dual_select_total,
            "dual_select_wea": dual_select_wea,
            "dual_select_rag": dual_select_rag,
            "dual_select_fused": dual_select_fused,
            "dual_switch_to_wea": dual_switch_to_wea,
            "dual_switch_to_rag": dual_switch_to_rag,
            "dual_switch_to_fused": dual_switch_to_fused,
            "dual_model_seconds": dual_model_seconds,
            "dual_input_tokens": dual_input_tokens_total,
            "dual_output_tokens": dual_output_tokens_total,
            "input_tokens": input_tokens_total,
            "output_tokens": output_tokens_total,
            "seconds": time.time() - start_time,
        }
        write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)

    with open(args.pred_file, mode, encoding="utf-8") as out_f, torch.inference_mode():
        new_preds = 0
        for i in tqdm(range(0, len(examples), args.batch_size)):
            if args.max_web_api_calls > 0 and web_api_calls >= args.max_web_api_calls:
                api_limit_reached = True
                break
            if web_error_reached:
                break
            batch = examples[i : i + args.batch_size]
            prompts = []
            kept = []
            meta = []
            questions = []
            rag_contexts = []
            combined_contexts = []
            web_contexts = []
            for ex in batch:
                if args.max_web_api_calls > 0 and web_api_calls >= args.max_web_api_calls:
                    api_limit_reached = True
                    break
                if web_error_reached:
                    break
                ex_id = str(ex.get("id"))
                if ex_id in existing_ids:
                    continue
                question = ex.get("question", "")
                if args.no_rag:
                    doc_ids = []
                    retrieval_evidence = ""
                else:
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
                retrieval_evidence_used = bool(
                    args.use_retrieval_evidence and retrieval_evidence
                )
                if retrieval_evidence_used:
                    rag_context = retrieval_evidence
                elif args.evidence_sentences > 0:
                    selected = select_evidence_sentences(
                        question,
                        doc_ids,
                        titles,
                        texts,
                        max_sentences=args.evidence_sentences,
                        per_doc=args.evidence_per_doc,
                        min_len=args.evidence_min_len,
                        mode=args.evidence_mode,
                        reranker=evidence_reranker,
                        rerank_prefilter=args.evidence_rerank_prefilter,
                    )
                    rag_context = build_evidence_context(
                        selected, max_chars=args.max_context_chars
                    )
                else:
                    rag_context = build_context_from_ids(
                        doc_ids, titles, texts, max_chars=args.max_context_chars
                    )
                ratio, matched, missing = compute_coverage(question, rag_context)
                tool_ids = []
                tool_query = ""
                tool_time = 0.0
                use_tool = (
                    args.tool_retrieval != "none"
                    and (args.tool_always or args.no_rag or not rag_context or ratio < args.coverage_threshold)
                )
                if use_tool:
                    tool_query = question
                    if missing:
                        tool_query = "{} {}".format(tool_query, " ".join(sorted(missing)[:5]))
                    tool_start = time.time()
                    tool_ids = tool_retrieve(
                        tool_query,
                        tool_index,
                        tool_embedder,
                        tool_reranker,
                        titles,
                        texts,
                        dense_k=args.tool_dense_k,
                        rerank_k=args.tool_rerank_k,
                        top_k=args.tool_top_k,
                        batch_size=args.tool_batch_size,
                    )
                    tool_time = time.time() - tool_start
                    tool_seconds += tool_time
                    tool_calls += 1
                    if tool_ids:
                        tool_used_total += 1
                        doc_ids = merge_doc_ids(
                            doc_ids,
                            tool_ids,
                            max_total=args.context_k + args.tool_top_k,
                        )
                        if args.evidence_sentences > 0:
                            selected = select_evidence_sentences(
                                question,
                                doc_ids,
                                titles,
                                texts,
                                max_sentences=args.evidence_sentences,
                                per_doc=args.evidence_per_doc,
                                min_len=args.evidence_min_len,
                                mode=args.evidence_mode,
                                reranker=evidence_reranker,
                                rerank_prefilter=args.evidence_rerank_prefilter,
                            )
                            rag_context = build_evidence_context(
                                selected, max_chars=args.max_context_chars
                            )
                        else:
                            rag_context = build_context_from_ids(
                                doc_ids, titles, texts, max_chars=args.max_context_chars
                            )
                        ratio, matched, missing = compute_coverage(question, rag_context)
                use_web, web_gate_reasons = decide_web_needed(
                    question=question,
                    rag_context=rag_context,
                    coverage_ratio=ratio,
                    missing_terms=missing,
                    gate_mode=args.web_gate_mode,
                    coverage_threshold=args.coverage_threshold,
                    min_missing_terms=args.web_min_missing_terms,
                    force_time_sensitive=args.web_force_for_time_sensitive,
                )
                time_sensitive_flag = is_time_sensitive_question(question)
                if time_sensitive_flag:
                    time_sensitive_questions += 1
                if args.disable_web:
                    use_web = False
                    web_gate_reasons = list(web_gate_reasons) + ["disable_web_flag"]
                if use_web:
                    web_gate_true_total += 1
                defer_web_query = bool(args.dual_answer_select and args.dual_slot_refine)
                web_context = ""
                web_time = 0.0
                web_cache_hit = False
                queried_web = False
                query_used = ""
                candidate_queries = []
                if use_web:
                    candidate_queries = build_web_queries(
                        question, missing, strategy=args.web_query_strategy
                    )
                    if not candidate_queries:
                        candidate_queries = [(question or "").strip()]
                    query_used = candidate_queries[0]
                if use_web and (not defer_web_query):
                    web_query_total += 1
                    queried_web = True
                    web_start = time.time()
                    results = None
                    if not args.disable_web_cache:
                        for candidate_query in candidate_queries:
                            if candidate_query in web_cache:
                                results = web_cache[candidate_query]
                                query_used = candidate_query
                                web_cache_hit = True
                                break
                    if results is None:
                        if args.web_cache_only:
                            results = []
                        else:
                            web_call_failed = False
                            api_candidates = (
                                candidate_queries
                                if args.web_try_all_queries_on_miss
                                else candidate_queries[:1]
                            )
                            for cq in api_candidates:
                                query_used = cq
                                web_api_calls += 1
                                if args.max_web_api_calls > 0 and web_api_calls >= args.max_web_api_calls:
                                    api_limit_reached = True
                                try:
                                    results = search_serpapi(query_used, num_results=args.web_top_k)
                                except Exception as err:
                                    web_call_failed = True
                                    web_error_count += 1
                                    if args.stop_on_web_error:
                                        web_error_reached = True
                                        web_last_error = str(err)
                                        web_last_error_query = query_used
                                        print(
                                            "[web-stop] query='{}' error={}".format(
                                                query_used, web_last_error
                                            )
                                        )
                                        results = []
                                        break
                                    results = []
                                if (not web_call_failed) and (not args.disable_web_cache):
                                    web_cache[query_used] = results
                                    append_web_cache(args.web_cache_file, query_used, results)
                                if results:
                                    break
                                web_call_failed = False
                            if results is None:
                                results = []
                    if web_error_reached:
                        break
                    web_time = time.time() - web_start
                    web_seconds += web_time
                    if web_cache_hit:
                        web_cache_hits += 1
                    web_context = build_web_context(results, max_chars=args.max_web_chars)
                    if web_context:
                        web_used_total += 1
                combined = rag_context
                if web_context:
                    combined = "{}\n\nWeb Search:\n{}".format(rag_context, web_context).strip()
                if args.max_total_chars is not None and len(combined) > args.max_total_chars:
                    combined = combined[: args.max_total_chars].rsplit(" ", 1)[0]
                # Generate the RAG draft first, then run WEA only when needed.
                prompt_context = combined
                if args.dual_answer_select and args.dual_slot_refine:
                    prompt_context = rag_context
                prompt = build_prompt(
                    tokenizer,
                    question,
                    prompt_context,
                    args.system_prompt,
                    use_chat_template,
                )
                prompts.append(prompt)
                kept.append(ex)
                questions.append(question)
                rag_contexts.append(rag_context)
                combined_contexts.append(combined)
                web_contexts.append(web_context)
                meta.append(
                    {
                        "coverage": round(ratio, 4),
                        "used_web": bool(web_context),
                        "web_gate_decision": bool(use_web),
                        "web_gate_mode": args.web_gate_mode,
                        "web_gate_reason": list(web_gate_reasons)[:5],
                        "time_sensitive_question": bool(time_sensitive_flag),
                        "queried_web": bool(queried_web),
                        "web_query_used": query_used,
                        "web_query_candidates": candidate_queries[:2],
                        "missing_terms": sorted(missing)[:10],
                        "web_cache_hit": web_cache_hit,
                        "web_time_sec": round(web_time, 4),
                        "post_rag_wea_needed": False,
                        "post_rag_wea_reason": "",
                        "used_tool": bool(tool_ids),
                        "tool_query": tool_query,
                        "tool_time_sec": round(tool_time, 4),
                        "retrieval_evidence_used": retrieval_evidence_used,
                    }
                )
                if api_limit_reached or web_error_reached:
                    break
            if not prompts:
                if api_limit_reached or web_error_reached:
                    break
                continue

            primary_preds, primary_elapsed, primary_input_lens, primary_output_lens = (
                generate_batch_predictions(model, tokenizer, prompts, args)
            )
            model_seconds += primary_elapsed

            dual_wea_map = {}
            if args.dual_answer_select:
                if args.dual_slot_refine:
                    refine_indices = []
                    refine_prompts = []
                    for j, m in enumerate(meta):
                        if not m.get("web_gate_decision"):
                            m["post_rag_wea_needed"] = False
                            m["post_rag_wea_reason"] = "web_gate_false"
                            continue
                        rag_pred = primary_preds[j]
                        need_wea, need_reason = decide_wea_refine_needed(
                            question=questions[j],
                            rag_answer=rag_pred,
                            rag_context=rag_contexts[j],
                            coverage_ratio=float(m.get("coverage", 0.0)),
                            missing_terms=m.get("missing_terms", []),
                            min_missing_terms=args.web_min_missing_terms,
                        )
                        m["post_rag_wea_needed"] = bool(need_wea)
                        m["post_rag_wea_reason"] = need_reason
                        if not need_wea:
                            continue

                        web_query_total += 1
                        m["queried_web"] = True
                        candidate_queries = m.get("web_query_candidates") or build_web_queries(
                            questions[j], m.get("missing_terms", []), strategy=args.web_query_strategy
                        )
                        if not candidate_queries:
                            candidate_queries = [(questions[j] or "").strip()]
                        query_used = candidate_queries[0]
                        m["web_query_used"] = query_used
                        web_start = time.time()
                        results = None
                        web_cache_hit = False
                        if not args.disable_web_cache:
                            for candidate_query in candidate_queries:
                                if candidate_query in web_cache:
                                    results = web_cache[candidate_query]
                                    query_used = candidate_query
                                    m["web_query_used"] = query_used
                                    web_cache_hit = True
                                    break
                        if results is None:
                            if args.web_cache_only:
                                results = []
                            else:
                                web_call_failed = False
                                api_candidates = (
                                    candidate_queries
                                    if args.web_try_all_queries_on_miss
                                    else candidate_queries[:1]
                                )
                                for cq in api_candidates:
                                    query_used = cq
                                    m["web_query_used"] = query_used
                                    web_api_calls += 1
                                    if (
                                        args.max_web_api_calls > 0
                                        and web_api_calls >= args.max_web_api_calls
                                    ):
                                        api_limit_reached = True
                                    try:
                                        results = search_serpapi(query_used, num_results=args.web_top_k)
                                    except Exception as err:
                                        web_call_failed = True
                                        web_error_count += 1
                                        if args.stop_on_web_error:
                                            web_error_reached = True
                                            web_last_error = str(err)
                                            web_last_error_query = query_used
                                            print(
                                                "[web-stop] query='{}' error={}".format(
                                                    query_used, web_last_error
                                                )
                                            )
                                            results = []
                                            break
                                        results = []
                                    if not web_call_failed and (not args.disable_web_cache):
                                        web_cache[query_used] = results
                                        append_web_cache(args.web_cache_file, query_used, results)
                                    if results:
                                        break
                                    web_call_failed = False
                                if results is None:
                                    results = []
                        if web_error_reached:
                            break
                        web_time = time.time() - web_start
                        web_seconds += web_time
                        if web_cache_hit:
                            web_cache_hits += 1
                        m["web_cache_hit"] = bool(web_cache_hit)
                        m["web_time_sec"] = round(web_time, 4)
                        web_context = build_web_context(results, max_chars=args.max_web_chars)
                        web_contexts[j] = web_context
                        if not web_context:
                            m["used_web"] = False
                            continue
                        web_used_total += 1
                        m["used_web"] = True
                        combined = "{}\n\nWeb Search:\n{}".format(rag_contexts[j], web_context).strip()
                        if args.max_total_chars is not None and len(combined) > args.max_total_chars:
                            combined = combined[: args.max_total_chars].rsplit(" ", 1)[0]
                        combined_contexts[j] = combined
                        refine_prompt = build_slot_refine_prompt(
                            tokenizer=tokenizer,
                            question=questions[j],
                            rag_answer=rag_pred,
                            rag_context=rag_contexts[j],
                            web_context=web_context,
                            missing_terms=m.get("missing_terms", []),
                            system_prompt=args.system_prompt,
                            use_chat_template=use_chat_template,
                            wea_fusion_mode=args.wea_fusion_mode,
                        )
                        refine_indices.append(j)
                        refine_prompts.append(refine_prompt)
                        if api_limit_reached or web_error_reached:
                            break
                    if web_error_reached:
                        pass
                    if refine_prompts:
                        wea_preds, dual_elapsed, dual_input_lens, dual_output_lens = (
                            generate_batch_predictions(model, tokenizer, refine_prompts, args)
                        )
                        model_seconds += dual_elapsed
                        dual_model_seconds += dual_elapsed
                        dual_input_tokens_total += sum(dual_input_lens)
                        dual_output_tokens_total += sum(dual_output_lens)
                        for k, idx in enumerate(refine_indices):
                            wea_pred_norm = normalize_factoid_answer(
                                wea_preds[k],
                                question=questions[idx],
                                context=combined_contexts[idx],
                            )
                            dual_wea_map[idx] = {
                                "rag_pred": primary_preds[idx],
                                "wea_pred": wea_pred_norm,
                                "wea_input_tokens": dual_input_lens[k],
                                "wea_output_tokens": dual_output_lens[k],
                            }
                        if args.wea_fusion_mode == "compose":
                            compose_prompts = []
                            compose_indices = []
                            for k, idx in enumerate(refine_indices):
                                compose_prompt = build_compose_fusion_prompt(
                                    tokenizer=tokenizer,
                                    question=questions[idx],
                                    rag_answer=primary_preds[idx],
                                    wea_answer=wea_preds[k],
                                    rag_context=rag_contexts[idx],
                                    web_context=web_contexts[idx],
                                    missing_terms=meta[idx].get("missing_terms", []),
                                    system_prompt=args.system_prompt,
                                    use_chat_template=use_chat_template,
                                )
                                compose_indices.append(idx)
                                compose_prompts.append(compose_prompt)
                            if compose_prompts:
                                compose_preds, compose_elapsed, compose_input_lens, compose_output_lens = (
                                    generate_batch_predictions(model, tokenizer, compose_prompts, args)
                                )
                                model_seconds += compose_elapsed
                                dual_model_seconds += compose_elapsed
                                dual_input_tokens_total += sum(compose_input_lens)
                                dual_output_tokens_total += sum(compose_output_lens)
                                for k, idx in enumerate(compose_indices):
                                    compose_pred_norm = normalize_factoid_answer(
                                        compose_preds[k],
                                        question=questions[idx],
                                        context=combined_contexts[idx],
                                    )
                                    dual_wea_map[idx]["compose_pred"] = compose_pred_norm
                                    dual_wea_map[idx]["compose_input_tokens"] = compose_input_lens[k]
                                    dual_wea_map[idx]["compose_output_tokens"] = compose_output_lens[k]
                else:
                    dual_indices = []
                    dual_prompts = []
                    for j, m in enumerate(meta):
                        if not m.get("used_web"):
                            continue
                        rag_prompt = build_prompt(
                            tokenizer,
                            questions[j],
                            rag_contexts[j],
                            args.system_prompt,
                            use_chat_template,
                        )
                        dual_indices.append(j)
                        dual_prompts.append(rag_prompt)
                    if dual_prompts:
                        rag_preds, dual_elapsed, dual_input_lens, dual_output_lens = (
                            generate_batch_predictions(model, tokenizer, dual_prompts, args)
                        )
                        model_seconds += dual_elapsed
                        dual_model_seconds += dual_elapsed
                        dual_input_tokens_total += sum(dual_input_lens)
                        dual_output_tokens_total += sum(dual_output_lens)
                        for k, idx in enumerate(dual_indices):
                            dual_wea_map[idx] = {
                                "rag_pred": rag_preds[k],
                                "rag_input_tokens": dual_input_lens[k],
                                "rag_output_tokens": dual_output_lens[k],
                            }

            for j, ex in enumerate(kept):
                pred_wea = primary_preds[j]
                pred = primary_preds[j]
                input_len = primary_input_lens[j]
                output_len = primary_output_lens[j]
                dual_selected = "single"
                dual_reason = ""
                dual_rag_pred = ""
                dual_compose_pred = ""
                dual_wea_score = None
                dual_rag_score = None
                dual_delta = None
                if j in dual_wea_map:
                    dual_select_total += 1
                    dual_rag_pred = dual_wea_map[j]["rag_pred"]
                    if args.dual_slot_refine:
                        pred_wea = dual_wea_map[j]["wea_pred"]
                    if args.wea_fusion_mode == "compose" and args.dual_slot_refine:
                        dual_compose_pred = (dual_wea_map[j].get("compose_pred", "") or "").strip()
                        rag_score, _ = score_answer_candidate(
                            questions[j], dual_rag_pred, rag_contexts[j]
                        )
                        wea_score, _ = score_answer_candidate(
                            questions[j], pred_wea, combined_contexts[j]
                        )
                        dual_wea_score = round(wea_score, 4)
                        dual_rag_score = round(rag_score, 4)
                        dual_delta = round(wea_score - rag_score, 4)
                        if dual_compose_pred and not contains_refusal_phrase(dual_compose_pred):
                            pred = dual_compose_pred
                            dual_selected = "fused"
                            dual_reason = "fusion_mode_compose"
                            dual_select_fused += 1
                            if (dual_rag_pred or "").strip() != dual_compose_pred:
                                dual_switch_to_fused += 1
                        elif pred_wea:
                            pred = pred_wea
                            dual_selected = "wea"
                            dual_reason = "compose_empty_fallback_wea"
                            dual_select_wea += 1
                            if (dual_rag_pred or "").strip() != (pred_wea or "").strip():
                                dual_switch_to_wea += 1
                        else:
                            pred = dual_rag_pred
                            dual_selected = "rag"
                            dual_reason = "compose_empty_fallback_rag"
                            dual_select_rag += 1
                    elif args.wea_fusion_mode == "merge" and args.dual_slot_refine:
                        rag_score, _ = score_answer_candidate(
                            questions[j], dual_rag_pred, rag_contexts[j]
                        )
                        wea_score, _ = score_answer_candidate(
                            questions[j], pred_wea, combined_contexts[j]
                        )
                        if args.merge_force_all:
                            if pred_wea:
                                changed = (dual_rag_pred or "").strip() != (pred_wea or "").strip()
                                rag_locked = (
                                    bool((dual_rag_pred or "").strip())
                                    and rag_score >= args.dual_rag_lock_score
                                )
                                force_guard_reason = ""
                                if changed:
                                    if (
                                        args.merge_force_min_wea_score is not None
                                        and wea_score < args.merge_force_min_wea_score
                                    ):
                                        force_guard_reason = "fusion_force_guard_min_wea_score"
                                    elif (
                                        args.merge_force_min_delta is not None
                                        and (wea_score - rag_score) < args.merge_force_min_delta
                                    ):
                                        force_guard_reason = "fusion_force_guard_min_delta"
                                    elif args.merge_force_use_conflict_guards:
                                        wea_web_score, wea_web_detail = score_answer_candidate(
                                            questions[j], pred_wea, web_contexts[j]
                                        )
                                        web_supported = bool(wea_web_detail.get("phrase_hit")) or float(
                                            wea_web_detail.get("token_support", 0.0)
                                        ) >= max(0.25, args.dual_wea_min_token_support * 0.8)
                                        year_conflict, entity_conflict = detect_answer_conflicts(
                                            dual_rag_pred, pred_wea
                                        )
                                        if rag_locked and not web_supported and wea_web_score < 0.45:
                                            force_guard_reason = "fusion_force_guard_weak_web_support"
                                        elif rag_locked and year_conflict and rag_score >= (wea_score - 0.05):
                                            force_guard_reason = "fusion_force_guard_year_conflict"
                                        elif rag_locked and entity_conflict and rag_score >= (wea_score - 0.05):
                                            force_guard_reason = "fusion_force_guard_entity_conflict"

                                if force_guard_reason:
                                    pred = dual_rag_pred
                                    dual_selected = "rag"
                                    dual_reason = force_guard_reason
                                    dual_select_rag += 1
                                else:
                                    pred = pred_wea
                                    dual_selected = "wea"
                                    dual_reason = "fusion_mode_merge_force"
                                    dual_select_wea += 1
                                    if dual_rag_pred != pred_wea:
                                        dual_switch_to_wea += 1
                            else:
                                pred = dual_rag_pred
                                dual_selected = "rag"
                                dual_reason = "fusion_force_empty_fallback_rag"
                                dual_select_rag += 1
                        else:
                            # Build merged answer first, then apply guard-based dual selection.
                            choice = choose_dual_answer(
                                question=questions[j],
                                rag_answer=dual_rag_pred,
                                wea_answer=pred_wea,
                                rag_context=rag_contexts[j],
                                wea_context=combined_contexts[j],
                                score_margin=args.dual_select_margin,
                                wea_min_score=args.dual_wea_min_score,
                                wea_min_token_support=args.dual_wea_min_token_support,
                                require_wea_context_match=args.dual_require_context_match,
                                rag_lock_score=args.dual_rag_lock_score,
                            )
                            pred = choice["answer"]
                            dual_selected = choice["selected"]
                            dual_reason = "fusion_merge_guard_" + choice["reason"]
                            if dual_selected == "wea":
                                dual_select_wea += 1
                                if dual_rag_pred != pred_wea:
                                    dual_switch_to_wea += 1
                            else:
                                dual_select_rag += 1
                                if dual_rag_pred != pred_wea:
                                    dual_switch_to_rag += 1
                        dual_wea_score = round(wea_score, 4)
                        dual_rag_score = round(rag_score, 4)
                        dual_delta = round(wea_score - rag_score, 4)
                    else:
                        choice = choose_dual_answer(
                            question=questions[j],
                            rag_answer=dual_rag_pred,
                            wea_answer=pred_wea,
                            rag_context=rag_contexts[j],
                            wea_context=combined_contexts[j],
                            score_margin=args.dual_select_margin,
                            wea_min_score=args.dual_wea_min_score,
                            wea_min_token_support=args.dual_wea_min_token_support,
                            require_wea_context_match=args.dual_require_context_match,
                            rag_lock_score=args.dual_rag_lock_score,
                        )
                        pred = choice["answer"]
                        dual_selected = choice["selected"]
                        dual_reason = choice["reason"]
                        dual_wea_score = choice["wea_score"]
                        dual_rag_score = choice["rag_score"]
                        dual_delta = choice["score_delta_wea_minus_rag"]
                        if dual_selected == "wea":
                            dual_select_wea += 1
                            if dual_rag_pred != pred_wea:
                                dual_switch_to_wea += 1
                        else:
                            dual_select_rag += 1
                            if dual_rag_pred != pred_wea:
                                dual_switch_to_rag += 1
                gold = ex.get("answer")
                if j in dual_wea_map:
                    norm_context = combined_contexts[j] or rag_contexts[j]
                    pred = normalize_factoid_answer(pred, question=questions[j], context=norm_context)
                if gold is not None:
                    em_sum += 1.0 if exact_match_score(pred, gold) else 0.0
                    f1_sum += f1_score(pred, gold)[0]
                    gold_cnt += 1
                record = {"id": ex.get("id"), "prediction": pred}
                record.update(meta[j])
                record["input_tokens"] = input_len
                record["output_tokens"] = output_len
                record["dual_select_applied"] = bool(j in dual_wea_map)
                record["dual_selected"] = dual_selected
                record["dual_select_reason"] = dual_reason
                record["dual_wea_pred"] = pred_wea if j in dual_wea_map else ""
                record["dual_rag_pred"] = dual_rag_pred
                record["dual_compose_pred"] = dual_compose_pred
                record["dual_wea_score"] = dual_wea_score
                record["dual_rag_score"] = dual_rag_score
                record["dual_score_delta_wea_minus_rag"] = dual_delta
                if j in dual_wea_map:
                    if "rag_input_tokens" in dual_wea_map[j]:
                        record["dual_rag_input_tokens"] = dual_wea_map[j]["rag_input_tokens"]
                        record["dual_rag_output_tokens"] = dual_wea_map[j]["rag_output_tokens"]
                    if "wea_input_tokens" in dual_wea_map[j]:
                        record["dual_wea_input_tokens"] = dual_wea_map[j]["wea_input_tokens"]
                        record["dual_wea_output_tokens"] = dual_wea_map[j]["wea_output_tokens"]
                    if "compose_input_tokens" in dual_wea_map[j]:
                        record["dual_compose_input_tokens"] = dual_wea_map[j]["compose_input_tokens"]
                        record["dual_compose_output_tokens"] = dual_wea_map[j]["compose_output_tokens"]
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
                    "max_web_api_calls": args.max_web_api_calls,
                    "stopped_by_web_api_limit": api_limit_reached,
                    "web_used": web_used_total,
                    "web_queries": web_query_total,
                    "web_api_calls": web_api_calls,
                    "web_cache_hits": web_cache_hits,
                    "web_cache_only": args.web_cache_only,
                    "disable_web_cache": args.disable_web_cache,
                    "stop_on_web_error": args.stop_on_web_error,
                    "stopped_by_web_error": web_error_reached,
                    "web_error_count": web_error_count,
                    "web_last_error": web_last_error,
                    "web_last_error_query": web_last_error_query,
                    "web_gate_mode": args.web_gate_mode,
                    "web_query_strategy": args.web_query_strategy,
                    "web_try_all_queries_on_miss": args.web_try_all_queries_on_miss,
                    "web_gate_true": web_gate_true_total,
                    "time_sensitive_questions": time_sensitive_questions,
                    "web_seconds": web_seconds,
                    "tool_used": tool_used_total,
                    "tool_calls": tool_calls,
                    "tool_seconds": tool_seconds,
                    "model_seconds": model_seconds,
                    "dual_answer_select": args.dual_answer_select,
                    "dual_slot_refine": args.dual_slot_refine,
                    "wea_fusion_mode": args.wea_fusion_mode,
                    "merge_force_all": args.merge_force_all,
                    "merge_force_min_wea_score": args.merge_force_min_wea_score,
                    "merge_force_min_delta": args.merge_force_min_delta,
                    "merge_force_use_conflict_guards": args.merge_force_use_conflict_guards,
                    "dual_select_margin": args.dual_select_margin,
                    "dual_wea_min_score": args.dual_wea_min_score,
                    "dual_wea_min_token_support": args.dual_wea_min_token_support,
                    "dual_require_context_match": args.dual_require_context_match,
                    "dual_rag_lock_score": args.dual_rag_lock_score,
                    "dual_select_total": dual_select_total,
                    "dual_select_wea": dual_select_wea,
                    "dual_select_rag": dual_select_rag,
                    "dual_select_fused": dual_select_fused,
                    "dual_switch_to_wea": dual_switch_to_wea,
                    "dual_switch_to_rag": dual_switch_to_rag,
                    "dual_switch_to_fused": dual_switch_to_fused,
                    "dual_model_seconds": dual_model_seconds,
                    "dual_input_tokens": dual_input_tokens_total,
                    "dual_output_tokens": dual_output_tokens_total,
                    "input_tokens": input_tokens_total,
                    "output_tokens": output_tokens_total,
                    "avg_latency_sec": (time.time() - start_time) / avg_div,
                    "avg_web_latency_sec": web_seconds / (web_query_total or 1),
                    "avg_tool_latency_sec": tool_seconds / (tool_calls or 1),
                    "avg_model_latency_sec": model_seconds / avg_div,
                    "avg_input_tokens": input_tokens_total / avg_div,
                    "avg_output_tokens": output_tokens_total / avg_div,
                    "seconds": time.time() - start_time,
                }
                write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)
                new_preds = 0
            if api_limit_reached or web_error_reached:
                break

    elapsed = time.time() - start_time
    avg_div = processed_total or 1
    metrics = {
        "exact_match": 100.0 * em_sum / gold_cnt if gold_cnt else 0.0,
        "f1": 100.0 * f1_sum / gold_cnt if gold_cnt else 0.0,
        "total": gold_cnt,
        "predicted": predicted_total,
        "max_web_api_calls": args.max_web_api_calls,
        "stopped_by_web_api_limit": api_limit_reached,
        "web_used": web_used_total,
        "web_queries": web_query_total,
        "web_api_calls": web_api_calls,
        "web_cache_hits": web_cache_hits,
        "web_cache_only": args.web_cache_only,
        "disable_web_cache": args.disable_web_cache,
        "stop_on_web_error": args.stop_on_web_error,
        "stopped_by_web_error": web_error_reached,
        "web_error_count": web_error_count,
        "web_last_error": web_last_error,
        "web_last_error_query": web_last_error_query,
        "web_gate_mode": args.web_gate_mode,
        "web_query_strategy": args.web_query_strategy,
        "web_try_all_queries_on_miss": args.web_try_all_queries_on_miss,
        "web_gate_true": web_gate_true_total,
        "time_sensitive_questions": time_sensitive_questions,
        "web_seconds": web_seconds,
        "tool_used": tool_used_total,
        "tool_calls": tool_calls,
        "tool_seconds": tool_seconds,
        "model_seconds": model_seconds,
        "dual_answer_select": args.dual_answer_select,
        "dual_slot_refine": args.dual_slot_refine,
        "wea_fusion_mode": args.wea_fusion_mode,
        "merge_force_all": args.merge_force_all,
        "merge_force_min_wea_score": args.merge_force_min_wea_score,
        "merge_force_min_delta": args.merge_force_min_delta,
        "merge_force_use_conflict_guards": args.merge_force_use_conflict_guards,
        "dual_select_margin": args.dual_select_margin,
        "dual_wea_min_score": args.dual_wea_min_score,
        "dual_wea_min_token_support": args.dual_wea_min_token_support,
        "dual_require_context_match": args.dual_require_context_match,
        "dual_rag_lock_score": args.dual_rag_lock_score,
        "dual_select_total": dual_select_total,
        "dual_select_wea": dual_select_wea,
        "dual_select_rag": dual_select_rag,
        "dual_select_fused": dual_select_fused,
        "dual_switch_to_wea": dual_switch_to_wea,
        "dual_switch_to_rag": dual_switch_to_rag,
        "dual_switch_to_fused": dual_switch_to_fused,
        "dual_model_seconds": dual_model_seconds,
        "dual_input_tokens": dual_input_tokens_total,
        "dual_output_tokens": dual_output_tokens_total,
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
        "avg_latency_sec": elapsed / avg_div,
        "avg_web_latency_sec": web_seconds / (web_query_total or 1),
        "avg_tool_latency_sec": tool_seconds / (tool_calls or 1),
        "avg_model_latency_sec": model_seconds / avg_div,
        "avg_input_tokens": input_tokens_total / avg_div,
        "avg_output_tokens": output_tokens_total / avg_div,
        "seconds": elapsed,
    }
    print(
        "EM {exact_match:.2f} | F1 {f1:.2f} | total {total} | preds {predicted} | web_used {web_used}".format(
            **metrics
        )
    )
    write_metrics_snapshot(metrics, args.metrics_file, args.metrics_log)


if __name__ == "__main__":
    main()
