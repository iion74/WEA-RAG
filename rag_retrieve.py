import argparse
import json
import os
import re

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from evidence_utils import build_evidence_context, select_evidence_sentences


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


def dense_search(index, embedder, query, top_k):
    embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores, ids = index.search(embedding, top_k)
    return ids[0].tolist(), scores[0].tolist()


def dense_search_multi(index, embedder, queries, top_k):
    embeddings = embedder.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores, ids = index.search(embeddings, top_k)
    score_map = {}
    for row in range(len(queries)):
        for doc_id, score in zip(ids[row].tolist(), scores[row].tolist()):
            if doc_id < 0:
                continue
            prev = score_map.get(doc_id)
            if prev is None or score > prev:
                score_map[doc_id] = score
    return score_map


def build_title_to_ids(titles):
    title_to_ids = {}
    for idx, title in enumerate(titles):
        title_to_ids.setdefault(title, []).append(idx)
    return title_to_ids


def candidate_ids_from_example_context(example, title_to_ids):
    candidate_ids = []
    seen = set()
    for para in example.get("context", []):
        if not para or len(para) < 1:
            continue
        title = para[0]
        for doc_id in title_to_ids.get(title, []):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            candidate_ids.append(doc_id)
    return candidate_ids


CAPITAL_TERM_RE = re.compile(r"(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SUBQ_SPLIT_RE = re.compile(
    r"\s+(?:and|or|after|before|while|whereas)\s+|,\s+|;\s+",
    flags=re.IGNORECASE,
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


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def unique_keep_order(items):
    out = []
    seen = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def decompose_subquestions(question, mode="heuristic", max_queries=3):
    base = normalize_text(question)
    if not base:
        return []
    if mode == "none":
        return [base]

    candidates = [base]
    parts = SUBQ_SPLIT_RE.split(base)
    for part in parts:
        part = normalize_text(part.strip(" ,;"))
        if len(part) < 10:
            continue
        if part.lower() == base.lower():
            continue
        candidates.append(part)

    candidates = unique_keep_order(candidates)
    return candidates[: max(1, max_queries)]


def extract_capitalized_terms(text):
    return [t.strip() for t in CAPITAL_TERM_RE.findall(text or "") if t.strip()]


def extract_doc_entities(title, text, max_entities=20):
    source = "{}. {}".format(title or "", (text or "")[:1200])
    terms = unique_keep_order(extract_capitalized_terms(source))
    if max_entities > 0:
        terms = terms[:max_entities]
    return {t.lower() for t in terms}


def extract_question_entities(question):
    capital_terms = [t.lower() for t in extract_capitalized_terms(question)]
    if capital_terms:
        return set(capital_terms)
    return {
        token.lower()
        for token in TOKEN_RE.findall(question or "")
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def normalize_score_map(score_map):
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {k: 0.0 for k in score_map}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


def build_bridge_queries(question, doc_ids, titles, texts, per_doc=2):
    if per_doc <= 0:
        return []
    queries = []
    for doc_id in doc_ids:
        if doc_id < 0 or doc_id >= len(texts):
            continue
        terms = unique_keep_order(
            extract_capitalized_terms("{} {}".format(titles[doc_id], texts[doc_id][:600]))
        )
        added = 0
        for term in terms:
            if len(term) < 3 or len(term) > 64:
                continue
            queries.append("{} {}".format(question, term))
            added += 1
            if added >= per_doc:
                break
    return unique_keep_order(queries)


def compute_graph_scores(candidate_ids, titles, texts, question, seed_docs=4):
    if not candidate_ids:
        return {}, {}
    question_entities = extract_question_entities(question)
    doc_entities = {}
    q_overlap = {}
    for doc_id in candidate_ids:
        ents = extract_doc_entities(titles[doc_id], texts[doc_id])
        doc_entities[doc_id] = ents
        q_overlap[doc_id] = len(ents.intersection(question_entities))

    seeds = sorted(candidate_ids, key=lambda x: q_overlap.get(x, 0), reverse=True)[:seed_docs]
    if not seeds:
        seeds = candidate_ids[:seed_docs]
    if all(q_overlap.get(doc_id, 0) == 0 for doc_id in seeds):
        seeds = candidate_ids[:seed_docs]

    graph_raw = {}
    for doc_id in candidate_ids:
        ents_i = doc_entities[doc_id]
        best_link = 0.0
        for seed_id in seeds:
            if seed_id == doc_id:
                continue
            ents_j = doc_entities[seed_id]
            if not ents_i or not ents_j:
                continue
            overlap = len(ents_i.intersection(ents_j))
            if overlap == 0:
                continue
            link = overlap / (1.0 + min(len(ents_i), len(ents_j)))
            if link > best_link:
                best_link = link
        graph_raw[doc_id] = best_link

    graph_norm = normalize_score_map(graph_raw)
    q_norm = normalize_score_map(q_overlap)
    return graph_norm, q_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", default="data/hotpot_dev_distractor_v1.json")
    parser.add_argument("--corpus_file", default="rag_corpus/docs.jsonl")
    parser.add_argument("--index_dir", default="rag_index")
    parser.add_argument("--out_file", default="rag_cache/hotpot_dev_distractor_topk.jsonl")
    parser.add_argument("--metrics_file", default="rag_cache/hotpot_dev_distractor_metrics.json")
    parser.add_argument("--dense_k", type=int, default=100)
    parser.add_argument("--rerank_k", type=int, default=80)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--two_hop", action="store_true")
    parser.add_argument("--hop_titles_k", type=int, default=2)
    parser.add_argument(
        "--subquestion_mode",
        default="heuristic",
        choices=["none", "heuristic"],
        help="none: only original question, heuristic: split question into sub-queries",
    )
    parser.add_argument(
        "--subquestion_max",
        type=int,
        default=3,
        help="max number of queries including original question",
    )
    parser.add_argument(
        "--bridge_query_per_doc",
        type=int,
        default=2,
        help="for two-hop: number of bridge entities used per seed document",
    )
    parser.add_argument(
        "--graph_connectivity_weight",
        type=float,
        default=0.2,
        help="weight for document-document connectivity score in final rerank",
    )
    parser.add_argument(
        "--question_overlap_weight",
        type=float,
        default=0.1,
        help="weight for question-entity overlap score in final rerank",
    )
    parser.add_argument(
        "--dense_weight",
        type=float,
        default=0.05,
        help="weight for dense retrieval normalized score in final rerank",
    )
    parser.add_argument("--evidence_sentences", type=int, default=0)
    parser.add_argument("--evidence_per_doc", type=int, default=2)
    parser.add_argument("--evidence_min_len", type=int, default=20)
    parser.add_argument("--evidence_max_chars", type=int, default=1200)
    parser.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--candidate_mode",
        default="global_dense",
        choices=["global_dense", "provided_context"],
        help="global_dense: open-domain dense retrieval; provided_context: rerank only titles from each example context",
    )
    parser.add_argument(
        "--fallback_global_if_empty",
        action="store_true",
        help="when candidate_mode=provided_context and no matching titles are found in corpus, fallback to global_dense retrieval",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    titles, texts = load_corpus(args.corpus_file)
    title_to_ids = build_title_to_ids(titles)
    reranker = CrossEncoder(args.rerank_model, device=args.device)
    index = None
    embedder = None
    if args.candidate_mode == "global_dense" or args.fallback_global_if_empty:
        index_path = os.path.join(args.index_dir, "dense.index")
        index = faiss.read_index(index_path)
        embedder = SentenceTransformer(args.embed_model, device=args.device)

    with open(args.raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit is not None:
        data = data[: args.limit]

    hit_sum = 0.0
    recall_sum = 0.0
    coverage_sum = 0.0
    query_count_sum = 0.0
    bridge_query_count_sum = 0.0
    total = 0

    with open(args.out_file, "w", encoding="utf-8") as out_f:
        for ex in tqdm(data, desc="retrieve"):
            question = ex.get("question", "")
            support_titles = {t for t, _ in ex.get("supporting_facts", [])}
            queries = decompose_subquestions(
                question, mode=args.subquestion_mode, max_queries=args.subquestion_max
            )
            if not queries:
                queries = [question]
            bridge_queries = []
            dense_score_map = {}

            if args.candidate_mode == "provided_context":
                candidate_ids = candidate_ids_from_example_context(ex, title_to_ids)
                if not candidate_ids and args.fallback_global_if_empty:
                    dense_score_map = dense_search_multi(index, embedder, queries, args.dense_k)
                    candidate_ids = sorted(
                        dense_score_map.items(), key=lambda x: x[1], reverse=True
                    )
                    candidate_ids = [doc_id for doc_id, _ in candidate_ids[: args.rerank_k]]
            else:
                dense_score_map = dense_search_multi(index, embedder, queries, args.dense_k)

                if args.two_hop and dense_score_map:
                    first_hop_ids = [
                        doc_id
                        for doc_id, _ in sorted(
                            dense_score_map.items(), key=lambda x: x[1], reverse=True
                        )
                    ][: args.hop_titles_k]
                    bridge_queries = build_bridge_queries(
                        question,
                        first_hop_ids,
                        titles,
                        texts,
                        per_doc=args.bridge_query_per_doc,
                    )
                    if bridge_queries:
                        hop_score_map = dense_search_multi(
                            index, embedder, bridge_queries, args.dense_k
                        )
                        for doc_id, score in hop_score_map.items():
                            prev = dense_score_map.get(doc_id)
                            if prev is None or score > prev:
                                dense_score_map[doc_id] = score

                candidate_ids = sorted(dense_score_map.items(), key=lambda x: x[1], reverse=True)
                candidate_ids = [doc_id for doc_id, _ in candidate_ids[: args.rerank_k]]

            if candidate_ids:
                pairs = [
                    (question, "{}\n{}".format(titles[idx], texts[idx]))
                    for idx in candidate_ids
                ]
                rerank_scores = reranker.predict(pairs, batch_size=args.batch_size)
                rerank_map = {
                    doc_id: float(score)
                    for doc_id, score in zip(candidate_ids, rerank_scores)
                }
                dense_norm = normalize_score_map(
                    {doc_id: dense_score_map.get(doc_id, 0.0) for doc_id in candidate_ids}
                )
                # The paper configuration uses hop_titles_k=2, yielding four graph seed documents.
                graph_seed_docs = max(4, args.hop_titles_k * 2)
                graph_norm, q_norm = compute_graph_scores(
                    candidate_ids,
                    titles,
                    texts,
                    question,
                    seed_docs=graph_seed_docs,
                )

                combined = {}
                for doc_id in candidate_ids:
                    score = rerank_map.get(doc_id, 0.0)
                    score += args.dense_weight * dense_norm.get(doc_id, 0.0)
                    score += args.graph_connectivity_weight * graph_norm.get(doc_id, 0.0)
                    score += args.question_overlap_weight * q_norm.get(doc_id, 0.0)
                    combined[doc_id] = score

                reranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
                top_ids = [doc_id for doc_id, _ in reranked[: args.top_k]]
                top_scores = [score for _, score in reranked[: args.top_k]]
            else:
                top_ids = []
                top_scores = []
            top_titles = [titles[idx] for idx in top_ids]

            evidence_sentences = []
            evidence_context = ""
            if args.evidence_sentences > 0 and top_ids:
                selected = select_evidence_sentences(
                    question,
                    top_ids,
                    titles,
                    texts,
                    max_sentences=args.evidence_sentences,
                    per_doc=args.evidence_per_doc,
                    min_len=args.evidence_min_len,
                )
                evidence_context = build_evidence_context(
                    selected, max_chars=args.evidence_max_chars
                )
                evidence_sentences = [
                    {
                        "title": title,
                        "sentence": sent,
                        "score": round(float(score), 6),
                    }
                    for score, _, title, sent in selected
                ]

            hit = 1.0 if support_titles & set(top_titles) else 0.0
            recall = (
                len(support_titles & set(top_titles)) / len(support_titles)
                if support_titles
                else 0.0
            )
            coverage = 1.0 if support_titles.issubset(set(top_titles)) else 0.0

            hit_sum += hit
            recall_sum += recall
            coverage_sum += coverage
            query_count_sum += len(queries)
            bridge_query_count_sum += len(bridge_queries)
            total += 1

            out_f.write(
                json.dumps(
                    {
                        "id": ex.get("id") or ex.get("_id"),
                        "question": question,
                        "top_ids": top_ids,
                        "top_titles": top_titles,
                        "top_scores": top_scores,
                        "queries_used": queries,
                        "bridge_queries": bridge_queries,
                        "evidence_context": evidence_context,
                        "evidence_sentences": evidence_sentences,
                        "hit": hit,
                        "recall": recall,
                        "coverage": coverage,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "hit@k": 100.0 * hit_sum / total if total else 0.0,
        "recall@k": 100.0 * recall_sum / total if total else 0.0,
        "full_coverage@k": 100.0 * coverage_sum / total if total else 0.0,
        "total": total,
        "avg_queries": query_count_sum / total if total else 0.0,
        "avg_bridge_queries": bridge_query_count_sum / total if total else 0.0,
        "dense_k": args.dense_k,
        "rerank_k": args.rerank_k,
        "top_k": args.top_k,
        "two_hop": args.two_hop,
        "subquestion_mode": args.subquestion_mode,
        "subquestion_max": args.subquestion_max,
        "bridge_query_per_doc": args.bridge_query_per_doc,
        "graph_connectivity_weight": args.graph_connectivity_weight,
        "question_overlap_weight": args.question_overlap_weight,
        "dense_weight": args.dense_weight,
        "evidence_sentences": args.evidence_sentences,
        "evidence_per_doc": args.evidence_per_doc,
        "candidate_mode": args.candidate_mode,
        "fallback_global_if_empty": args.fallback_global_if_empty,
        "limit": args.limit,
    }
    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        "hit@k {hit@k:.2f} | recall@k {recall@k:.2f} | full@k {full_coverage@k:.2f} | total {total}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
