import re

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


def tokenize(text):
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        if len(token) >= 2
    ]


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
