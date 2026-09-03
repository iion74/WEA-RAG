#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_context(paragraphs, include_titles=True, max_paragraphs=None, max_chars=None):
    blocks = []
    for idx, item in enumerate(paragraphs or []):
        if max_paragraphs is not None and idx >= max_paragraphs:
            break
        if not item or len(item) < 2:
            continue
        title, sentences = item[0], item[1]
        paragraph_text = " ".join(sentences or []).strip()
        if include_titles:
            block = "{}\n{}".format(title, paragraph_text).strip()
        else:
            block = paragraph_text
        if block:
            blocks.append(block)
    context = "\n\n".join(blocks)
    if max_chars is not None and len(context) > max_chars:
        context = context[:max_chars].rsplit(" ", 1)[0]
    return context


def normalize_record(ex, include_titles=True, max_paragraphs=None, max_chars=None):
    raw_context = ex.get("context", [])
    out = {
        "id": ex.get("id") or ex.get("_id"),
        "question": ex.get("question", ""),
        "context": build_context(
            raw_context,
            include_titles=include_titles,
            max_paragraphs=max_paragraphs,
            max_chars=max_chars,
        ),
        "answer": ex.get("answer"),
    }
    for key in [
        "type",
        "level",
        "supporting_facts",
        "evidences",
        "entity_ids",
        "evidences_id",
        "answer_id",
    ]:
        if key in ex:
            out[key] = ex[key]
    if raw_context:
        out["context_paragraphs"] = raw_context
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--max_paragraphs", type=int, default=None)
    parser.add_argument("--max_context_chars", type=int, default=None)
    parser.add_argument("--no_titles", action="store_true")
    args = parser.parse_args()

    raw_path = Path(args.raw_file)
    out_path = Path(args.out_file)
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("raw_file must contain a JSON list of examples")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in data:
            row = normalize_record(
                ex,
                include_titles=not args.no_titles,
                max_paragraphs=args.max_paragraphs,
                max_chars=args.max_context_chars,
            )
            if not row["id"]:
                raise SystemExit("example missing id/_id")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print("wrote {} rows to {}".format(written, out_path))


if __name__ == "__main__":
    main()
