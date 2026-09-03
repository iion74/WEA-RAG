import argparse
import json
import os


def add_paragraph(title, paragraph, store, max_chars):
    if not paragraph:
        return
    if title not in store:
        store[title] = paragraph[:max_chars] if max_chars else paragraph
        return
    existing = store[title]
    if paragraph in existing:
        return
    merged = (existing + " " + paragraph).strip()
    store[title] = merged[:max_chars] if max_chars else merged


def collect_titles(paths, max_chars, max_paragraphs_per_example, max_examples):
    title_to_text = {}
    total_context = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx_ex, ex in enumerate(data):
            if max_examples is not None and idx_ex >= max_examples:
                break
            for idx, (title, sentences) in enumerate(ex.get("context", [])):
                if max_paragraphs_per_example is not None and idx >= max_paragraphs_per_example:
                    break
                total_context += 1
                paragraph = " ".join(sentences).strip()
                add_paragraph(title, paragraph, title_to_text, max_chars)
    return title_to_text, total_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", default="data/hotpot_train_v1.1.json"
    )
    parser.add_argument(
        "--dev_files",
        nargs="+",
        default=[
            "data/hotpot_dev_distractor_v1.json",
            "data/hotpot_dev_fullwiki_v1.json",
        ],
    )
    parser.add_argument("--out_file", default="rag_corpus/docs.jsonl")
    parser.add_argument("--max_chars", type=int, default=2000)
    parser.add_argument("--max_paragraphs_per_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    paths = []
    if args.train_file and str(args.train_file).lower() not in ["none", "null", ""]:
        paths.append(args.train_file)
    paths.extend(args.dev_files)
    title_to_text, total_context = collect_titles(
        paths, args.max_chars, args.max_paragraphs_per_example, args.max_examples
    )

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc_count = 0
    with open(args.out_file, "w", encoding="utf-8") as out_f:
        for doc_id, (title, text) in enumerate(title_to_text.items()):
            out_f.write(
                json.dumps(
                    {"doc_id": doc_id, "title": title, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            doc_count += 1

    meta = {
        "docs": doc_count,
        "total_context_items": total_context,
        "max_chars": args.max_chars,
        "max_paragraphs_per_example": args.max_paragraphs_per_example,
        "max_examples": args.max_examples,
        "sources": paths,
    }
    meta_path = os.path.join(os.path.dirname(args.out_file) or ".", "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        "wrote {} docs (from {} context items) to {}".format(
            doc_count, total_context, args.out_file
        )
    )


if __name__ == "__main__":
    main()
