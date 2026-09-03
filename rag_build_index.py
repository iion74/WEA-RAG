import argparse
import json
import os

import faiss
from sentence_transformers import SentenceTransformer


def iter_docs(corpus_file):
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            yield record["doc_id"], record["title"], record["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default="rag_corpus/docs.jsonl")
    parser.add_argument("--index_dir", default="rag_index")
    parser.add_argument(
        "--embed_model", default="BAAI/bge-large-en-v1.5"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    embedder = SentenceTransformer(args.embed_model, device=args.device)

    # Build FAISS index incrementally to avoid holding all embeddings in RAM.
    index = None
    doc_count = 0
    batch_texts = []
    for _, title, text in iter_docs(args.corpus_file):
        batch_texts.append("{}\n{}".format(title, text))
        if len(batch_texts) >= args.batch_size:
            embeddings = embedder.encode(
                batch_texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if index is None:
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            doc_count += len(batch_texts)
            batch_texts = []

    if batch_texts:
        embeddings = embedder.encode(
            batch_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if index is None:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        doc_count += len(batch_texts)

    if index is None:
        raise SystemExit("no documents found in corpus")

    index_path = os.path.join(args.index_dir, "dense.index")
    faiss.write_index(index, index_path)

    meta = {
        "corpus_file": args.corpus_file,
        "docs": doc_count,
        "embed_model": args.embed_model,
        "batch_size": args.batch_size,
        "device": args.device,
        "index_path": index_path,
    }
    with open(os.path.join(args.index_dir, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("built dense index with {} docs at {}".format(doc_count, index_path))


if __name__ == "__main__":
    main()
