from src.preprocessing import load_documents


def chunk_text(text, chunk_size=150, overlap=30):
    """
    Split text into overlapping chunks, measured in words.
    - chunk_size: target number of words per chunk
    - overlap: number of words repeated at the start of the next chunk
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap  # step forward, but re-include the overlap

    return chunks


def chunk_documents(documents, chunk_size=150, overlap=30):
    """
    Chunk every document in the list.
    Returns a list of dicts: [{ "source": ..., "title": ..., "chunk_id": ..., "text": chunk_text }, ...]
    """
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)

        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "source": doc["source"],
                "title": doc["title"],
                "chunk_id": f"{doc['source']}_{i}",
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Loaded {len(docs)} documents -> produced {len(chunks)} chunks\n")

    # Show a concrete before/after example for the first document
    first_doc = docs[0]
    first_doc_chunks = [c for c in chunks if c["source"] == first_doc["source"]]

    print(f"Example — '{first_doc['title']}' ({len(first_doc['text'].split())} words)")
    print(f"split into {len(first_doc_chunks)} chunks:\n")

    for c in first_doc_chunks:
        word_count = len(c["text"].split())
        preview = c["text"][:100].replace("\n", " ")
        print(f"[{c['chunk_id']}] ({word_count} words): {preview}...")