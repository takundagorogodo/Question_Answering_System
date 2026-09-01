import os
import re

def clean_text(text):
    """
    Normalize whitespace in a block of text.
    - Collapses multiple blank lines into one
    - Collapses multiple spaces into one
    - Strips leading/trailing whitespace
    """
    text = re.sub(r'\n{3,}', '\n\n', text)   # 3+ newlines -> 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)       # multiple spaces/tabs -> 1 space
    return text.strip()


def load_documents(folder_path="data/documents"):
    """
    Load all .txt files from folder_path.
    Returns a list of dicts: [{ "source": filename, "title": title, "text": cleaned_text }, ...]
    """
    documents = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text)

        # First line of each file is used as the title
        title = cleaned.split("\n")[0].strip()

        documents.append({
            "source": filename,
            "title": title,
            "text": cleaned
        })

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents:\n")
    for doc in docs:
        word_count = len(doc["text"].split())
        print(f"- {doc['source']} | Title: {doc['title']} | Words: {word_count}")