# chunk_embed.py
# chunks extracted text, embeds it and stores in ChromaDB

import chromadb
from sentence_transformers import SentenceTransformer
import os
from config import  CHROMA_PATH, CHUNK_MIN_WORDS, CHUNK_MAX_WORDS

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_extracted_text(extracted_dir="extracted"):
    texts = []
    files = sorted(
        [f for f in os.listdir(extracted_dir)
         if f.startswith("page_") and f.endswith(".txt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    for file in files:
        with open(f"{extracted_dir}/{file}", "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip() and content.strip() != "BLANK PAGE":
                texts.append((file, content))

    print(f"Loaded {len(texts)} pages")
    return texts


def chunk_by_structure(text, page_name):
    chunks = []
    blocks = [b.strip() for b in text.split('\n\n') if b.strip()]

    current_block = ""

    for block in blocks:
        word_count = len(block.split())

        if word_count < CHUNK_MIN_WORDS:
            current_block += "\n" + block
            continue

        if len(current_block.split()) < 30:
            current_block += "\n\n" + block
        else:
            if current_block.strip():
                chunks.append({
                    "text": current_block.strip(),
                    "page": page_name
                })
            current_block = block

    if current_block.strip():
        chunks.append({
            "text": current_block.strip(),
            "page": page_name
        })

    # split oversized chunks
    final_chunks = []
    for chunk in chunks:
        words = chunk["text"].split()
        if len(words) > CHUNK_MAX_WORDS:
            for i in range(0, len(words), CHUNK_MAX_WORDS):
                piece = " ".join(words[i:i+CHUNK_MAX_WORDS])
                if piece.strip():
                    final_chunks.append({
                        "text": piece,
                        "page": chunk["page"]
                    })
        else:
            final_chunks.append(chunk)

    return final_chunks


def store_in_chromadb(all_texts,collection_name):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # fresh start
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    collection = client.create_collection(collection_name)


    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    all_ids    = []
    all_meta   = []
    counter    = 0

    for page_name, text in all_texts:
        chunks = chunk_by_structure(text, page_name)
        for chunk in chunks:
            all_chunks.append(chunk["text"])
            all_ids.append(f"chunk_{counter}")
            all_meta.append({"page": chunk["page"]})
            counter += 1

    print(f"Total chunks: {len(all_chunks)}")

    # show sample chunks to verify quality
    print("\n--- Sample Chunks ---")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"\nChunk {i+1} ({all_meta[i]['page']}):")
        print(chunk[:300])
        print("---")

    print("\nEmbedding and storing...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        ids=all_ids,
        metadatas=all_meta
    )

    print(f"\nStored {len(all_chunks)} chunks in ChromaDB collection: '{collection_name}'")


if __name__ == "__main__":
    texts = load_extracted_text()
    store_in_chromadb(texts, "collection_name")