import chromadb
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
from config import CHROMA_PATH, N_RESULTS

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def search_notes(query: str, collection_name: str) -> str:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(collection_name)
        query_embedding = embedder.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS
        )

        chunks = results["documents"][0]
        pages  = [m["page"] for m in results["metadatas"][0]]

        if not chunks:
            return "No relevant content found in notes."

        formatted = []
        for chunk, page in zip(chunks, pages):
            formatted.append(f"[{page}]\n{chunk}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Error searching notes: {e}"


def search_web(query: str) -> str:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=4):
                results.append(f"Source: {r['href']}\n{r['body']}")
        if not results:
            return "No web results found."
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching web: {e}"


def get_tools(collection_name: str):
    return {
        "search_notes": {
            "function": lambda q: search_notes(q, collection_name),
            "description": "Search the user's uploaded PDF notes. Use for questions about the document content."
        },
        "search_web": {
            "function": search_web,
            "description": "Search the web for current information, facts, or anything not in the uploaded document."
        }
    }