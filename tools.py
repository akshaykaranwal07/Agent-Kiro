import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from config import CHROMA_PATH, COLLECTION_NAME, N_RESULTS

# load embedding model once
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#tool
def search_notes(query: str) -> str:
    """Search your PDF notes using semantic search"""
    try:
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)

        query_embedding = embedder.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS
        )

        chunks = results["documents"][0]
        pages  = [m["page"] for m in results["metadatas"][0]]

        if not chunks:
            return "No relevant content found in notes."

        # format results with page references
        formatted = []
        for chunk, page in zip(chunks, pages):
            formatted.append(f"[{page}]\n{chunk}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Error searching notes: {e}"


def search_web(query: str) -> str:
    """Search the web using DuckDuckGo"""
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


# tool registry — agent uses this to know what tools exist
TOOLS = {
    "search_notes": {
        "function":    search_notes,
        "description": "Search the user's personal PDF notes. Use this for questions about the document content, study notes, or anything the user might have written in their notes."
    },
    "search_web": {
        "function":    search_web,
        "description": "Search the web for current information. Use this for general knowledge questions, recent events, facts not likely in personal notes."
    }
}