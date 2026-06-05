# query.py
# ask questions against your PDF

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from config import CHROMA_PATH, COLLECTION_NAME, N_RESULTS, LLM_MODEL

load_dotenv()

# load once at startup
print("Loading models...")
embedder    = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq()
print("Ready!\n")


def query_rag(question):
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedder.encode([question]).tolist()

    # retrieve more chunks for comparison questions
    n = N_RESULTS
    if any(word in question.lower() for word in ["difference", "compare", "vs", "between"]):
        n = n + 3

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n
    )

    context = "\n\n".join(results["documents"][0])

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant answering questions from a document.
                Answer based only on the provided context.
                - Be concise and accurate
                - Include relevant details, formulas, or examples if available
                - If the answer is not in the context, say so clearly"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print(f"RAG — chatting with: {COLLECTION_NAME}")
    print("Type 'exit' to quit\n")

    while True:
        question = input("Ask a question: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        answer = query_rag(question)
        print(f"\nAnswer: {answer}\n")