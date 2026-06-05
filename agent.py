# agent.py
from groq import Groq
from dotenv import load_dotenv
from tools import TOOLS
import json

load_dotenv()
client = Groq()

# conversation history — persists during session
conversation_history = []


def decide_tool(question: str) -> str:
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    ])

    # include recent history so agent understands context
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]  # last 2 exchanges
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        ])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a routing agent. Decide which tool to use to answer the question.

Available tools:
{tool_descriptions}

Recent conversation:
{history_text}

Reply ONLY with JSON:
{{"tool": "search_notes", "reason": "brief reason"}}
or
{{"tool": "search_web", "reason": "brief reason"}}

Nothing else."""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=100
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        return result["tool"], result["reason"]
    except:
        return "search_notes", "defaulting to notes search"


def answer_question(question: str, context: str, tool_used: str) -> str:
    source = "the user's personal notes" if tool_used == "search_notes" else "web search results"

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.
            Answer based only on the provided context from {source}.
            - Be concise and accurate
            - If answer isn't in context, say so clearly
            - Do not mix information from different sources"""
        }
    ]

    messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": f"Context from {source}:\n{context}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    return response.choices[0].message.content


def run_agent(question: str) -> str:
    global conversation_history

    print(f"\nQuestion: {question}")

    # step 1 — decide tool
    tool_name, reason = decide_tool(question)
    print(f"Tool selected: {tool_name} ({reason})")

    # step 2 — run tool
    tool_fn = TOOLS[tool_name]["function"]
    print("Searching...")
    context = tool_fn(question)

    # step 3 — generate answer
    answer = answer_question(question, context,tool_name)

    # step 4 — save to history
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    # keep history manageable — last 10 exchanges
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    return answer,tool_name


if __name__ == "__main__":
    print("Agentic RAG — type 'exit' to quit")
    print("Type 'clear' to reset conversation\n")

    while True:
        question = input("\nAsk anything: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "clear":
            conversation_history = []
            print("Conversation cleared.")
            continue

        answer = run_agent(question)
        print(f"\nAnswer: {answer}")