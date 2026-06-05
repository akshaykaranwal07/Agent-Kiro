from groq import Groq
from dotenv import load_dotenv
from tools import get_tools
import json

load_dotenv()
client = Groq()

conversation_history = []


def decide_tool(question: str, tools: dict) -> tuple:
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in tools.items()
    ])

    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        ])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a routing agent. Decide which tool to use.

Available tools:
{tool_descriptions}

Recent conversation:
{history_text}

Reply ONLY with JSON:
{{"tool": "search_notes", "reason": "brief reason"}}
or
{{"tool": "search_web", "reason": "brief reason"}}"""
            },
            {"role": "user", "content": question}
        ],
        max_tokens=100
    )

    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        return result["tool"], result["reason"]
    except:
        return "search_notes", "defaulting to notes"


def answer_question(question: str, context: str, tool_used: str) -> str:
    source = "the user's uploaded PDF" if tool_used == "search_notes" else "web search results"

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.
            Answer based only on context from {source}.
            Be concise and accurate.
            If answer isn't in context, say so clearly."""
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


def run_agent(question: str, collection_name: str) -> tuple:
    global conversation_history

    tools = get_tools(collection_name)
    tool_name, reason = decide_tool(question, tools)
    print(f"Tool: {tool_name} — {reason}")

    tool_fn = tools[tool_name]["function"]
    context = tool_fn(question)

    answer = answer_question(question, context, tool_name)

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    return answer, tool_name