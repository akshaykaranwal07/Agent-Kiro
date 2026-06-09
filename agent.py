import os
import json
from groq import Groq
from dotenv import load_dotenv
from tools import get_tools
from config import LLM_MODEL

load_dotenv()

client = Groq()
MODEL  = LLM_MODEL


def decide_tool(question: str, tools: dict, history: list = None) -> tuple:
    if history is None:
        history = []

    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in tools.items()
    ])

    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        ])

    prompt = f"""You are a routing agent. Decide which tool to use.

Available tools:
{tool_descriptions}

Recent conversation:
{history_text}

Question: {question}

Reply ONLY with JSON, nothing else:
{{"tool": "search_notes", "reason": "brief reason"}}
or
{{"tool": "search_web", "reason": "brief reason"}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
        return result["tool"], result["reason"]
    except:
        return "search_notes", "defaulting to notes"


def answer_question(question: str, context: str, tool_used: str, history: list = None) -> str:
    if history is None:
        history = []

    source = "the user's uploaded PDF" if tool_used == "search_notes" else "web search results"

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.
Answer based only on the context provided from {source}.
Be concise and accurate.
If the answer isn't in the context, say so clearly."""
        }
    ]

    if history:
        messages.extend(history[-6:])

    messages.append({
        "role": "user",
        "content": f"Context from {source}:\n{context}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return response.choices[0].message.content


def run_agent(question: str, collection_name: str, history: list = None) -> tuple:
    if history is None:
        history = []

    tools     = get_tools(collection_name)
    tool_name, reason = decide_tool(question, tools, history)
    print(f"Tool: {tool_name} — {reason}")

    tool_fn = tools[tool_name]["function"]
    context = tool_fn(question)

    answer = answer_question(question, context, tool_name, history)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    if len(history) > 20:
        del history[:2]

    return answer, tool_name