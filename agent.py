import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tools import get_tools

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash"

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

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    raw = response.text.strip()

    # clean markdown code blocks if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
        return result["tool"], result["reason"]
    except:
        return "search_notes", "defaulting to notes"


def answer_question(question: str, context: str, tool_used: str) -> str:
    source = "the user's uploaded PDF" if tool_used == "search_notes" else "web search results"

    # build history as a single string for Gemini
    history_text = ""
    if conversation_history:
        for m in conversation_history[-6:]:
            history_text += f"{m['role'].upper()}: {m['content']}\n\n"

    prompt = f"""You are a helpful assistant.
Answer based only on the context provided from {source}.
Be concise and accurate.
If the answer isn't in the context, say so clearly.

{f"Previous conversation:{chr(10)}{history_text}" if history_text else ""}

Context from {source}:
{context}

Question: {question}"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


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