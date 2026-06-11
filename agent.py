import os
import json
from groq import Groq
from dotenv import load_dotenv
from tools import get_tools
from config import LLM_MODEL
from prompts import ROUTING_PROMPT, ANSWER_PROMPT, QUERY_REWRITE_PROMPT, COT_PROMPT, PROMPT_VERSION

load_dotenv()

client = Groq()
MODEL  = LLM_MODEL

def rewrite_query(question: str, history: list = None) -> str:
    """Transform vague question into specific search query"""
    if history is None:
        history = []

    # only rewrite if question is vague (short or lacks context)
    if len(question.split()) > 8:
        return question  # already specific enough

    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        ])

    prompt = QUERY_REWRITE_PROMPT.format(
        history=history_text or "No previous conversation",
        question=question
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"Query rewrite: '{question}' → '{rewritten}'")
    return rewritten


def should_use_cot(question: str) -> bool:
    """Detect if question needs step by step reasoning"""
    cot_triggers = [
        "compare", "difference", "why", "explain",
        "how does", "analyze", "versus", "vs"
    ]
    return any(trigger in question.lower() for trigger in cot_triggers)


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

    # CHANGED: use prompt from prompts.py
    prompt = ROUTING_PROMPT.format(
        history=history_text or "No previous conversation",
        tools=tool_descriptions,
        question=question
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0  # ADDED: deterministic routing, no randomness
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
        return result["tool"], result["reason"]
    except:
        return "search_notes", "defaulting to notes"


def answer_question(question: str, context: str, tool_used: str, history: list = None, use_cot: bool = False) -> str:
    if history is None:
        history = []

    source = "the user's uploaded PDF" if tool_used == "search_notes" else "web search results"

    if use_cot:
        print(f"Using chain of thought")
        user_message = COT_PROMPT.format(
            source=source,
            context=context,
            question=question
        )

    else:
        user_message = ANSWER_PROMPT.format(
        source=source,
        context=context,
        question=question
    )

    messages = [
        {
            "role": "system",
            "content": f"You are Kiro v{PROMPT_VERSION}, a precise AI assistant. Never hallucinate."
        }
    ]

    if history:
        messages.extend(history[-6:])

    messages.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )

    return response.choices[0].message.content


def run_agent(question: str, collection_name: str, history: list = None) -> tuple:

    print(f"Prompt version: {PROMPT_VERSION}")  # ADD THIS


    if history is None:
        history = []

    search_query=rewrite_query(question,history)

    tools     = get_tools(collection_name)
    tool_name, reason = decide_tool(search_query, tools, history)
    print(f"Tool: {tool_name} — {reason}")

    tool_fn = tools[tool_name]["function"]
    context = tool_fn(search_query)

    use_cot=should_use_cot(question)

    answer = answer_question(question, context, tool_name, history,use_cot)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    if len(history) > 20:
        del history[:2]

    return answer, tool_name