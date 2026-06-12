# eval.py
from agent import run_agent
from dotenv import load_dotenv
from groq import Groq
from prompts import PROMPT_VERSION
import json
import os

load_dotenv()

client          = Groq()
EVAL_COLLECTION = os.getenv("EVAL_COLLECTION", "tenses_notes")

print(f"Running eval on prompt version: {PROMPT_VERSION}")
print(f"Collection: {EVAL_COLLECTION}")

# ── Test cases with categories ──────────────────────────
test_cases = [

    # factual — tests retrieval from notes
    {
        "question": "What are the four aspects of tenses?",
        "expected": "Indefinite, Continuous, Perfect, Perfect Continuous",
        "category": "factual",
        "eval_type": "relevance"
    },
    {
        "question": "What is the indefinite aspect used for?",
        "expected": "Permanent, frequently, general information",
        "category": "factual",
        "eval_type": "relevance"
    },
    {
        "question": "What is the continuous aspect used for?",
        "expected": "Temporary, incomplete, ongoing action",
        "category": "factual",
        "eval_type": "relevance"
    },
    {
        "question": "What is the perfect aspect used for?",
        "expected": "Completed action, degree of work done",
        "category": "factual",
        "eval_type": "relevance"
    },
    {
        "question": "What is tense?",
        "expected": "Time of action, state happening, possession or degree of an action",
        "category": "factual",
        "eval_type": "relevance"
    },

    # grounding — must refuse to hallucinate
    {
        "question": "What is the author's phone number?",
        "expected": "I couldn't find that in the provided content.",
        "category": "grounding",
        "eval_type": "exact"
    },
    {
        "question": "What happened on 15th March 2045?",
        "expected": "I couldn't find that in the provided content.",
        "category": "grounding",
        "eval_type": "exact"
    },

    # web routing — must go to web
    {
        "question": "Who is the current Prime Minister of India?",
        "expected": "Narendra Modi",
        "category": "web_routing",
        "eval_type": "contains"
    },
    {
        "question": "What year was Python created?",
        "expected": "1991",
        "category": "web_routing",
        "eval_type": "contains"
    },

    # vague query — must handle short questions
    {
        "question": "What is the V2 past form of go?",
        "expected": "went",
        "category": "vague_query",
        "eval_type": "contains"
    },
]


# ── Evaluator ───────────────────────────────────────────

def evaluate(question, expected, actual, eval_type):

    if eval_type == "exact":
        correct = expected.lower() in actual.lower()
        reason  = "contains expected phrase" if correct else "missing expected phrase — possible hallucination"
        return correct, reason

    if eval_type == "contains":
        correct = expected.lower() in actual.lower()
        reason  = f"contains '{expected}'" if correct else f"missing '{expected}'"
        return correct, reason

    if eval_type == "not_empty":
        correct = len(actual.strip()) > 20
        reason  = "returned content" if correct else "returned empty"
        return correct, reason

    # relevance — LLM judge
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a strict RAG evaluator.
                Judge if the actual answer correctly addresses the question.

                Rules:
                - If actual says 'context does not mention' or 'not found' → FAIL
                - Minor wording differences → PASS
                - Extra information beyond expected → PASS
                - Missing key information → FAIL
                - Case differences → PASS

                Reply ONLY with JSON:
                {"correct": true, "reason": "brief reason"}
                or
                {"correct": false, "reason": "brief reason"}"""
            },
            {
                "role": "user",
                "content": f"Question: {question}\nExpected: {expected}\nActual: {actual}"
            }
        ],
        max_tokens=100
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
        return result["correct"], result["reason"]
    except:
        return False, "Could not parse evaluator response"


# ── Run eval ────────────────────────────────────────────

def run_eval():
    print("\nRunning RAG Evaluation")
    print("=" * 50)

    results         = []
    correct_count   = 0
    category_scores = {}

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}/{len(test_cases)} [{test['category']}]: {test['question']}")

        try:
            actual, tool_used = run_agent(test["question"], EVAL_COLLECTION)
        except Exception as e:
            actual    = f"ERROR: {e}"
            tool_used = "error"

        is_correct, reason = evaluate(
            test["question"],
            test["expected"],
            actual,
            test["eval_type"]
        )

        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            correct_count += 1

        # track by category
        cat = test["category"]
        if cat not in category_scores:
            category_scores[cat] = {"correct": 0, "total": 0}
        category_scores[cat]["total"] += 1
        if is_correct:
            category_scores[cat]["correct"] += 1

        print(f"Status:    {status}")
        print(f"Tool used: {tool_used}")
        print(f"Expected:  {test['expected']}")
        print(f"Got:       {actual[:150]}...")
        print(f"Reason:    {reason}")
        print("-" * 50)

        results.append({
            "question":  test["question"],
            "category":  test["category"],
            "expected":  test["expected"],
            "actual":    actual,
            "tool_used": tool_used,
            "correct":   is_correct,
            "reason":    reason
        })

    score = (correct_count / len(test_cases)) * 100

    print(f"\n{'=' * 50}")
    print(f"FINAL SCORE: {correct_count}/{len(test_cases)} = {score:.1f}%")
    print(f"\nBy category:")
    for cat, counts in category_scores.items():
        cat_score = (counts["correct"] / counts["total"]) * 100
        print(f"  {cat:25s} → {counts['correct']}/{counts['total']} = {cat_score:.0f}%")

    # save full results for dashboard
    with open("eval_results.json", "w") as f:
        json.dump({
            "collection":      EVAL_COLLECTION,
            "prompt_version":  PROMPT_VERSION,
            "total_score":     score,
            "category_scores": category_scores,
            "results":         results
        }, f, indent=2)

    print("\nResults saved to eval_results.json")
    return score


if __name__ == "__main__":
    run_eval()