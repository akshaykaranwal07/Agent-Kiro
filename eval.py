# eval.py
# evaluate RAG accuracy against known test cases

from query import query_rag
from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()

# ── add your test cases here ─────────────────────────────
test_cases = [
    {
        "question": "What are the four aspects of tenses?",
        "expected": "Indefinite, Continuous, Perfect, Perfect Continuous"
    },
    {
        "question": "What is the indefinite aspect used for?",
        "expected": "Permanent, frequently, general information"
    },
    {
        "question": "What is the continuous aspect used for?",
        "expected": "Temporary, incomplete, ongoing action"
    },
    {
        "question": "What is the perfect aspect used for?",
        "expected": "Completed action, degree of work done"
    },
    {
        "question": "What is perfect continuous aspect?",
        "expected": "Partially incomplete and partially complete action"
    },
    {
        "question": "What is tense?",
        "expected": "Time of action, state happening, possession or degree of an action"
    },
    {
        "question": "What is the V2 past form of go?",
        "expected": "went"
    },
    {
        "question": "What is the V3 perfect form of run?",
        "expected": "run"
    },
    {
        "question": "What is the V4 continuous form of come?",
        "expected": "coming"
    },
    {
        "question": "What is the V5 singular form of explain?",
        "expected": "explains"
    },
]
# ─────────────────────────────────────────────────────────


def evaluate_with_llm(question, expected, actual):
    client = Groq()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a strict RAG evaluator.
                Judge if the actual answer correctly answers the question.

                Rules:
                - If actual says 'context does not mention' or 'not found' → always FAIL
                - Minor wording differences → PASS
                - Extra information beyond expected → PASS
                - Missing key information → FAIL
                - Case differences → PASS

                Reply ONLY with JSON, nothing else:
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

    try:
        result = json.loads(raw)
        return result["correct"], result["reason"]
    except:
        return False, "Could not parse evaluator response"


def run_eval():
    print("Running RAG Evaluation")
    print("=" * 50)

    correct_count = 0
    results       = []

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}/{len(test_cases)}: {test['question']}")

        actual = query_rag(test["question"])
        is_correct, reason = evaluate_with_llm(
            test["question"],
            test["expected"],
            actual
        )

        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            correct_count += 1

        print(f"Status:   {status}")
        print(f"Expected: {test['expected']}")
        print(f"Got:      {actual[:150]}...")
        print(f"Reason:   {reason}")
        print("-" * 50)

        results.append({
            "question":  test["question"],
            "expected":  test["expected"],
            "actual":    actual,
            "correct":   is_correct,
            "reason":    reason
        })

    score = (correct_count / len(test_cases)) * 100
    print(f"\nFINAL SCORE: {correct_count}/{len(test_cases)} = {score:.1f}%")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to eval_results.json")

    return score


if __name__ == "__main__":
    run_eval()