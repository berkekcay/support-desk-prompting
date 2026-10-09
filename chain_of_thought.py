"""Chain-of-thought — refund eligibility.

Task: given the refund policy and an order, decide YES or NO. Use chain-of-thought
so the model reasons step by step, but make the FINAL line exactly `DECISION: YES`
or `DECISION: NO` so it can be graded.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python chain_of_thought.py
"""
from ollama import chat

MODEL = "llama3.1:8b"

POLICY = """Refund policy:
- Refunds allowed within 30 days of purchase.
- Item must be unused / in original condition.
- Final-sale items are never refundable.
"""

YOUR_SYSTEM_PROMPT = """You check whether an order is eligible for a refund under the refund policy below.
Reason step by step before deciding. Check every rule in this order and write one short line per step:

Step 1 - Time: how many days ago was it bought? Is that within 30 days?
Step 2 - Condition: is the item unused / in original condition? Any use at all fails this step, even once: worn, washed, used, installed, assembled. If the order does not mention condition, assume it is unused.
Step 3 - Final sale: is it a final-sale item? If the order does not mention final sale, assume it is not.
Step 4 - Conclusion: the refund is allowed only if steps 1, 2 and 3 all pass. If any step fails, it is not allowed.

The very last line of your answer must be exactly one of these, with nothing after it:
DECISION: YES
DECISION: NO

Example:
Order: Bought 3 days ago, the box was opened and the item was used twice.
Step 1 - Time: 3 days ago, within 30 days. Pass.
Step 2 - Condition: the item was used. Fail.
Step 3 - Final sale: not mentioned, so not final sale. Pass.
Step 4 - Conclusion: step 2 failed, so no refund.
DECISION: NO"""

CASES = [
    ("Bought 10 days ago, unused, not final sale.", "YES"),
    ("Bought 45 days ago, unused.", "NO"),
    ("Bought 5 days ago but it was a final-sale item.", "NO"),
]


def decide(system_prompt: str, order: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt + "\n" + POLICY},
            {"role": "user", "content": order},
        ],
        options={"temperature": 0.0},
    )
    text = resp.message.content.strip()
    last = text.splitlines()[-1].upper() if text else ""
    return "YES" if "DECISION: YES" in last else "NO" if "DECISION: NO" in last else "?"


def run() -> bool:
    passed = 0
    for order, expected in CASES:
        got = decide(YOUR_SYSTEM_PROMPT, order)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {order!r} -> {got}, expected {expected}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
