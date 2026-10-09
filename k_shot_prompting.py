"""K-shot prompting — ticket classification.

Task: classify a customer message into one of: billing, shipping, returns,
technical, other. Use in-context EXAMPLES (k-shot) so the model outputs exactly the
label, lowercase, with no extra text.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python k_shot_prompting.py
"""
from ollama import chat

MODEL = "llama3.1:8b"
NUM_RUNS = 5

YOUR_SYSTEM_PROMPT = """You are a ticket router for an online store's support desk.
Classify each customer message into exactly one label:

billing   - payments, charges, refunds of money already paid, invoices, payment methods
shipping  - delivery status, tracking, late or lost packages, shipping options
returns   - sending an item back, exchanges, return labels, return eligibility
technical - the website or app not working: crashes, errors, login problems, bugs
other     - anything that fits none of the above

Reply with the label only: one lowercase word, no punctuation, no explanation.

Examples:
Message: I was billed for a subscription I cancelled last month.
Label: billing

Message: Can I pay with PayPal instead of a credit card?
Label: billing

Message: The tracking number you sent doesn't show any updates.
Label: shipping

Message: My parcel says delivered but it never arrived.
Label: shipping

Message: The jacket is too small, how do I return it?
Label: returns

Message: Can I exchange this lamp for a different color?
Label: returns

Message: I get an error 500 when I click checkout.
Label: technical

Message: I can't log in, the password reset email never comes.
Label: technical

Message: Do you have a physical store in Istanbul?
Label: other"""

CASES = [
    ("My card was charged twice for order 4412.", "billing"),
    ("Where is my package? It's been two weeks.", "shipping"),
    ("I want to send the shoes back for a refund.", "returns"),
    ("The app crashes every time I open my cart.", "technical"),
]


def classify(system_prompt: str, message: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for message, expected in CASES:
        got = classify(YOUR_SYSTEM_PROMPT, message)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {message!r} -> got {got!r}, expected {expected!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
