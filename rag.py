"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be",
    "do", "does", "did", "i", "you", "your", "we", "our", "my", "me", "it", "its", "can",
    "have", "has", "how", "what", "what's", "whats", "when", "where", "which", "who", "why",
    "there", "this", "that", "any", "if", "at", "by", "from", "as", "will", "would", "am",
}


def _keywords(text: str) -> set[str]:
    """Lowercase content words, cut to a 4-letter prefix so 'ship' matches 'shipping'
    and 'internationally' matches 'international'."""
    cleaned = "".join(c if c.isalnum() or c == "'" else " " for c in text.lower())
    words = cleaned.split()
    return {w[:4] for w in words if w not in STOPWORDS and len(w) > 2}


# Keyword-overlap retriever: rank chunks by shared keywords, drop chunks with no overlap.
def YOUR_RETRIEVER(question: str, chunks: list[str], top_k: int = 3) -> list[str]:
    query = _keywords(question)
    scored = [(len(query & _keywords(chunk)), chunk) for chunk in chunks]
    ranked = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: -pair[0])
    return [chunk for _, chunk in ranked[:top_k]]


YOUR_SYSTEM_PROMPT = """You are the support assistant for an online store.
Answer the customer's question using ONLY the text under "Context". Do not use outside
knowledge and do not guess.

- If the context answers the question, reply in one or two sentences and keep the exact
  numbers and wording from the context (for example "30 days" or "international shipping").
- If the context is empty, or it does not contain the answer, reply exactly:
  I don't know. Please contact our support team for help with that.
- Never make up personal information about anyone."""

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
