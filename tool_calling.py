"""Tool calling — order lookup.

Task: the model must decide to call the `get_order_status` tool with the order id it
finds in the user's message, then use the returned status in a short reply.

Fill in YOUR_SYSTEM_PROMPT and the tool schema TODO.
Run:  python tool_calling.py
"""
from ollama import chat

MODEL = "llama3.1:8b"

FAKE_DB = {
    "4412": "shipped, arriving Friday",
    "9001": "processing",
}


def get_order_status(order_id: str) -> str:
    return FAKE_DB.get(order_id, "not found")


# TODO: describe the tool so the model knows when and how to call it.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up the current status of a customer's order in the store's order "
                "database. Call this whenever the customer asks where an order is, whether "
                "it has shipped, or anything else about a specific order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": (
                            "The order number exactly as the customer wrote it, digits only "
                            "and without '#' or the word 'order', e.g. '4412'."
                        ),
                    },
                },
                "required": ["order_id"],
            },
        },
    }
]

YOUR_SYSTEM_PROMPT = """You are the support assistant for an online store.
You cannot see order details yourself. When a customer mentions an order number, always call
get_order_status with that number first, before writing any reply. Never guess or invent an
order status. After you get the result, answer in one or two friendly sentences using only the
status the tool returned. If the customer did not give an order number, ask for it."""


def run() -> bool:
    message = "Hi, can you check on order 4412 for me?"
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        tools=TOOLS,
        options={"temperature": 0.0},
    )
    calls = resp.message.tool_calls or []
    if not calls:
        print("[FAIL] model did not call the tool")
        return False
    call = calls[0]
    order_id = str(call.function.arguments.get("order_id"))
    ok = order_id == "4412"
    print(f"[{'PASS' if ok else 'FAIL'}] tool called with order_id={order_id!r}")
    if ok:
        print("tool result:", get_order_status(order_id))
    return ok


if __name__ == "__main__":
    run()
