# Week 3 writeup: prompting techniques for a support-desk assistant

All four starter checks pass with `llama3.1:8b` at temperature 0. I edited only the `TODO`
parts: the four prompts, the tool schema descriptions and the retriever in `rag.py`. Ollama
ran on another machine on my network that already had `llama3.1:8b`, so I pointed the client
at it with `OLLAMA_HOST=http://<host>:11434`. The `ollama` package reads that variable by itself,
so the code and the model name are the same as in the starter.

To see what each prompt actually changed, I first ran every file with the empty starter prompt.
Then, after the checks passed, I tried each prompt on a few new inputs that aren't in `CASES`.
That second step found two real problems, which are fixed in commit `464572c`.

| File | Empty prompt | Final prompt |
|------|--------------|--------------|
| `k_shot_prompting.py` | 0/4 | 4/4 |
| `chain_of_thought.py` | 0/3 | 3/3 |
| `tool_calling.py` | PASS | PASS |
| `rag.py` | 2/3 (the two passes were made up) | 3/3 |

## 1. K-shot prompting: ticket classification

Final prompt:

```text
You are a ticket router for an online store's support desk.
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
(... 8 more examples, two per label and one "other" ...)
```

The examples are my own sentences, not the test cases, so the check measures whether the model
learned the labels rather than whether it can copy.

Sample: `"My card was charged twice for order 4412."` gives `billing`. On new inputs it also
got `"The blender arrived broken, I want to give it back."` as `returns` and
`"Are you hiring for summer internships?"` as `other`.

What I learned: without examples the model answered every ticket with a long, polite reply,
and with a few labeled examples it switched to printing one word, so the examples taught it the
output format as much as the categories.

## 2. Chain of thought: refund eligibility

Final prompt:

```text
You check whether an order is eligible for a refund under the refund policy below.
Reason step by step before deciding. Check every rule in this order and write one short line per step:

Step 1 - Time: how many days ago was it bought? Is that within 30 days?
Step 2 - Condition: is the item unused / in original condition? Any use at all fails this step,
even once: worn, washed, used, installed, assembled. If the order does not mention condition,
assume it is unused.
Step 3 - Final sale: is it a final-sale item? If the order does not mention final sale, assume it is not.
Step 4 - Conclusion: the refund is allowed only if steps 1, 2 and 3 all pass. If any step fails, it is not allowed.

The very last line of your answer must be exactly one of these, with nothing after it:
DECISION: YES
DECISION: NO

(+ one worked example with a used item)
```

Sample input `"Bought 5 days ago but it was a final-sale item."`, output:

```text
Step 1 - Time: 5 days ago, within 30 days. Pass.
Step 2 - Condition: not mentioned, so assume unused. Pass.
Step 3 - Final sale: yes, it is a final-sale item. Fail.
Step 4 - Conclusion: step 3 failed, so no refund.
DECISION: NO
```

My first version passed all three cases but approved `"Bought 2 weeks ago, I wore it once."`.
The reasoning said "the item was worn, but since it was only once, it's still in original
condition." Because I could read that line, I knew exactly which rule to tighten, and adding
"any use at all fails this step, even once" fixed it without breaking the graded cases.

What I learned: step-by-step reasoning made the answer checkable, because when the model was
wrong the mistake was written down in a specific step and I could fix that step in the prompt.

## 3. Tool calling: order lookup

Final tool description: "Look up the current status of a customer's order in the store's order
database. Call this whenever the customer asks where an order is, whether it has shipped, or
anything else about a specific order." The `order_id` parameter says "The order number exactly
as the customer wrote it, digits only and without '#' or the word 'order', e.g. '4412'."

Final system prompt:

```text
You are the support assistant for an online store.
You cannot see order details yourself. When a customer mentions an order number, always call
get_order_status with that number first, before writing any reply. Never guess or invent an
order status. After you get the result, answer in one or two friendly sentences using only the
status the tool returned. If the message contains no order number, do not call any tool; ask
the customer for their order number instead.
```

Sample: `"Hi, can you check on order 4412 for me?"` makes the call `get_order_status(order_id="4412")`,
which returns `"shipped, arriving Friday"`. The starter's check stops there. In a separate test
I sent the result back to the model, and it replied "Your order 4412 has been shipped and is
expected to arrive on Friday." For `"Has #9001 shipped yet?"` it passed `"9001"` without the `#`,
which is what the parameter description asks for.

The check passed even with the empty prompt, since llama3.1 is trained for tool use. The weak
spot is the opposite case: for `"Where is my order?"` the model still calls the tool with
`order_id=""`, even after I told it not to in the system prompt and, in a version I later
reverted, in the tool description too. It does ask for the order number afterwards, once the
tool returns "not found".

What I learned: the model picks up when to call a tool very easily, but the prompt couldn't
stop it from calling with an empty argument, so that check has to live in the code that runs
the tool.

## 4. RAG: answers grounded in the help center

Retriever: it lowercases the question and each line of `help_center.txt`, drops stopwords, and
cuts every word to its first four letters, so "ship" matches "shipping" and "internationally"
matches "international". Lines are ranked by how many keywords they share with the question.
Lines with no overlap are dropped, and at most three are returned.

Final system prompt:

```text
You are the support assistant for an online store.
Answer the customer's question using ONLY the text under "Context". Do not use outside
knowledge and do not guess.

- If the context answers the question, reply in one or two sentences and keep the exact
  numbers and wording from the context (for example "30 days" or "international shipping").
- If the context is empty, or it does not contain the answer, reply exactly:
  I don't know. Please contact our support team for help with that.
- Never make up personal information about anyone.
```

Samples: `"Do you ship internationally?"` retrieves the international shipping line first and
answers "we offer international shipping to most countries; delivery takes 7-21 business days".
`"What's your CEO's home address?"` retrieves nothing and answers "I don't know. Please contact
our support team for help with that." On new questions, "Is there a warranty on clothes?" also
got "I don't know", which is right because the help center only covers electronics.

With the empty starter the check said 2/3, but both passes were invented answers ("the return
policy varies depending on the store..."). They happened to contain the expected keywords,
since the retriever returned nothing. The CEO question failed only because the model wrote
"can't" with a curly apostrophe.

What I learned: the keyword check could not tell a grounded answer from a lucky guess, and the
retriever plus the "only from context" rule changed the answers from plausible guesses to
sentences copied from the help center.

## Closing

LLMs did well whenever the task had a fixed shape I could show them: a label from a list, a
decision at the end of numbered steps, a function with a typed argument, or an answer copied
from a given paragraph. Examples and output rules moved this 8B model from 0 to all passing
with no code changes. It needed help from code in three places. It talked itself into approving
a worn item until the rule spelled out that one use counts. It kept calling the tool with an
empty order id however I phrased the prompt, so the code that runs the tool has to check the
argument. And without a retriever it made up store policy with full confidence. None of these
showed up in the graded checks; I only found them by trying inputs of my own.
