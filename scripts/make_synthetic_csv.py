"""Generate a tiny synthetic CSV matching the user's schema for smoke-testing.

NOT for real training — just enough to verify the pipeline wires together.
Mirrors the user's column names and intentional imbalance:
    - 6 intents (smaller than the real 17 — keeps the smoke test fast)
    - counts range from a tiny class (~8 examples) up to a larger one (~40)
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

random.seed(42)


TEMPLATES: dict[str, list[str]] = {
    "card_lost": [
        "I lost my card",
        "my debit card is missing",
        "I can't find my credit card",
        "where is my card",
        "lost my bank card today",
        "card gone please help",
        "I think my card is stolen",
        "missing card need replacement",
    ],
    "card_declined": [
        "my card was declined",
        "payment failed on my card",
        "transaction denied at store",
        "card not working at checkout",
        "my purchase was rejected",
        "card declined for online order",
        "atm declined my card",
        "merchant says my card was declined",
        "why was my card declined yesterday",
        "card stopped working at the till",
    ],
    "balance_check": [
        "what is my balance",
        "show me my account balance",
        "current balance please",
        "how much money do I have",
        "balance enquiry",
        "tell me my account total",
        "check my balance",
        "what's left in my account",
        "remaining funds",
        "account balance now",
        "show available balance",
        "balance please",
    ],
    "transfer_money": [
        "send money to John",
        "transfer 500 to my brother",
        "move funds to savings",
        "wire 1000 to that account",
        "I want to transfer money",
        "send 200 dollars to Maria",
        "transfer between accounts",
        "move money from checking to savings",
        "make a transfer to vendor",
        "send funds abroad",
        "schedule a recurring transfer",
        "pay 50 to my friend",
        "transfer rupees to dad",
        "send wire to supplier",
    ],
    "loan_apply": [
        "I want to apply for a loan",
        "how do I get a personal loan",
        "apply for home loan",
        "loan application process",
        "I need a 10000 loan",
        "auto loan options",
        "small business loan apply",
        "education loan request",
        "interest rate for personal loan",
        "loan eligibility check",
        "can I take a top up loan",
        "loan apply online",
        "apply mortgage",
        "process to take car loan",
        "loan against fixed deposit",
    ],
    "general_greeting": [
        "hi",
        "hello",
        "hey there",
        "good morning",
        "hi there bot",
        "hey",
        "hello bot",
        "good evening",
        "namaste",
        "yo",
        "hi banking bot",
        "hello can you help",
        "hey hi",
        "morning",
        "good day",
        "greetings",
        "hi everyone",
        "hi how are you",
        "yo bot",
        "salut",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/user_dataset.csv", help="output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history_samples = [
        "",
        "bot: how can I help you today?",
        "user: hi | bot: hi, how can I help?",
        "user: hello | bot: hi there, what do you need?",
    ]

    rows = []
    for intent, queries in TEMPLATES.items():
        for q in queries:
            rows.append(
                {
                    "current_user_query": q,
                    "conversation_history": random.choice(history_samples),
                    "expected_intent": intent,
                }
            )
    random.shuffle(rows)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["current_user_query", "conversation_history", "expected_intent"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows across {len(TEMPLATES)} intents → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
