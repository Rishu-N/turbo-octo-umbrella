"""Phase 1 tests: typed-placeholder multilingual normalization (src/preprocess/normalize.py)."""

from __future__ import annotations

import copy

import pytest

from src.config import load_config
from src.preprocess.normalize import normalize


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


# --- basics ------------------------------------------------------------------------------------
def test_empty_and_none(cfg):
    assert normalize("", cfg) == ""
    assert normalize(None, cfg) == ""


def test_lowercase_and_punctuation(cfg):
    assert normalize("What's my account balance?", cfg) == "whats my account balance"
    assert normalize("I lost my card, please block it", cfg) == "i lost my card please block it"


# --- card vs phone vs amount (the headline disambiguation) -------------------------------------
def test_card(cfg):
    assert normalize("my card 4111 1111 1111 1111 is not working", cfg) == "my card <CARD> is not working"
    assert normalize("card 4111111111111111 blocked", cfg) == "card <CARD> blocked"


def test_phone(cfg):
    assert normalize("update my phone number to 9876543210", cfg) == "update my phone number to <PHONE>"
    assert "<PHONE>" in normalize("call +91 98765 43210", cfg)


def test_amount(cfg):
    assert normalize("an unknown transaction of $500", cfg) == "an unknown transaction of <AMOUNT>"
    assert normalize("transfer 5000 rupees", cfg) == "transfer <AMOUNT>"


def test_card_phone_amount_are_distinct(cfg):
    out = normalize("card 4111 1111 1111 1111 phone 9876543210 amount $500", cfg)
    assert "<CARD>" in out and "<PHONE>" in out and "<AMOUNT>" in out


# --- meaningful tokens preserved (the number IS the signal) ------------------------------------
def test_zero_amount_preserved(cfg):
    out = normalize("why is there a $0.00 charge on my statement", cfg)
    assert "0.00" in out
    assert "<AMOUNT>" not in out and "<NUM>" not in out


def test_twice_preserved(cfg):
    out = normalize("I was charged twice for the same order", cfg)
    assert "twice" in out and "<NUM>" not in out


def test_multi_entity(cfg):
    out = normalize("My card ending 1234 was charged $0.00 twice", cfg)
    assert "<NUM>" in out          # 1234 -> <NUM> (4 digits, not a card)
    assert "0.00" in out           # $0.00 preserved (allowlisted)
    assert "twice" in out
    assert "<CARD>" not in out and "<AMOUNT>" not in out


# --- other typed placeholders ------------------------------------------------------------------
def test_email(cfg):
    assert normalize("change my email to john@example.com please", cfg) == "change my email to <EMAIL> please"


def test_num(cfg):
    assert normalize("why was I charged a 199 fee", cfg) == "why was i charged a <NUM> fee"


def test_date(cfg):
    assert normalize("my statement for 12/05/2024", cfg) == "my statement for <DATE>"


def test_month(cfg):
    assert normalize("send me the April statement", cfg) == "send me the <MONTH> statement"


def test_id(cfg):
    assert normalize("track transaction TXN12345 please", cfg) == "track transaction <ID> please"


# --- multilingual: Hindi (Devanagari) & Hinglish (Romanized) -----------------------------------
def test_hindi_passthrough(cfg):
    assert normalize("मेरा बैलेंस कितना है।", cfg) == "मेरा बैलेंस कितना है"


def test_devanagari_numerals_to_phone(cfg):
    assert normalize("मेरा नंबर ९८७६५४३२१० है", cfg) == "मेरा नंबर <PHONE> है"


def test_hindi_month(cfg):
    assert normalize("मुझे अप्रैल का स्टेटमेंट चाहिए", cfg) == "मुझे <MONTH> का स्टेटमेंट चाहिए"


def test_hinglish_passthrough(cfg):
    assert normalize("mera card kaam nahi kar raha", cfg) == "mera card kaam nahi kar raha"


def test_hinglish_card(cfg):
    assert normalize("mera card 4111111111111111 block karo", cfg) == "mera card <CARD> block karo"


# --- configurability ---------------------------------------------------------------------------
def test_allowlist_configurable(cfg):
    c = copy.deepcopy(cfg)
    c["normalization"]["meaningful_tokens_allowlist"] = []   # drop 0.00 from the allowlist
    out = normalize("why is there a $0.00 charge", c)
    assert "<AMOUNT>" in out and "0.00" not in out


def test_placeholder_toggle(cfg):
    c = copy.deepcopy(cfg)
    c["normalization"]["placeholders"]["card"] = False
    out = normalize("my card 4111 1111 1111 1111 here", c)
    assert "<CARD>" not in out


def test_partial_config_defaults():
    # Works with an empty config (sensible defaults kick in).
    assert normalize("Hello, World 123!", {}) == "hello world <NUM>"


# --- robustness --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        "My card ending 1234 was charged $0.00 twice",
        "update my phone number to 9876543210",
        "मुझे अप्रैल का स्टेटमेंट चाहिए",
        "transfer 5000 to 9876543210 on 12/05/2024",
    ],
)
def test_idempotent(cfg, q):
    once = normalize(q, cfg)
    assert normalize(once, cfg) == once


def test_deterministic(cfg):
    q = "transfer 5000 to 9876543210 on 12/05/2024"
    assert normalize(q, cfg) == normalize(q, cfg)


@pytest.mark.parametrize("q", ["yes do it", "the second one", "what about for savings"])
def test_referential_short_queries_have_no_placeholders(cfg, q):
    # Short referential queries carry no entities; they are just cleaned/lowercased (no tags inserted).
    assert "<" not in normalize(q, cfg)
