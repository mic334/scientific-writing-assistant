from __future__ import annotations


JOURNAL_RULES = {
    "Chemistry - A European Journal": {
        "max_words": 200,
        "style": "formal",
        "focus": "balanced novelty and results",
    },
    "Angewandte Chemie": {
        "max_words": 180,
        "style": "concise",
        "focus": "high novelty and impact",
    },
    "Journal of Physical Chemistry A": {
        "max_words": 200,
        "style": "precise",
        "focus": "clear mechanistic interpretation and physical insight",
    },
    "Generic Scientific Journal": {
        "max_words": 220,
        "style": "neutral",
        "focus": "clear methods and results",
    },
}


def get_journal_rules(journal: str) -> dict[str, str | int]:
    return JOURNAL_RULES.get(journal, JOURNAL_RULES["Generic Scientific Journal"])


def list_supported_journals() -> list[str]:
    return sorted(JOURNAL_RULES)
