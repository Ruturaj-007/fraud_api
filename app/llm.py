import os

from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Lazy/optional client — a missing or bad key must never crash app startup
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def fallback_explanation(
    top_features: list[dict],
    fraud_probability: float,
    category: str,
    amt: float,
) -> str:
    """Deterministic, still-useful explanation built from actual model signals.

    Used whenever the LLM is unavailable, unconfigured, or errors out.
    """

    reasons = []

    for f in top_features:
        name = f["feature"]

        if name in ("amt", "amt_zscore_per_card"):
            reasons.append(
                "the amount is unusual compared to this card's normal spending"
            )

        elif name == "hour":
            reasons.append("it occurred at an unusual hour")

        elif name == "seconds_since_last_trans":
            reasons.append(
                "it happened very soon after a previous transaction (high velocity)"
            )

        elif name.startswith("category_"):
            reasons.append(
                f"the '{category}' category has a higher historical fraud rate"
            )

        elif name == "city_pop":
            reasons.append(
                "the location's population profile is atypical for this card"
            )

    if not reasons:
        reasons = [
            "multiple model risk signals crossed the flagging threshold"
        ]

    reason_text = "; ".join(dict.fromkeys(reasons))  # dedupe, keep order

    return (
        f"This ₹{amt:,.0f} {category.replace('_', ' ')} transaction was flagged with "
        f"{fraud_probability:.0%} fraud probability because {reason_text}."
    )


def explain_score(
    top_features: list[dict],
    fraud_probability: float,
    category: str,
    amt: float,
    trace_id: str,
    signals: list[str] | None = None,
) -> tuple[str, bool]:
    """Returns (explanation_text, was_ai_generated). Never raises — a failed
    LLM call falls back to a deterministic explanation instead of breaking /score.
    """

    if client is None:
        return (
            fallback_explanation(
                top_features,
                fraud_probability,
                category,
                amt,
            ),
            False,
        )

    relevant_features = [
        f
        for f in top_features
        if not f["feature"].startswith("category_")
        or f["feature"] == f"category_{category}"
    ]

    feature_summary = (
        ", ".join(
            f"{f['feature']} (importance {f['importance']:.2f})"
            for f in relevant_features
        )
        or "general behavioral anomaly"
    )

    signal_text = "\n".join(f"- {s}" for s in signals) if signals else "- No specific signals surfaced"

    prompt = (
        "You are a fraud analyst assistant. Write a short risk-assessment note "
        "for a flagged transaction, for a human fraud analyst to read before deciding "
        "whether to approve or block it.\n\n"
        f"Transaction facts:\n"
        f"- Category: {category}\n"
        f"- Amount: Rs {amt}\n"
        f"- Model fraud probability: {fraud_probability:.0%}\n\n"
        f"Actual risk signals detected for THIS specific transaction:\n"
        f"{signal_text}\n\n"
        f"Supporting model feature importances (general context, not transaction-specific):\n"
        f"{feature_summary}\n\n"
        "Instructions:\n"
        "- Base your explanation primarily on the ACTUAL risk signals above — they are "
        "specific to this transaction. Use the feature importances only as supporting context.\n"
        "- Write 2 to 4 sentences (40-80 words total).\n"
        "- Sentence 1: state what looks risky and why, in plain business language.\n"
        "- Sentence 2: explain the practical implication (e.g. what pattern this "
        "usually indicates — stolen card, account takeover, unusual one-off purchase, etc. "
        "— only if it's a reasonable inference from the signals given, not a guess).\n"
        "- Optionally, a short closing sentence suggesting what the analyst should check next.\n"
        "- Only reference the category and signals listed above — do not mention any "
        "other transaction category or invent numeric details not given.\n"
        "- Translate feature names into plain business language "
        "(e.g. 'amt_zscore_per_card' -> 'unusually high spend for this card').\n"
        "- Be factual and specific to this transaction, not generic filler.\n"
        "- Output ONLY the note text, no preamble, no markdown, no labels."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )

        text = (completion.choices[0].message.content or "").strip()

        if not text:
            raise ValueError("empty completion")

        return text, True

    except Exception:
        return (
            fallback_explanation(
                top_features,
                fraud_probability,
                category,
                amt,
            ),
            False,
        )