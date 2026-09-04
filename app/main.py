from dotenv import load_dotenv
load_dotenv()

import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import TransactionIn, ScoreOut
from app.features import build_feature_row, risk_signals
from app.model import model, scaler, feature_columns, threshold, category_baselines
from app.exceptions import InvalidTransactionError, LLMProviderError
from app.llm import explain_score

app = FastAPI(title="Fraud Risk Manager API")


@app.exception_handler(InvalidTransactionError)
async def invalid_transaction_handler(req: Request, exc: InvalidTransactionError):
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_transaction", "message": exc.reason, "trace_id": exc.trace_id},
    )


@app.exception_handler(LLMProviderError)
async def llm_error_handler(req: Request, exc: LLMProviderError):
    return JSONResponse(
        status_code=502,
        content={"error": "llm_provider_error", "message": f"LLM explanation failed: {exc.reason}", "trace_id": exc.trace_id},
    )


@app.post("/score", response_model=ScoreOut)
def score_transaction(txn: TransactionIn):
    trace_id = str(uuid.uuid4())
    X = build_feature_row(txn, feature_columns)
    prob = float(model.predict_proba(X)[:, 1][0])

    importances = model.feature_importances_
    top_idx = importances.argsort()[::-1][:3]
    top_features = [
        {"feature": feature_columns[i], "importance": float(importances[i])}
        for i in top_idx
    ]

    is_flagged = prob >= threshold
    explanation = None
    explanation_source = None
    signals = risk_signals(txn) if is_flagged else []

    if is_flagged:
        # * only call the LLM for flagged transactions — saves cost/latency on the
        # * ~99% of legit traffic that doesn't need an explanation at all.
        # * explain_score never raises now — it degrades to a deterministic
        # * fallback instead of failing the whole request.
        # * pass the real per-transaction risk_signals so the LLM explains
        # * THIS transaction instead of reasoning from global feature importance alone.
        explanation, is_ai = explain_score(top_features, prob, txn.category, txn.amt, trace_id, signals)
        explanation_source = "ai" if is_ai else "fallback"

    return ScoreOut(
        fraud_probability=round(prob, 4),
        is_flagged=is_flagged,
        threshold_used=threshold,
        top_features=top_features,
        risk_signals=signals,
        plain_explanation=explanation,
        explanation_source=explanation_source,
        trace_id=trace_id,
    )


@app.get("/risk-segments")
def get_risk_segments():
    return {"elevated_risk_categories": category_baselines}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "ai_enabled": explain_score.__module__ is not None}