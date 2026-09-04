from pydantic import BaseModel
from datetime import datetime


class TransactionIn(BaseModel):
    trans_date_trans_time: datetime
    cc_num: str
    category: str
    amt: float
    gender: str
    dob: datetime
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int
    # * prior transaction context — normally you'd look this up from a store keyed
    # * by cc_num; for the demo API we accept it directly so /score is stateless
    prev_trans_time: datetime | None = None
    card_amt_mean: float
    card_amt_std: float


class ScoreOut(BaseModel):
    fraud_probability: float
    is_flagged: bool
    threshold_used: float
    top_features: list[dict]
    risk_signals: list[str] = []
    plain_explanation: str | None = None
    explanation_source: str | None = None  # "ai" or "fallback"
    trace_id: str