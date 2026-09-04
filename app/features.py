import numpy as np
import pandas as pd
from app.schemas import TransactionIn


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def compute_amt_zscore(txn: TransactionIn) -> float:
    std = txn.card_amt_std if txn.card_amt_std != 0 else 1.0
    return (txn.amt - txn.card_amt_mean) / std


def build_feature_row(txn: TransactionIn, feature_columns: list[str]) -> pd.DataFrame:
    # * mirrors the notebook's engineer_features() logic, but for a single
    # * incoming transaction instead of a bulk dataframe
    age = (txn.trans_date_trans_time - txn.dob).days // 365
    hour = txn.trans_date_trans_time.hour
    day_of_week = txn.trans_date_trans_time.weekday()

    if txn.prev_trans_time:
        seconds_since_last = (txn.trans_date_trans_time - txn.prev_trans_time).total_seconds()
    else:
        seconds_since_last = 32538.0  # * median fallback from training data

    row = {
        "amt": txn.amt,
        "hour": hour,
        "day_of_week": day_of_week,
        "age": age,
        "seconds_since_last_trans": seconds_since_last,
        "amt_zscore_per_card": compute_amt_zscore(txn),
        "city_pop": txn.city_pop,
        f"category_{txn.category}": 1,
        f"gender_{txn.gender}": 1,
    }

    df = pd.DataFrame([row])
    return df.reindex(columns=feature_columns, fill_value=0)


def risk_signals(txn: TransactionIn) -> list[str]:
    """Plain-English, PER-TRANSACTION risk signals computed from this
    transaction's actual values — distinct from the model's global
    feature-importance ranking. Uses haversine_km, which previously
    existed but was never called anywhere."""
    signals = []

    if abs(compute_amt_zscore(txn)) >= 2:
        signals.append("Transaction amount is unusually high for this card's typical spending")

    hour = txn.trans_date_trans_time.hour
    if hour <= 5 or hour >= 23:
        signals.append("Transaction occurred at an unusual hour")

    if txn.prev_trans_time:
        gap_seconds = (txn.trans_date_trans_time - txn.prev_trans_time).total_seconds()
        if gap_seconds < 120:
            signals.append("Very short gap since the previous transaction (high velocity)")

    distance = haversine_km(txn.lat, txn.long, txn.merch_lat, txn.merch_long)
    if distance > 100:
        signals.append(f"Merchant location is ~{distance:.0f} km from the cardholder's last known location")

    if not signals:
        signals.append("Multiple model risk signals combined to cross the flagging threshold")
    return signals