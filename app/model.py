from dotenv import load_dotenv
load_dotenv()

import pickle
import json
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent.parent / "model_artifacts"

with open(ARTIFACT_DIR / "fraud_model.pkl", "rb") as f:
    artifact = pickle.load(f)

with open(ARTIFACT_DIR / "category_baselines.json") as f:
    category_baselines = json.load(f)

model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
threshold = artifact["threshold"]