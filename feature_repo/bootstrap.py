import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feast import FeatureStore

from feature_repo.features import (
    FEATURE_COLUMNS,
    FEATURE_REFS,
    customer_activity,
    risk_model_v1,
    user,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parent
RUNTIME_REPO = REPO / ".runtime"
DATA_DIR = REPO / "data"
DATA_PATH = DATA_DIR / "customer_activity.parquet"
MODEL_PATH = ROOT / "artifacts" / "risk_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "training_metrics.json"


def make_demo_data() -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(hours=2)
    timestamps = pd.date_range(end=end, periods=120, freq="D")
    rows = []
    for user_id in range(1001, 1033):
        user_offset = user_id - 1001
        for day, event_timestamp in enumerate(timestamps):
            transactions = max(
                0,
                int(round(4 + 2.2 * __import__("math").sin(day / 6 + user_offset))),
            )
            failed_payments = int((day + user_offset) % 9 == 0) + int(
                (2 * day + user_offset) % 31 == 0
            )
            avg_order_value = round(
                38 + ((day * 7 + user_offset * 13) % 145) + 8 * (user_offset % 4),
                2,
            )
            account_age_days = 90 + day + user_offset * 3
            late_payment = int(
                (failed_payments >= 2 and transactions <= 5)
                or failed_payments >= 3
                or (transactions <= 1 and avg_order_value >= 135)
            )
            rows.append(
                {
                    "user_id": user_id,
                    "event_timestamp": event_timestamp,
                    "created_timestamp": event_timestamp + pd.Timedelta(minutes=5),
                    "transactions_7d": transactions,
                    "failed_payments_30d": failed_payments,
                    "avg_order_value": avg_order_value,
                    "account_age_days": account_age_days,
                    "is_late_payment_next_7d": late_payment,
                }
            )
    return pd.DataFrame(rows)


def configure_online_store() -> None:
    source_config = REPO / "feature_store.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["entity_key_serialization_version"] = 2
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        config["online_store"] = {
            "type": "redis",
            "connection_string": redis_url,
        }
    else:
        config["online_store"] = {"path": "data/online_store.db"}
    RUNTIME_REPO.mkdir(parents=True, exist_ok=True)
    config_path = RUNTIME_REPO / "feature_store.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def train_model(store: FeatureStore, events: pd.DataFrame) -> dict:
    entity_rows = events[["user_id", "event_timestamp", "is_late_payment_next_7d"]].copy()
    entity_rows["event_timestamp"] = entity_rows["event_timestamp"] + pd.Timedelta(hours=6)
    training = store.get_historical_features(
        entity_df=entity_rows,
        features=FEATURE_REFS,
    ).to_df()
    training = training.dropna(subset=FEATURE_COLUMNS + ["is_late_payment_next_7d"])
    X = training[FEATURE_COLUMNS]
    y = training["is_late_payment_next_7d"].astype(int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=7),
    )
    model.fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "training_rows": int(len(training)),
        "positive_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
        "feature_service": "risk_model_v1",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def bootstrap_project() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    events = make_demo_data()
    events.to_parquet(DATA_PATH, index=False)
    configure_online_store()
    store = FeatureStore(repo_path=str(RUNTIME_REPO))
    store.apply([user, customer_activity, risk_model_v1])
    start = events["event_timestamp"].min().to_pydatetime()
    end = (events["event_timestamp"].max() + pd.Timedelta(minutes=10)).to_pydatetime()
    store.materialize(start_date=start, end_date=end)
    metrics = train_model(store, events)
    return {
        "users": int(events["user_id"].nunique()),
        "feature_rows": int(len(events)),
        "first_event": events["event_timestamp"].min().isoformat(),
        "latest_event": events["event_timestamp"].max().isoformat(),
        "online_store": "redis" if os.environ.get("REDIS_URL") else "sqlite",
        **metrics,
    }


if __name__ == "__main__":
    print(json.dumps(bootstrap_project(), indent=2))