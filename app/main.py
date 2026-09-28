from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from feature_repo.bootstrap import DATA_PATH, MODEL_PATH, RUNTIME_REPO, bootstrap_project
from feature_repo.features import FEATURE_COLUMNS, FEATURE_REFS, risk_model_v1
from feast import FeatureStore


STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.metadata = bootstrap_project()
    application.state.store = FeatureStore(repo_path=str(RUNTIME_REPO))
    application.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Retail Risk Feature Service",
    version="1.0.0",
    description="Point-in-time training and online Feast features for retail risk scoring.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    user_id: int = Field(ge=1001, le=1032)


class PointInTimeRequest(BaseModel):
    user_id: int = Field(ge=1001, le=1032)
    event_timestamp: datetime


def _feature_values(response: dict) -> dict:
    values = {}
    for name in FEATURE_COLUMNS:
        raw = response.get(name, [None])
        values[name] = raw[0] if raw else None
    return values


def _historical_row(store: FeatureStore, user_id: int, event_timestamp: datetime) -> dict:
    entity_df = pd.DataFrame(
        {
            "user_id": [user_id],
            "event_timestamp": [pd.Timestamp(event_timestamp)],
        }
    )
    result = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()
    if result.empty:
        return {}
    return {name: result.iloc[0][name] for name in FEATURE_COLUMNS}


@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "retail-risk-feature-store"}


@app.get("/api/metadata")
def metadata():
    data = dict(app.state.metadata)
    latest = pd.Timestamp(data["latest_event"])
    data["recommended_cutoff"] = (latest - pd.Timedelta(days=14) + pd.Timedelta(hours=12)).isoformat()
    data["feature_count"] = len(FEATURE_COLUMNS)
    data["features"] = FEATURE_COLUMNS
    return data


@app.get("/api/users")
def users():
    events = pd.read_parquet(DATA_PATH)
    return {"users": sorted(int(value) for value in events["user_id"].unique())}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    started = perf_counter()
    store = app.state.store
    online = store.get_online_features(
        features=risk_model_v1,
        entity_rows=[{"user_id": request.user_id}],
    ).to_dict()
    features = _feature_values(online)
    if any(value is None for value in features.values()):
        raise HTTPException(status_code=404, detail="No materialized feature row for this user")
    probability = float(
        app.state.model.predict_proba(
            pd.DataFrame([features], columns=FEATURE_COLUMNS)
        )[0][1]
    )
    events = pd.read_parquet(DATA_PATH)
    latest_event = events.loc[events["user_id"] == request.user_id, "event_timestamp"].max()
    return {
        "user_id": request.user_id,
        "prediction": "review" if probability >= 0.5 else "clear",
        "risk_probability": round(probability, 4),
        "risk_percent": round(probability * 100, 1),
        "features": features,
        "feature_service": "risk_model_v1",
        "online_as_of": latest_event.isoformat(),
        "latency_ms": round((perf_counter() - started) * 1000, 2),
    }


@app.post("/api/point-in-time")
def point_in_time(request: PointInTimeRequest):
    cutoff = request.event_timestamp
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    history = _historical_row(app.state.store, request.user_id, cutoff)
    if not history or all(pd.isna(value) for value in history.values()):
        raise HTTPException(status_code=404, detail="No historical features exist at this event time")

    events = pd.read_parquet(DATA_PATH)
    later = events.loc[
        (events["user_id"] == request.user_id)
        & (events["event_timestamp"] > pd.Timestamp(cutoff))
    ].sort_values("event_timestamp")
    future_row = later.iloc[0] if not later.empty else None
    lookahead = None
    if future_row is not None:
        lookahead = {
            "event_timestamp": future_row["event_timestamp"].isoformat(),
            "features": {name: future_row[name].item() for name in FEATURE_COLUMNS},
        }
    return {
        "user_id": request.user_id,
        "event_timestamp": cutoff.isoformat(),
        "historical_features": {
            name: value.item() if hasattr(value, "item") else value
            for name, value in history.items()
        },
        "next_snapshot_after_cutoff": lookahead,
        "excluded_future_snapshot": lookahead is not None,
    }


@app.get("/api/timeline/{user_id}")
def timeline(user_id: int):
    if user_id < 1001 or user_id > 1032:
        raise HTTPException(status_code=404, detail="Unknown user")
    events = pd.read_parquet(DATA_PATH)
    user_events = events.loc[events["user_id"] == user_id].sort_values("event_timestamp")
    sampled = user_events.iloc[::8]
    entity_df = sampled[["user_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = entity_df["event_timestamp"] + pd.Timedelta(hours=6)
    history = app.state.store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()
    return {
        "points": [
            {
                "event_timestamp": row["event_timestamp"].isoformat(),
                "transactions_7d": int(row["transactions_7d"]),
                "failed_payments_30d": int(row["failed_payments_30d"]),
            }
            for _, row in history.iterrows()
        ]
    }