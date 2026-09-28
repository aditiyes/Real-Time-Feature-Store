import pandas as pd
from fastapi.testclient import TestClient
from feast import FeatureStore

from app.main import app
from feature_repo.bootstrap import DATA_PATH, REPO, RUNTIME_REPO, bootstrap_project
from feature_repo.features import FEATURE_COLUMNS, FEATURE_REFS


def test_historical_features_exclude_future_rows():
    config_path = REPO / "feature_store.yaml"
    original_config = config_path.read_bytes()
    bootstrap_project()
    assert config_path.read_bytes() == original_config
    events = pd.read_parquet(DATA_PATH)
    user_events = events.loc[events["user_id"] == 1001].sort_values("event_timestamp")
    expected = user_events.iloc[35]
    future = user_events.iloc[36]
    cutoff = expected["event_timestamp"] + pd.Timedelta(hours=6)
    store = FeatureStore(repo_path=str(RUNTIME_REPO))
    result = store.get_historical_features(
        entity_df=pd.DataFrame(
            {"user_id": [1001], "event_timestamp": [cutoff]}
        ),
        features=FEATURE_REFS,
    ).to_df()
    row = result.iloc[0]
    assert row["transactions_7d"] == expected["transactions_7d"]
    assert any(row[name] != future[name] for name in FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS).issubset(result.columns)


def test_prediction_reads_the_materialized_online_features():
    with TestClient(app) as client:
        page = client.get("/")
        response = client.post("/api/predict", json={"user_id": 1001})
    assert page.status_code == 200
    assert "Travel back in time" in page.text
    assert response.status_code == 200
    body = response.json()
    assert body["feature_service"] == "risk_model_v1"
    assert 0 <= body["risk_probability"] <= 1
    assert set(FEATURE_COLUMNS) == set(body["features"])


def test_point_in_time_endpoint_labels_a_future_snapshot_as_excluded():
    with TestClient(app) as client:
        metadata = client.get("/api/metadata").json()
        cutoff = pd.Timestamp(metadata["recommended_cutoff"]).isoformat()
        response = client.post(
            "/api/point-in-time",
            json={"user_id": 1001, "event_timestamp": cutoff},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["excluded_future_snapshot"] is True
    assert body["next_snapshot_after_cutoff"]["event_timestamp"] > cutoff
    assert set(FEATURE_COLUMNS) == set(body["historical_features"])