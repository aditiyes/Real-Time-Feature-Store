# Fieldnotes: Real-Time Feature Store

A portfolio-sized MLOps project built with [Feast](https://github.com/feast-dev/feast). It defines one reusable customer feature contract, retrieves point-in-time training rows from offline Parquet, materializes the same features to an online store, and serves predictions through FastAPI.

## What it demonstrates

- `customer_activity` is a Feast `FeatureView` with an entity key, event timestamp, TTL, typed schema, and offline file source.
- `risk_model_v1` is shared by the training retrieval and online prediction path.
- Bootstrap creates deterministic synthetic retail activity, runs Feast registry apply and online materialization, then trains a logistic-regression baseline from Feast historical retrieval.
- `/api/predict` reads the materialized online features before scoring. The dashboard displays the returned feature vector and prediction latency.
- `/api/point-in-time` compares the feature vector available at an entity timestamp with the next later snapshot. Feast returns the earlier values; the later values are explicitly labeled as excluded to make leakage visible.
- Local runs and Render's free single-service Blueprint use Feast's SQLite online store. Docker Compose can opt into Redis.

All example records are synthetic. The model and feature data are generated when the service starts and are not checked into the repository.

## Run locally on Windows

Python 3.9 or newer is supported by the pinned dependencies.

```powershell
cd $HOME\Downloads\MLOps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. Startup generates the offline Parquet source, registers definitions, materializes online features, and trains the model. The API docs are at `http://127.0.0.1:8000/docs`.

## Run with Redis

Install Docker Desktop, then from the project directory:

```powershell
docker compose up --build
```

The app waits for Redis health, sets Feast's online store to Redis, and exposes the dashboard at `http://localhost:8000`.

## Verify point-in-time correctness

```powershell
python -m pytest -q
```

The PIT test submits an entity timestamp between feature events and asserts that Feast returns the prior feature vector rather than the next event's values. The API-level test also confirms that the online model service fetches its registered feature contract.

Useful endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Container health check |
| `GET` | `/api/metadata` | Store, dataset, feature, and training metadata |
| `GET` | `/api/users` | Synthetic entity IDs for the demo |
| `POST` | `/api/predict` | Read online features and return a risk score |
| `POST` | `/api/point-in-time` | Retrieve historical features at an event timestamp |
| `GET` | `/api/timeline/{user_id}` | Feast point-in-time snapshots for the chart |

Example prediction request:

```json
{"user_id": 1001}
```

## Deploy on Render

The included `render.yaml` deploys one Docker web service on Render's free plan using Feast's SQLite online store. No Redis service is required for this profile. The container bootstraps the Feast repo and model on startup, and `/api/health` is the health check. Free web services can sleep when idle and their local filesystem is ephemeral; the app recreates demo data and rematerializes features on startup. The Docker Compose option above uses Redis when you want the separate online-store process.

For another host, run the Docker image with a reachable Redis instance and set `REDIS_URL` to its connection string. Keep the online store private to the application network.

## Project map

```text
MLOps/
  app/main.py                  FastAPI prediction and PIT endpoints
  app/static/                  Responsive feature operations dashboard
  feature_repo/features.py     Feast entity, FeatureView, FeatureService
  feature_repo/bootstrap.py    Synthetic source, materialization, training
  feature_repo/feature_store.yaml
  tests/test_feature_store.py  PIT and online-serving checks
  compose.yaml                 App plus Redis
  render.yaml                  Render Blueprint
```

## Model note

The target and customer activity are synthetic, and the classifier is a transparent baseline for demonstrating feature plumbing, not a payment-risk model for production use. The startup ROC AUC is training-set telemetry, not a holdout evaluation.