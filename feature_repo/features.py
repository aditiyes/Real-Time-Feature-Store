from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64


user = Entity(name="user_id", join_keys=["user_id"], value_type=ValueType.INT64)

activity_source = FileSource(
    name="customer_activity_source",
    path=str(Path(__file__).resolve().parent / "data" / "customer_activity.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

customer_activity = FeatureView(
    name="customer_activity",
    entities=[user],
    ttl=timedelta(days=120),
    schema=[
        Field(name="transactions_7d", dtype=Int64),
        Field(name="failed_payments_30d", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="account_age_days", dtype=Int64),
    ],
    source=activity_source,
    online=True,
    tags={"owner": "risk-platform", "domain": "payments"},
)

risk_model_v1 = FeatureService(
    name="risk_model_v1",
    features=[customer_activity],
    tags={"model": "retail-risk", "version": "1"},
)

FEATURE_COLUMNS = [
    "transactions_7d",
    "failed_payments_30d",
    "avg_order_value",
    "account_age_days",
]
FEATURE_REFS = [f"customer_activity:{name}" for name in FEATURE_COLUMNS]