import json

import pytest
from fastapi.testclient import TestClient

CUSTOMER = {
    "tenure_months": 3,
    "monthly_charges": 110.5,
    "total_charges": 331.5,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "cash",
    "support_calls": 4,
    "senior_citizen": 0,
}


@pytest.fixture(scope="module")
def client(trained_artifacts, monkeypatch_module):
    monkeypatch_module.setenv("MODEL_DIR", str(trained_artifacts))
    from serving.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_quality_gate_metrics(trained_artifacts):
    metrics = json.loads((trained_artifacts / "metrics.json").read_text())
    assert metrics["roc_auc"] >= 0.75


def test_ready(client):
    assert client.get("/readyz").json()["status"] == "ready"


def test_high_risk_customer(client):
    body = client.post("/predict", json=CUSTOMER).json()
    assert body["will_churn"] is True
    assert 0.5 <= body["churn_probability"] <= 1


def test_loyal_customer(client):
    loyal = {**CUSTOMER, "tenure_months": 60, "contract": "two-year", "support_calls": 0,
             "monthly_charges": 45.0, "total_charges": 2700.0, "payment_method": "upi", "internet_service": "dsl"}
    assert client.post("/predict", json=loyal).json()["will_churn"] is False


def test_rejects_bad_input(client):
    assert client.post("/predict", json={**CUSTOMER, "contract": "weekly"}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=CUSTOMER)
    assert "churn_predictions_total" in client.get("/metrics/").text
