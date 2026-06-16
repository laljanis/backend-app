"""
API endpoint tests.

Run:
    pip install -r requirements-dev.txt
    pytest tests/ -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure the backend root is on the path so `import main` works from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

import main as app_module
from main import app

client = TestClient(app)

DATA_PATH = Path(__file__).parent.parent / "demo_accounts.json"
with open(DATA_PATH) as f:
    DEMO_ACCOUNTS = json.load(f)

VALID_ID = DEMO_ACCOUNTS[0]["id"]
MISSING_ID = "ACC-DOESNOTEXIST"


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRoot:
    def test_returns_200(self):
        assert client.get("/").status_code == 200

    def test_service_name(self):
        assert client.get("/").json()["service"] == "pre-delinquency-risk-scoring"

    def test_accounts_loaded_matches_file(self):
        assert client.get("/").json()["accounts_loaded"] == len(DEMO_ACCOUNTS)

    def test_live_model_flag_present(self):
        assert "live_model_loaded" in client.get("/").json()


# ---------------------------------------------------------------------------
# GET /api/portfolio/summary
# ---------------------------------------------------------------------------

class TestPortfolioSummary:
    def test_returns_200(self):
        assert client.get("/api/portfolio/summary").status_code == 200

    def test_total_equals_account_count(self):
        data = client.get("/api/portfolio/summary").json()
        assert data["total_accounts"] == len(DEMO_ACCOUNTS)

    def test_tier_counts_sum_to_total(self):
        data = client.get("/api/portfolio/summary").json()
        assert sum(t["count"] for t in data["tiers"].values()) == data["total_accounts"]

    def test_all_three_tiers_present(self):
        tiers = client.get("/api/portfolio/summary").json()["tiers"]
        assert set(tiers.keys()) == {"Watch", "Nudge", "Intervene"}

    def test_percentages_are_reasonable(self):
        tiers = client.get("/api/portfolio/summary").json()["tiers"]
        total_pct = sum(t["pct"] for t in tiers.values())
        # Allow ±2 for rounding across three tiers
        assert 98 <= total_pct <= 102

    def test_no_tier_exceeds_100_pct(self):
        tiers = client.get("/api/portfolio/summary").json()["tiers"]
        for tier in tiers.values():
            assert tier["pct"] <= 100

    def test_thresholds_present_and_ordered(self):
        thresholds = client.get("/api/portfolio/summary").json()["thresholds"]
        assert 0.0 < thresholds["nudge"] < thresholds["intervene"] < 1.0


# ---------------------------------------------------------------------------
# GET /api/accounts
# ---------------------------------------------------------------------------

class TestListAccounts:
    def test_returns_200(self):
        assert client.get("/api/accounts").status_code == 200

    def test_returns_all_accounts(self):
        assert len(client.get("/api/accounts").json()) == len(DEMO_ACCOUNTS)

    def test_sorted_by_score_descending(self):
        scores = [a["score"] for a in client.get("/api/accounts").json()]
        assert scores == sorted(scores, reverse=True)

    def test_each_account_has_required_fields(self):
        required = {"id", "score", "tier", "trend", "top_driver"}
        for account in client.get("/api/accounts").json():
            assert required.issubset(account.keys())

    def test_trend_is_list_of_floats(self):
        for account in client.get("/api/accounts").json():
            assert isinstance(account["trend"], list)
            assert all(isinstance(v, (int, float)) for v in account["trend"])

    def test_filter_watch(self):
        accounts = client.get("/api/accounts?tier=Watch").json()
        assert client.get("/api/accounts?tier=Watch").status_code == 200
        assert all(a["tier"] == "Watch" for a in accounts)

    def test_filter_nudge(self):
        accounts = client.get("/api/accounts?tier=Nudge").json()
        assert all(a["tier"] == "Nudge" for a in accounts)

    def test_filter_intervene(self):
        accounts = client.get("/api/accounts?tier=Intervene").json()
        assert all(a["tier"] == "Intervene" for a in accounts)

    def test_invalid_tier_returns_400(self):
        assert client.get("/api/accounts?tier=UNKNOWN").status_code == 400

    def test_filter_results_still_sorted(self):
        scores = [a["score"] for a in client.get("/api/accounts?tier=Nudge").json()]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# GET /api/accounts/{account_id}
# ---------------------------------------------------------------------------

class TestGetAccount:
    def test_valid_id_returns_200(self):
        assert client.get(f"/api/accounts/{VALID_ID}").status_code == 200

    def test_missing_id_returns_404(self):
        assert client.get(f"/api/accounts/{MISSING_ID}").status_code == 404

    def test_response_fields(self):
        data = client.get(f"/api/accounts/{VALID_ID}").json()
        assert {"id", "score", "tier", "trend", "drivers", "action"}.issubset(data.keys())

    def test_drivers_are_non_empty(self):
        drivers = client.get(f"/api/accounts/{VALID_ID}").json()["drivers"]
        assert len(drivers) > 0

    def test_driver_directions_are_valid(self):
        for d in client.get(f"/api/accounts/{VALID_ID}").json()["drivers"]:
            assert d["direction"] in ("up", "down")

    def test_driver_impact_is_non_negative_int(self):
        for d in client.get(f"/api/accounts/{VALID_ID}").json()["drivers"]:
            assert isinstance(d["impact"], int)
            assert d["impact"] >= 0

    def test_action_has_title_and_detail(self):
        action = client.get(f"/api/accounts/{VALID_ID}").json()["action"]
        assert action["title"]
        assert action["detail"]

    def test_id_in_response_matches_request(self):
        assert client.get(f"/api/accounts/{VALID_ID}").json()["id"] == VALID_ID

    def test_every_account_in_demo_data_is_reachable(self):
        for account in DEMO_ACCOUNTS:
            r = client.get(f"/api/accounts/{account['id']}")
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/sliders/{account_id}  — model absent (503)
# ---------------------------------------------------------------------------

class TestSlidersNoModel:
    def test_returns_503_when_model_not_loaded(self, monkeypatch):
        monkeypatch.setattr(app_module, "risk_model", None)
        assert client.get(f"/api/sliders/{VALID_ID}").status_code == 503


# ---------------------------------------------------------------------------
# GET /api/sliders/{account_id}  — model present
# ---------------------------------------------------------------------------

RAW_VALID_ID = VALID_ID.removeprefix("ACC-")

SLIDER_FEATURES = ["EXT_MEAN", "INST_MEAN_LATE", "CC_UTIL_MEAN", "BUR_DEBT_TO_CREDIT_RATIO"]


@pytest.fixture
def mock_risk_model():
    model = MagicMock()
    model.baselines = {
        RAW_VALID_ID: {"features": [0.5, 0.3, 15.0, 0.4], "tier": "Nudge", "score": 0.06},
    }
    model.feature_index = {name: i for i, name in enumerate(SLIDER_FEATURES)}
    model.slider_config = [
        {"feature": "EXT_MEAN", "label": "External credit score (mean)", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
        {"feature": "INST_MEAN_LATE", "label": "Avg days late on installments", "min": 0.0, "max": 30.0, "step": 0.5, "default": 0.0},
        {"feature": "CC_UTIL_MEAN", "label": "Credit card utilisation (mean)", "min": 0.0, "max": 1.5, "step": 0.05, "default": 0.25},
        {"feature": "BUR_DEBT_TO_CREDIT_RATIO", "label": "Bureau debt-to-credit ratio", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.4},
    ]
    return model


class TestSlidersWithModel:
    def test_returns_200_for_valid_account(self, monkeypatch, mock_risk_model):
        monkeypatch.setattr(app_module, "risk_model", mock_risk_model)
        assert client.get(f"/api/sliders/{VALID_ID}").status_code == 200

    def test_returns_sliders_for_every_configured_feature(self, monkeypatch, mock_risk_model):
        monkeypatch.setattr(app_module, "risk_model", mock_risk_model)
        sliders = client.get(f"/api/sliders/{VALID_ID}").json()
        assert len(sliders) == len(mock_risk_model.slider_config)

    def test_each_slider_has_required_fields(self, monkeypatch, mock_risk_model):
        monkeypatch.setattr(app_module, "risk_model", mock_risk_model)
        required = {"feature", "label", "min", "max", "step", "current"}
        for s in client.get(f"/api/sliders/{VALID_ID}").json():
            assert required.issubset(s.keys())

    def test_slider_min_less_than_max(self, monkeypatch, mock_risk_model):
        monkeypatch.setattr(app_module, "risk_model", mock_risk_model)
        for s in client.get(f"/api/sliders/{VALID_ID}").json():
            assert s["min"] < s["max"]

    def test_missing_account_returns_404(self, monkeypatch, mock_risk_model):
        monkeypatch.setattr(app_module, "risk_model", mock_risk_model)
        assert client.get(f"/api/sliders/{MISSING_ID}").status_code == 404


# ---------------------------------------------------------------------------
# POST /api/predict  — model absent (503)
# ---------------------------------------------------------------------------

class TestPredictNoModel:
    def test_returns_503_when_model_not_loaded(self, monkeypatch):
        monkeypatch.setattr(app_module, "risk_model", None)
        r = client.post("/api/predict", json={"account_id": VALID_ID, "overrides": {}})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/predict  — model present
# ---------------------------------------------------------------------------

@pytest.fixture
def predicting_model():
    model = MagicMock()
    model.predict.return_value = (
        0.061234,
        "Nudge",
        [{"label": "High credit utilization", "feature": "BUR_DEBT_TO_CREDIT_RATIO", "impact": 80, "shap": 0.02, "direction": "up"}],
    )
    return model


class TestPredictWithModel:
    def test_valid_request_returns_200(self, monkeypatch, predicting_model):
        monkeypatch.setattr(app_module, "risk_model", predicting_model)
        r = client.post("/api/predict", json={"account_id": VALID_ID, "overrides": {}})
        assert r.status_code == 200

    def test_response_has_required_fields(self, monkeypatch, predicting_model):
        monkeypatch.setattr(app_module, "risk_model", predicting_model)
        data = client.post("/api/predict", json={"account_id": VALID_ID, "overrides": {}}).json()
        assert {"id", "score", "tier", "drivers", "action"}.issubset(data.keys())

    def test_tier_is_valid(self, monkeypatch, predicting_model):
        monkeypatch.setattr(app_module, "risk_model", predicting_model)
        tier = client.post("/api/predict", json={"account_id": VALID_ID, "overrides": {}}).json()["tier"]
        assert tier in ("Watch", "Nudge", "Intervene")

    def test_overrides_are_forwarded_to_model(self, monkeypatch, predicting_model):
        monkeypatch.setattr(app_module, "risk_model", predicting_model)
        overrides = {"BUR_DEBT_TO_CREDIT_RATIO": 0.9}
        client.post("/api/predict", json={"account_id": VALID_ID, "overrides": overrides})
        predicting_model.predict.assert_called_once_with(RAW_VALID_ID, overrides=overrides)

    def test_missing_account_returns_404(self, monkeypatch):
        model = MagicMock()
        model.predict.return_value = None
        monkeypatch.setattr(app_module, "risk_model", model)
        r = client.post("/api/predict", json={"account_id": MISSING_ID, "overrides": {}})
        assert r.status_code == 404

    def test_score_is_rounded_to_four_decimals(self, monkeypatch, predicting_model):
        monkeypatch.setattr(app_module, "risk_model", predicting_model)
        score = client.post("/api/predict", json={"account_id": VALID_ID, "overrides": {}}).json()["score"]
        assert round(score, 4) == score
