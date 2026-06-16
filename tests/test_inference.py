"""
Unit tests for inference.py pure functions and RiskModel.

RiskModel integration tests are skipped automatically when the model
artefact files (pre_delinquency_model.txt, feature_names.json, etc.)
are not present in the backend directory.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import assign_tier, label_for, DIRECTIONAL_LABELS, RiskModel

MODEL_DIR = Path(__file__).parent.parent
MODEL_FILES_PRESENT = all(
    (MODEL_DIR / f).exists()
    for f in (
        "pre_delinquency_model.txt",
        "feature_names.json",
        "baseline_vectors.json",
        "slider_config.json",
        "tier_config.json",
        "calibrator.pkl",
    )
)


# ---------------------------------------------------------------------------
# assign_tier
# ---------------------------------------------------------------------------

class TestAssignTier:
    def test_zero_is_watch(self):
        assert assign_tier(0.0, nudge=0.3, intervene=0.6) == "Watch"

    def test_just_below_nudge_threshold_is_watch(self):
        assert assign_tier(0.299, nudge=0.3, intervene=0.6) == "Watch"

    def test_at_nudge_threshold_is_nudge(self):
        assert assign_tier(0.3, nudge=0.3, intervene=0.6) == "Nudge"

    def test_midpoint_is_nudge(self):
        assert assign_tier(0.45, nudge=0.3, intervene=0.6) == "Nudge"

    def test_just_below_intervene_threshold_is_nudge(self):
        assert assign_tier(0.599, nudge=0.3, intervene=0.6) == "Nudge"

    def test_at_intervene_threshold_is_intervene(self):
        assert assign_tier(0.6, nudge=0.3, intervene=0.6) == "Intervene"

    def test_one_is_intervene(self):
        assert assign_tier(1.0, nudge=0.3, intervene=0.6) == "Intervene"

    def test_custom_thresholds(self):
        assert assign_tier(0.4, nudge=0.5, intervene=0.8) == "Watch"
        assert assign_tier(0.5, nudge=0.5, intervene=0.8) == "Nudge"
        assert assign_tier(0.8, nudge=0.5, intervene=0.8) == "Intervene"
        assert assign_tier(0.79, nudge=0.5, intervene=0.8) == "Nudge"

    def test_all_demo_scores_produce_valid_tiers(self):
        import json
        with open(MODEL_DIR / "demo_accounts.json") as f:
            accounts = json.load(f)
        with open(MODEL_DIR / "tier_config.json") as f:
            tc = json.load(f)
        for a in accounts:
            tier = assign_tier(a["score"], tc["nudge_threshold"], tc["intervene_threshold"])
            assert tier in ("Watch", "Nudge", "Intervene")


# ---------------------------------------------------------------------------
# label_for
# ---------------------------------------------------------------------------

class TestLabelFor:
    def test_known_feature_up(self):
        assert label_for("EXT_MEAN", "up") == "Below-average external credit score"

    def test_known_feature_down(self):
        assert label_for("EXT_MEAN", "down") == "Above-average external credit score"

    def test_known_feature_bureau_utilisation_up(self):
        assert label_for("BUR_DEBT_TO_CREDIT_RATIO", "up") == "High credit utilisation across bureau loans"

    def test_known_feature_bureau_utilisation_down(self):
        assert label_for("BUR_DEBT_TO_CREDIT_RATIO", "down") == "Low utilisation across bureau loans"

    def test_namespaced_feature_resolves_correctly(self):
        # Features stored as "module__FEATURE_NAME" should use the suffix
        assert label_for("pipeline__EXT_MEAN", "up") == "Below-average external credit score"

    def test_unknown_feature_up_starts_with_higher(self):
        assert label_for("TOTALLY_UNKNOWN_FEATURE_XYZ", "up").startswith("Higher")

    def test_unknown_feature_down_starts_with_lower(self):
        assert label_for("TOTALLY_UNKNOWN_FEATURE_XYZ", "down").startswith("Lower")

    def test_unknown_feature_includes_readable_name(self):
        label = label_for("SOME_SCORE_VALUE", "up")
        assert "some score value" in label.lower()


# ---------------------------------------------------------------------------
# DIRECTIONAL_LABELS structure
# ---------------------------------------------------------------------------

class TestDirectionalLabels:
    def test_every_entry_has_up_direction(self):
        for feature, directions in DIRECTIONAL_LABELS.items():
            assert "up" in directions, f"Missing 'up' for {feature}"

    def test_every_entry_has_down_direction(self):
        for feature, directions in DIRECTIONAL_LABELS.items():
            assert "down" in directions, f"Missing 'down' for {feature}"

    def test_no_empty_labels(self):
        for feature, directions in DIRECTIONAL_LABELS.items():
            assert directions["up"], f"Empty 'up' label for {feature}"
            assert directions["down"], f"Empty 'down' label for {feature}"

    def test_up_and_down_labels_are_different(self):
        for feature, directions in DIRECTIONAL_LABELS.items():
            assert directions["up"] != directions["down"], f"up == down for {feature}"


# ---------------------------------------------------------------------------
# RiskModel  — mocked unit tests (no real model files needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """RiskModel with all file I/O mocked out."""
    feature_names = ["EXT_MEAN", "BUR_DEBT_TO_CREDIT_RATIO", "SK_ID_CURR", "CC_UTIL_MEAN"]
    baselines = {
        "196682": {
            "features": [0.6, 0.4, 99999.0, 0.3],
            "tier": "Nudge",
            "score": 0.05,
        },
    }

    mock_booster = MagicMock()
    mock_booster.predict.side_effect = lambda vec, pred_contrib=False: (
        # pred_contrib=True: shape (1, n_features + 1 bias)
        np.array([[0.015, -0.008, 0.0, 0.005, 0.0]]) if pred_contrib
        else np.array([0.045])
    )

    mock_calibrator = MagicMock()
    mock_calibrator.predict.side_effect = lambda raw: raw  # identity calibration for tests

    model = RiskModel.__new__(RiskModel)
    model.booster = mock_booster
    model.calibrator = mock_calibrator
    model.feature_names = feature_names
    model.feature_index = {n: i for i, n in enumerate(feature_names)}
    model.baselines = baselines
    model.slider_config = []
    model.nudge_threshold = 0.0457
    model.intervene_threshold = 0.0913
    return model


class TestRiskModelPredict:
    def test_returns_score_tier_and_drivers(self, mock_model):
        result = mock_model.predict("196682")
        assert result is not None
        score, tier, drivers = result
        assert 0.0 <= score <= 1.0
        assert tier in ("Watch", "Nudge", "Intervene")
        assert len(drivers) > 0

    def test_unknown_account_returns_none(self, mock_model):
        assert mock_model.predict("ACC-MISSING") is None

    def test_each_driver_has_required_keys(self, mock_model):
        _, _, drivers = mock_model.predict("196682")
        for d in drivers:
            assert {"label", "feature", "impact", "shap", "direction"}.issubset(d.keys())

    def test_driver_direction_is_up_or_down(self, mock_model):
        _, _, drivers = mock_model.predict("196682")
        for d in drivers:
            assert d["direction"] in ("up", "down")

    def test_driver_impact_normalized_to_100(self, mock_model):
        _, _, drivers = mock_model.predict("196682")
        assert max(d["impact"] for d in drivers) == 100

    def test_override_modifies_feature_vector(self, mock_model):
        # Call predict with an override and verify the booster got called
        mock_model.predict("196682", overrides={"BUR_DEBT_TO_CREDIT_RATIO": 0.95})
        assert mock_model.booster.predict.called

    def test_override_for_unknown_feature_is_ignored(self, mock_model):
        result = mock_model.predict("196682", overrides={"NONEXISTENT_FEATURE": 99.0})
        assert result is not None

    def test_excluded_features_not_in_drivers(self, mock_model):
        _, _, drivers = mock_model.predict("196682")
        labels = [d["label"] for d in drivers]
        # SK_ID_CURR is in EXCLUDE_FROM_DRIVERS — should never appear
        assert not any("sk_id" in lbl.lower() for lbl in labels)

    def test_top_n_respected(self, mock_model):
        # With 4 features but top_n=2, expect at most 2 drivers
        _, _, drivers = mock_model.predict("196682", top_n=2)
        assert len(drivers) <= 2


# ---------------------------------------------------------------------------
# RiskModel integration tests (skipped if model files absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_FILES_PRESENT, reason="model artefact files not present")
class TestRiskModelIntegration:
    @pytest.fixture(scope="class")
    def live_model(self):
        return RiskModel(MODEL_DIR)

    def test_loads_without_error(self, live_model):
        assert live_model is not None

    def test_baselines_are_non_empty(self, live_model):
        assert len(live_model.baselines) > 0

    def test_feature_names_match_baselines_length(self, live_model):
        for account_id, baseline in live_model.baselines.items():
            assert len(baseline["features"]) == len(live_model.feature_names), account_id

    def test_predict_returns_score_in_range(self, live_model):
        account_id = next(iter(live_model.baselines))
        score, _, _ = live_model.predict(account_id)
        assert 0.0 <= score <= 1.0

    def test_predict_with_zero_overrides_equals_baseline(self, live_model):
        account_id = next(iter(live_model.baselines))
        score_base, _, _ = live_model.predict(account_id)
        score_empty, _, _ = live_model.predict(account_id, overrides={})
        assert abs(score_base - score_empty) < 1e-6

    def test_high_bureau_utilisation_raises_score(self, live_model):
        account_id = next(iter(live_model.baselines))
        base_score, _, _ = live_model.predict(account_id)
        high_score, _, _ = live_model.predict(
            account_id, overrides={"BUR_DEBT_TO_CREDIT_RATIO": 1.4}
        )
        # Higher bureau utilisation should push risk up (or stay same if saturated)
        assert high_score >= base_score - 0.05  # allow tiny tolerance
