"""
Live inference for the pre-delinquency risk model.

Loads the trained XGBoost booster plus per-account baseline feature
vectors, applies "what-if" overrides for a small set of slider features,
and returns a recomputed score, tier, and SHAP-based drivers using
XGBoost's built-in pred_contribs (no separate shap install needed).
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


# ---------------------------------------------------------------------------
# Tiering (kept consistent with the export script / dashboard)
# ---------------------------------------------------------------------------

def assign_tier(score: float, low: float = 0.3, high: float = 0.6) -> str:
    if score < low:
        return "Watch"
    elif score < high:
        return "Nudge"
    return "Intervene"


# ---------------------------------------------------------------------------
# Plain-English, direction-aware driver labels
# (mirrors the export script used for demo_accounts.json)
# ---------------------------------------------------------------------------

EXCLUDE_FROM_DRIVERS = {
    "SK_ID_CURR",
    "CODE_GENDER_M", "CODE_GENDER_F", "CODE_GENDER_XNA",
    "NAME_FAMILY_STATUS_Married", "NAME_FAMILY_STATUS_Single / not married",
    "NAME_FAMILY_STATUS_Civil marriage", "NAME_FAMILY_STATUS_Widow",
    "NAME_FAMILY_STATUS_Separated", "NAME_FAMILY_STATUS_Unknown",
}

DIRECTIONAL_LABELS = {
    "EXT_SOURCE_MEAN": {"up": "Below-average external credit score", "down": "Above-average external credit score"},
    "EXT_SOURCE_3": {"up": "Weak score from credit bureau (source 3)", "down": "Strong score from credit bureau (source 3)"},
    "EXT_SOURCE_2": {"up": "Weak score from credit bureau (source 2)", "down": "Strong score from credit bureau (source 2)"},
    "EXT_SOURCE_1": {"up": "Weak score from credit bureau (source 1)", "down": "Strong score from credit bureau (source 1)"},
    "EXT_SOURCE_MIN": {"up": "Weakest of the external credit scores is low", "down": "Weakest of the external credit scores is solid"},
    "EXT_SOURCE_STD": {"up": "External credit scores disagree with each other", "down": "External credit scores are consistent"},
    "EXT_SOURCE_PROD": {"up": "Combined external credit signals are weak", "down": "Combined external credit signals are strong"},
    "EXT_SOURCE_AGE": {"up": "External credit score low relative to age", "down": "External credit score solid relative to age"},
    "EXT_SOURCE_EMPLOYED": {"up": "Credit score low relative to employment history", "down": "Credit score solid relative to employment history"},
    "EMPLOYED_TO_AGE_RATIO": {"up": "Shorter employment history relative to age", "down": "Longer employment history relative to age"},
    "CREDIT_GOODS_RATIO": {"up": "Borrowing more than the item's value", "down": "Loan amount in line with item's value"},
    "CREDIT_TO_AGE_RATIO": {"up": "Loan size is large relative to age", "down": "Loan size is modest relative to age"},
    "DEBT_BURDEN": {"up": "High overall debt burden", "down": "Manageable overall debt burden"},
    "PAYMENT_BURDEN": {"up": "High repayment burden relative to income", "down": "Repayment burden is manageable relative to income"},
    "INST_MAX_DAYS_LATE": {"up": "History of significant payment delays", "down": "No history of major payment delays"},
    "INST_MEAN_DAYS_LATE": {"up": "Payments tend to arrive late", "down": "Payments tend to arrive on time"},
    "INST_PAYMENT_RATIO": {"up": "Recent payments below amount due", "down": "Recent payments have matched amounts due"},
    "MAX_OVERDUE_DAYS": {"up": "Past loans went significantly overdue", "down": "Past loans were not significantly overdue"},
    "WORST_STATUS": {"up": "Worst bureau status on record is poor", "down": "Bureau status history is clean"},
    "PREV_COUNT": {"up": "Many previous loan applications", "down": "Few previous loan applications"},
    "PREV_APPROVED_COUNT": {"up": "Many previously approved loans", "down": "Few previously approved loans"},
    "PREV_APPROVAL_RATE": {"up": "History of loan application rejections", "down": "Strong history of loan approvals"},
    "POS_MONTHS_COUNT": {"up": "Limited point-of-sale loan history", "down": "Established point-of-sale loan history"},
    "POS_MAX_DPD": {"up": "Point-of-sale loans went significantly overdue", "down": "Point-of-sale loans stayed current"},
    "POS_BAD_MONTHS_COUNT": {"up": "Several months of poor point-of-sale repayment", "down": "Consistently good point-of-sale repayment"},
    "DEBT_TO_CREDIT_RATIO": {"up": "High credit utilization", "down": "Low to moderate credit utilization"},
    "CC_UTILITY_MEAN": {"up": "High average credit card utilization", "down": "Low average credit card utilization"},
    "CC_UTILITY_MAX": {"up": "Peak credit card utilization is high", "down": "Peak credit card utilization is low"},
    "AMT_GOODS_PRICE": {"up": "High-value item being financed", "down": "Lower-value item being financed"},
    "AMT_ANNUITY": {"up": "High monthly repayment amount", "down": "Manageable monthly repayment amount"},
    "AMT_CREDIT": {"up": "Large total loan amount", "down": "Smaller total loan amount"},
    "AMT_INCOME_TOTAL": {"up": "Lower declared income", "down": "Higher declared income"},
    "AGE_YEARS": {"up": "Younger applicant", "down": "Older, more established applicant"},
    "EMPLOYED_YEARS": {"up": "Shorter employment history", "down": "Longer employment history"},
    "DAYS_ID_PUBLISH": {"up": "Recent ID document change", "down": "ID document has not changed recently"},
    "DAYS_REGISTRATION": {"up": "Recent registration update", "down": "Registration has been stable"},
    "DAYS_LAST_PHONE_CHANGE": {"up": "Recent change of contact phone number", "down": "Phone number has not changed recently"},
    "AMT_REQ_CREDIT_BUREAU_QRT": {"up": "Recent increase in credit checks by other lenders", "down": "Few recent credit checks by other lenders"},
    "AMT_REQ_CREDIT_BUREAU_YEAR": {"up": "Many credit checks by other lenders this year", "down": "Few credit checks by other lenders this year"},
    "REGION_RATING_CLIENT": {"up": "Lives in a lower-rated region", "down": "Lives in a higher-rated region"},
    "REGION_RATING_CLIENT_W_CITY": {"up": "Lower-rated region (city-adjusted)", "down": "Higher-rated region (city-adjusted)"},
    "REGION_POPULATION_RELATIVE": {"up": "Lives in a less densely populated area", "down": "Lives in a more densely populated area"},
    "REG_CITY_NOT_WORK_CITY": {"up": "Registered address differs from work city", "down": "Registered address matches work city"},
    "REG_CITY_NOT_LIVE_CITY": {"up": "Registered address differs from current city", "down": "Registered address matches current city"},
    "FLAG_EMP_PHONE": {"up": "No employer phone on file", "down": "Employer phone on file"},
    "FLAG_DOCUMENT_3": {"up": "Missing a standard identity document", "down": "Standard identity documents on file"},
    "FLOORSMAX_AVG": {"up": "Lives in a lower-floor-count building", "down": "Lives in a higher-floor-count building"},
    "FLOORSMAX_MODE": {"up": "Lives in a lower-floor-count building (typical)", "down": "Lives in a higher-floor-count building (typical)"},
    "FLOORSMAX_MEDI": {"up": "Lives in a lower-floor-count building (median)", "down": "Lives in a higher-floor-count building (median)"},
    "OWN_CAR_AGE": {"up": "Owns an older car", "down": "Owns a newer car, or no car"},
    "FLAG_OWN_CAR_N": {"up": "Does not own a car", "down": "Owns a car"},
    "FLAG_OWN_REALTY_Y": {"up": "Owns real estate", "down": "Does not own real estate"},
    "CNT_CHILDREN": {"up": "Has more dependents", "down": "Has fewer dependents"},
    "CNT_FAM_MEMBERS": {"up": "Larger household size", "down": "Smaller household size"},
}


def label_for(feature: str, direction: str) -> str:
    base = feature.split("__")[-1]
    if base in DIRECTIONAL_LABELS:
        return DIRECTIONAL_LABELS[base][direction]
    prefix = "Higher" if direction == "up" else "Lower"
    return f"{prefix} {base.replace('_', ' ').lower()}"


# ---------------------------------------------------------------------------
# Risk model wrapper
# ---------------------------------------------------------------------------

class RiskModel:
    def __init__(self, data_dir: Path):
        self.booster = xgb.Booster()
        self.booster.load_model(str(data_dir / "pre_delinquency_model.json"))

        with open(data_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)

        with open(data_dir / "baseline_vectors.json") as f:
            self.baselines = json.load(f)

        with open(data_dir / "slider_config.json") as f:
            self.slider_config = json.load(f)

        self.feature_index = {name: i for i, name in enumerate(self.feature_names)}

    def account_ids(self):
        return list(self.baselines.keys())

    def predict(self, account_id: str, overrides: dict[str, float] | None = None, top_n: int = 10):
        """
        Returns (score, drivers) for the given account, with optional
        feature overrides applied on top of its baseline vector.
        """
        if account_id not in self.baselines:
            return None

        vector = np.array(self.baselines[account_id], dtype=float).copy()

        if overrides:
            for feature, value in overrides.items():
                idx = self.feature_index.get(feature)
                if idx is None:
                    continue
                vector[idx] = value

        dmatrix = xgb.DMatrix(vector.reshape(1, -1))

        score = float(self.booster.predict(dmatrix)[0])

        # pred_contribs returns one contribution per feature, plus a bias term
        contribs = self.booster.predict(dmatrix, pred_contribs=True)[0]
        contribs = contribs[:-1]  # drop bias

        valid_idx = [
            i for i in range(len(contribs))
            if self.feature_names[i] not in EXCLUDE_FROM_DRIVERS
        ]
        sorted_idx = sorted(valid_idx, key=lambda i: abs(contribs[i]), reverse=True)
        top_idx = sorted_idx[:top_n]

        max_impact = max(abs(contribs[i]) for i in top_idx) if top_idx else 1.0
        if max_impact == 0:
            max_impact = 1.0

        drivers = []
        for i in top_idx:
            direction = "up" if contribs[i] > 0 else "down"
            drivers.append({
                "label": label_for(self.feature_names[i], direction),
                "impact": round(float(abs(contribs[i]) / max_impact) * 100),
                "direction": direction,
            })

        return score, drivers
