"""
Live inference for the pre-delinquency risk model (LightGBM).

Loads the trained LightGBM booster plus per-account baseline feature
vectors, applies "what-if" overrides for a small set of slider features,
and returns a recomputed calibrated score, tier, and SHAP-based drivers
using LightGBM's built-in pred_contrib (no separate shap install needed).

Artifacts expected in data_dir/
  pre_delinquency_model.txt  – LightGBM booster (text format)
  feature_names.json         – ordered list of feature names
  baseline_vectors.json      – {SK_ID_CURR: {features, tier, score}}
  slider_config.json         – interactive slider definitions
  tier_config.json           – nudge/intervene thresholds
  calibrator.pkl             – isotonic regression calibrator
"""

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def assign_tier(score: float, nudge: float, intervene: float) -> str:
    if score >= intervene:
        return "Intervene"
    elif score >= nudge:
        return "Nudge"
    return "Watch"


# ---------------------------------------------------------------------------
# Plain-English, direction-aware driver labels
# Updated to match new feature names from the 15-cell pipeline
# ---------------------------------------------------------------------------

EXCLUDE_FROM_DRIVERS = {
    "SK_ID_CURR",
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
}

DIRECTIONAL_LABELS = {
    # External credit scores
    "EXT_MEAN":               {"up": "Below-average external credit score",            "down": "Above-average external credit score"},
    "EXT_MIN":                {"up": "Weakest external credit score is low",           "down": "Weakest external credit score is solid"},
    "EXT_STD":                {"up": "External credit scores are inconsistent",        "down": "External credit scores are consistent"},
    "EXT_PROD":               {"up": "Combined external credit signals are weak",      "down": "Combined external credit signals are strong"},
    "EXT_X_AGE":              {"up": "Credit score low relative to age",               "down": "Credit score solid relative to age"},
    "EXT_X_EMPLOYED":         {"up": "Credit score low relative to employment history","down": "Credit score solid relative to employment history"},
    "EXT_X_CREDIT":           {"up": "Credit score low relative to debt load",         "down": "Credit score solid relative to debt load"},
    "EXT_X_DEBT_RATIO":       {"up": "Low credit score combined with high debt",       "down": "Strong credit score with manageable debt"},
    "EXT_X_CREDIT_TERM":      {"up": "Long loan term with weak credit score",          "down": "Short loan term with solid credit score"},

    # Financial ratios
    "CREDIT_INCOME_RATIO":    {"up": "Large loan relative to income",                  "down": "Loan is modest relative to income"},
    "ANNUITY_INCOME_RATIO":   {"up": "High monthly repayment burden",                  "down": "Manageable monthly repayment burden"},
    "CREDIT_GOODS_RATIO":     {"up": "Borrowing above the item's value",               "down": "Loan in line with item's value"},
    "CREDIT_TERM":            {"up": "Very long loan term",                             "down": "Short loan term"},
    "INCOME_PER_PERSON":      {"up": "Low income per household member",                "down": "Healthy income per household member"},
    "CHILDREN_RATIO":         {"up": "High proportion of dependants",                  "down": "Low proportion of dependants"},
    "EMPLOYED_TO_AGE":        {"up": "Shorter employment history relative to age",     "down": "Long employment history relative to age"},

    # Time-based
    "AGE_YEARS":              {"up": "Younger applicant",                              "down": "Older, more established applicant"},
    "EMPLOYED_YEARS":         {"up": "Short employment history",                       "down": "Long, stable employment history"},
    "ID_RECENCY":             {"up": "Recent ID document change",                      "down": "ID document has been stable"},
    "PHONE_RECENCY":          {"up": "Recent change of phone number",                  "down": "Phone number has been stable"},
    "REGIST_RECENCY":         {"up": "Recent registration update",                     "down": "Registration has been stable"},

    # Bureau features
    "BUR_LOAN_COUNT":         {"up": "Many bureau loans on record",                    "down": "Few bureau loans on record"},
    "BUR_ACTIVE_COUNT":       {"up": "Many active bureau loans",                       "down": "Few active bureau loans"},
    "BUR_MAX_OVERDUE_DAYS":   {"up": "Past loans went significantly overdue",          "down": "Past loans were not significantly overdue"},
    "BUR_MEAN_OVERDUE_DAYS":  {"up": "Payments to other lenders tend to be late",      "down": "Payments to other lenders tend to be on time"},
    "BUR_OVERDUE_RATE":       {"up": "High proportion of bureau loans overdue",        "down": "Low proportion of bureau loans overdue"},
    "BUR_DEBT_TO_CREDIT_RATIO":{"up": "High credit utilisation across bureau loans",   "down": "Low utilisation across bureau loans"},
    "BUR_PROLONG_COUNT":      {"up": "Frequently extends loan terms (stress signal)",  "down": "Rarely needs to extend loan terms"},
    "BUR_CLOSED_RATE":        {"up": "Most bureau loans still open",                   "down": "Most bureau loans successfully closed"},

    # Bureau balance / trajectory
    "BB_WORST_STATUS_ALL":    {"up": "Worst bureau delinquency status is severe",      "down": "Bureau delinquency history is clean"},
    "BB_MEAN_STATUS_ALL":     {"up": "Chronic delinquency in bureau history",          "down": "Minimal delinquency in bureau history"},
    "BB_DPD_RATE_ALL":        {"up": "Frequently delinquent in bureau history",        "down": "Rarely delinquent in bureau history"},
    "BB_WORST_STATUS_LAST6M": {"up": "Recent bureau delinquency is severe",            "down": "Recent bureau status is clean"},
    "BB_STATUS_TREND":        {"up": "Bureau delinquency worsening recently",          "down": "Bureau status stable or improving"},
    "BB_MONTHS_SINCE_LAST_DPD":{"up": "Recent delinquency event in bureau",            "down": "No recent delinquency in bureau"},

    # Previous applications
    "PREV_REFUSAL_RATE":      {"up": "High rate of previous loan rejections",          "down": "Strong history of loan approvals"},
    "PREV_APPROVAL_RATE":     {"up": "Low prior loan approval rate",                   "down": "High prior loan approval rate"},
    "PREV_CANCEL_RATE":       {"up": "Frequently cancels loan applications",           "down": "Rarely cancels loan applications"},
    "PREV_AMT_DIFF_MEAN":     {"up": "Often granted less than requested",              "down": "Typically granted close to amount requested"},
    "PREV_INTEREST_RATE":     {"up": "High interest rate on previous loans",           "down": "Low interest rate on previous loans"},
    "PREV_DAYS_MIN":          {"up": "Applied for another loan very recently",         "down": "No very recent prior application"},

    # POS CASH trajectory
    "POS_MAX_DPD":            {"up": "POS loan went significantly overdue",            "down": "POS loans stayed current"},
    "POS_BAD_MONTH_RATE":     {"up": "Frequently missed POS payments",                 "down": "Consistently met POS payments"},
    "POS_STRESS_TREND":       {"up": "POS delinquency worsening recently",             "down": "POS payment behaviour stable or improving"},
    "POS_MAX_BAD_STREAK":     {"up": "Extended streak of missed POS payments",         "down": "No extended streak of missed POS payments"},
    "POS_MONTHS_SINCE_LAST_DPD":{"up": "Recent POS delinquency event",                "down": "No recent POS delinquency"},
    "POS_BAD_RATE_LAST3M":    {"up": "Missed POS payments in last 3 months",          "down": "No missed POS payments in last 3 months"},

    # Credit card trajectory
    "CC_UTIL_MEAN":           {"up": "High average credit card utilisation",           "down": "Low average credit card utilisation"},
    "CC_UTIL_LAST3M":         {"up": "Recent credit card utilisation is high",         "down": "Recent credit card utilisation is low"},
    "CC_UTIL_TREND":          {"up": "Credit card utilisation rising recently",        "down": "Credit card utilisation stable or falling"},
    "CC_DRAW_RATIO_MEAN":     {"up": "High average credit card draw ratio",            "down": "Low average credit card draw ratio"},
    "CC_DRAW_RATIO_LAST3M":   {"up": "High draws from credit card recently",           "down": "Low draws from credit card recently"},
    "CC_DRAW_TREND":          {"up": "Credit card draws accelerating (stress signal)", "down": "Credit card draw rate stable or falling"},
    "CC_DPD_MAX":             {"up": "Credit card went significantly overdue",         "down": "Credit card stayed current"},
    "CC_BAD_MONTH_RATE":      {"up": "Frequently overdue on credit card",              "down": "Consistently current on credit card"},
    "CC_BAD_RATE_TREND":      {"up": "Credit card delinquency worsening recently",     "down": "Credit card status stable or improving"},
    "CC_ATM_DRAW_MEAN":       {"up": "Heavy ATM cash withdrawals (stress signal)",     "down": "Low ATM cash withdrawals"},
    "CC_PAY_MIN_RATIO":       {"up": "Paying only minimum on credit card",             "down": "Paying more than minimum on credit card"},
    "CC_OVERLIMIT_COUNT":     {"up": "Frequently exceeded credit card limit",          "down": "Rarely exceeded credit card limit"},

    # Installments trajectory
    "INST_MAX_LATE":          {"up": "History of significant payment delays",          "down": "No history of major payment delays"},
    "INST_MEAN_LATE":         {"up": "Payments tend to arrive late",                   "down": "Payments tend to arrive on time"},
    "INST_PAY_RATIO_MEAN":    {"up": "Typically underpays installments",               "down": "Consistently pays full installment amount"},
    "INST_UNDERPAID_RATE":    {"up": "Frequently underpays installments",              "down": "Rarely underpays installments"},
    "INST_MISSED_COUNT":      {"up": "Has missed installment payments",                "down": "Has not missed installment payments"},
    "INST_LATE_MEAN_LAST90":  {"up": "Payments late in the last 90 days",             "down": "Payments on time in the last 90 days"},
    "INST_LATE_TREND":        {"up": "Payment lateness increasing recently",           "down": "Payment lateness stable or improving"},
    "INST_UNDERPAID_TREND":   {"up": "Underpayment worsening recently",                "down": "Payment amounts stable or improving"},
    "INST_UNDERPAID_STREAK":  {"up": "Consecutive installments underpaid",             "down": "No streak of underpaid installments"},

    # Cross-table interactions
    "MULTI_SOURCE_DPD":       {"up": "Overdue across multiple loan types",             "down": "Current across all loan types"},
    "COMPOSITE_STRESS_TREND": {"up": "Multiple stress signals worsening together",     "down": "Stress signals stable across all accounts"},
    "LATE_X_REFUSAL":         {"up": "Late payments combined with prior rejections",   "down": "On-time payments and clean application history"},

    # Structural
    "THIN_FILE":              {"up": "Limited credit history available",               "down": "Established credit history"},
    "ADDR_MISMATCH":          {"up": "Address inconsistencies on file",                "down": "Consistent address information"},
    "DOC_COUNT":              {"up": "Fewer documents provided",                       "down": "Full set of documents provided"},
    "CONTACT_COUNT":          {"up": "Limited contact information provided",           "down": "Full contact information on file"},

    # Amounts
    "AMT_INCOME_TOTAL":       {"up": "Lower declared income",                          "down": "Higher declared income"},
    "AMT_CREDIT":             {"up": "Larger total loan amount",                       "down": "Smaller total loan amount"},
    "AMT_ANNUITY":            {"up": "High monthly repayment amount",                  "down": "Low monthly repayment amount"},
    "AMT_GOODS_PRICE":        {"up": "High-value item being financed",                 "down": "Lower-value item being financed"},

    # Region
    "REGION_POPULATION_RELATIVE":{"up": "Lives in a less densely populated area",     "down": "Lives in a more densely populated area"},
}


def label_for(feature: str, direction: str) -> str:
    """Return a plain-English label for a feature + direction pair."""
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
        data_dir = Path(data_dir)

        # LightGBM booster
        self.booster = lgb.Booster(
            model_file=str(data_dir / "pre_delinquency_model.txt")
        )

        # Feature names
        with open(data_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)

        # Baseline account vectors
        with open(data_dir / "baseline_vectors.json") as f:
            self.baselines = json.load(f)

        # Slider config
        with open(data_dir / "slider_config.json") as f:
            self.slider_config = json.load(f)

        # Tier thresholds
        with open(data_dir / "tier_config.json") as f:
            tc = json.load(f)
        self.nudge_threshold     = tc["nudge_threshold"]
        self.intervene_threshold = tc["intervene_threshold"]

        # Isotonic calibrator
        with open(data_dir / "calibrator.pkl", "rb") as f:
            self.calibrator = pickle.load(f)

        # Feature index lookup
        self.feature_index = {
            name: i for i, name in enumerate(self.feature_names)
        }

    def account_ids(self) -> list[str]:
        return list(self.baselines.keys())

    def account_meta(self, account_id: str) -> dict:
        """Return stored tier and score for an account."""
        if account_id not in self.baselines:
            return {}
        return {
            "tier":  self.baselines[account_id]["tier"],
            "score": self.baselines[account_id]["score"],
        }

    def predict(
        self,
        account_id: str,
        overrides: dict[str, float] | None = None,
        top_n: int = 10,
    ) -> tuple[float, str, list[dict]] | None:
        """
        Returns (calibrated_score, tier, drivers) for the given account.

        overrides : {feature_name: new_value} — what-if slider values.
                    Applied on top of the stored baseline vector.
        top_n     : number of driver explanations to return.
        """
        if account_id not in self.baselines:
            return None

        vector = np.array(
            self.baselines[account_id]["features"], dtype=float
        ).copy()

        # Apply any what-if overrides
        if overrides:
            for feature, value in overrides.items():
                idx = self.feature_index.get(feature)
                if idx is not None:
                    vector[idx] = float(value)

        # LightGBM prediction (raw score)
        raw_score = float(
            self.booster.predict(vector.reshape(1, -1))[0]
        )

        # Calibrated probability
        cal_score = float(
            self.calibrator.predict(np.array([raw_score]))[0]
        )
        cal_score = float(np.clip(cal_score, 0.0, 1.0))

        # Tier
        tier = assign_tier(
            cal_score,
            self.nudge_threshold,
            self.intervene_threshold,
        )

        # SHAP contributions via LightGBM pred_contrib
        # Returns shape (1, n_features + 1); last column is bias
        contribs_raw = self.booster.predict(
            vector.reshape(1, -1), pred_contrib=True
        )
        contribs = contribs_raw[0, :-1]  # drop bias term

        # Filter excluded features
        valid_idx = [
            i for i in range(len(contribs))
            if self.feature_names[i] not in EXCLUDE_FROM_DRIVERS
        ]

        # Sort by absolute contribution
        sorted_idx = sorted(
            valid_idx,
            key=lambda i: abs(contribs[i]),
            reverse=True,
        )
        top_idx = sorted_idx[:top_n]

        # Normalise to 0–100 impact scale
        max_impact = max((abs(contribs[i]) for i in top_idx), default=1.0)
        if max_impact == 0:
            max_impact = 1.0

        drivers = []
        for i in top_idx:
            direction = "up" if contribs[i] > 0 else "down"
            drivers.append({
                "label":     label_for(self.feature_names[i], direction),
                "feature":   self.feature_names[i],
                "impact":    round(abs(contribs[i]) / max_impact * 100),
                "shap":      round(float(contribs[i]), 6),
                "direction": direction,
            })

        return cal_score, tier, drivers


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python inference.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model_artifacts")
    model    = RiskModel(data_dir)

    print(f"Loaded model  : {len(model.feature_names)} features")
    print(f"Loaded accounts: {len(model.account_ids())}")
    print(f"Thresholds    : Nudge≥{model.nudge_threshold:.3f} | Intervene≥{model.intervene_threshold:.3f}\n")

    for account_id in model.account_ids()[:3]:
        result = model.predict(account_id, top_n=5)
        if result is None:
            continue
        score, tier, drivers = result
        print(f"Account {account_id}")
        print(f"  Score : {score:.4f}  Tier: {tier}")
        print(f"  Top drivers:")
        for d in drivers:
            bar   = "█" * (d["impact"] // 10)
            arrow = "↑" if d["direction"] == "up" else "↓"
            print(f"    {arrow} {bar:<10} {d['impact']:3d}%  {d['label']}")
        print()