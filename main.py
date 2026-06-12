"""
Pre-Delinquency Risk Scoring - Backend API

Serves account-level risk scores, tiers, and SHAP-derived drivers to the
React dashboard. Reads from demo_accounts.json (exported from the Kaggle
notebook) -- no live model inference required for the hackathon demo.

Run:
    pip install fastapi uvicorn
    uvicorn main:app --reload

Then the API is available at http://localhost:8000
Interactive docs at http://localhost:8000/docs
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import RiskModel, assign_tier


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pre-Delinquency Risk Scoring API",
    description="Account risk scores, tiers, and explainability drivers.",
    version="1.0.0",
)

# Allow the React dev server (and any origin, for the demo) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "demo_accounts.json"

with open(DATA_PATH) as f:
    ACCOUNTS = json.load(f)

ACCOUNTS_BY_ID = {a["id"]: a for a in ACCOUNTS}

# Live model for the "what-if" slider (Path B). If the required files
# aren't present, /api/predict and /api/sliders will return a 503 but
# the rest of the API keeps working off the static demo data.
try:
    risk_model = RiskModel(Path(__file__).parent)
except FileNotFoundError as e:
    risk_model = None
    print(f"Live model not loaded ({e}). /api/predict and /api/sliders disabled.")

# UI-friendly slider definitions: label, units, and a sensible display
# range. DEBT_TO_CREDIT_RATIO and INST_MAX_DAYS_LATE have long-tailed
# raw distributions, so the UI range is narrower than the model's
# observed min/max -- the model can still score values outside this
# range, but the slider stays meaningful for a demo.
SLIDER_DISPLAY = {
    "DEBT_TO_CREDIT_RATIO": {
        "label": "Credit utilization (debt to credit ratio)",
        "min": 0.0,
        "max": 1.5,
        "step": 0.01,
        "unit": "ratio",
    },
    "INST_MAX_DAYS_LATE": {
        "label": "Worst payment delay (days late)",
        "min": -30,
        "max": 90,
        "step": 1,
        "unit": "days",
    },
    "EXT_SOURCE_MEAN": {
        "label": "External credit bureau score",
        "min": 0.0,
        "max": 0.86,
        "step": 0.01,
        "unit": "score",
    },
    "CC_UTILITY_MEAN": {
        "label": "Average credit card utilization",
        "min": 0.0,
        "max": 1.2,
        "step": 0.01,
        "unit": "ratio",
    },
}

ACTIONS = {
    "Watch": {
        "title": "Continue monitoring",
        "detail": (
            "No action required. Keep this account in the standard "
            "monitoring cycle and re-score next period."
        ),
    },
    "Nudge": {
        "title": "Send a proactive reminder",
        "detail": (
            "Trigger an SMS and email reminder ahead of the next due date, "
            "and surface this account to the agent queue for a light-touch "
            "check-in."
        ),
    },
    "Intervene": {
        "title": "Escalate for outreach",
        "detail": (
            "Route to a collections specialist for a call. Discuss a "
            "payment plan or restructuring before the account goes past due."
        ),
    },
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class Driver(BaseModel):
    label: str
    impact: int
    direction: str  # "up" or "down"


class AccountSummary(BaseModel):
    id: str
    score: float
    tier: str
    trend: list[float]
    top_driver: str


class Action(BaseModel):
    title: str
    detail: str


class AccountDetail(BaseModel):
    id: str
    score: float
    tier: str
    trend: list[float]
    drivers: list[Driver]
    action: Action


class TierStat(BaseModel):
    count: int
    pct: int


class PortfolioSummary(BaseModel):
    total_accounts: int
    tiers: dict[str, TierStat]


class SliderDef(BaseModel):
    feature: str
    label: str
    min: float
    max: float
    step: float
    unit: str
    current: float


class PredictRequest(BaseModel):
    account_id: str
    overrides: dict[str, float] = {}


class PredictResponse(BaseModel):
    id: str
    score: float
    tier: str
    drivers: list[Driver]
    action: Action


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "pre-delinquency-risk-scoring",
        "accounts_loaded": len(ACCOUNTS),
        "live_model_loaded": risk_model is not None,
        "endpoints": [
            "/api/portfolio/summary",
            "/api/accounts",
            "/api/accounts/{account_id}",
            "/api/sliders/{account_id}",
            "/api/predict",
        ],
    }


@app.get("/api/portfolio/summary", response_model=PortfolioSummary)
def portfolio_summary():
    """Counts and percentages per tier, for the top stat cards."""
    tier_counts = {"Watch": 0, "Nudge": 0, "Intervene": 0}
    for account in ACCOUNTS:
        tier_counts[account["tier"]] += 1

    total = len(ACCOUNTS)
    tiers = {
        tier: TierStat(count=count, pct=round(count / total * 100) if total else 0)
        for tier, count in tier_counts.items()
    }

    return PortfolioSummary(total_accounts=total, tiers=tiers)


@app.get("/api/accounts", response_model=list[AccountSummary])
def list_accounts(tier: Optional[str] = None):
    """
    Account queue for the dashboard table.

    Optional query param `tier` filters to Watch / Nudge / Intervene.
    Results are sorted by risk score, highest first.
    """
    data = ACCOUNTS

    if tier:
        if tier not in ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier '{tier}'. Must be one of: {list(ACTIONS)}",
            )
        data = [a for a in data if a["tier"] == tier]

    sorted_data = sorted(data, key=lambda a: a["score"], reverse=True)

    return [
        AccountSummary(
            id=a["id"],
            score=a["score"],
            tier=a["tier"],
            trend=a["trend"],
            top_driver=a["drivers"][0]["label"] if a["drivers"] else "",
        )
        for a in sorted_data
    ]


@app.get("/api/accounts/{account_id}", response_model=AccountDetail)
def get_account(account_id: str):
    """Full detail for one account: score, trend, drivers, recommended action."""
    account = ACCOUNTS_BY_ID.get(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    return AccountDetail(
        id=account["id"],
        score=account["score"],
        tier=account["tier"],
        trend=account["trend"],
        drivers=[Driver(**d) for d in account["drivers"]],
        action=Action(**ACTIONS[account["tier"]]),
    )


@app.get("/api/sliders/{account_id}", response_model=list[SliderDef])
def get_sliders(account_id: str):
    """
    Slider definitions for the what-if panel, with the account's current
    value for each slider feature pre-filled.
    """
    if risk_model is None:
        raise HTTPException(status_code=503, detail="Live model not available")

    if account_id not in risk_model.baselines:
        raise HTTPException(status_code=404, detail="Account not found")

    vector = risk_model.baselines[account_id]

    sliders = []
    for feature, display in SLIDER_DISPLAY.items():
        idx = risk_model.feature_index.get(feature)
        current = float(vector[idx]) if idx is not None else display["min"]
        sliders.append(SliderDef(
            feature=feature,
            label=display["label"],
            min=display["min"],
            max=display["max"],
            step=display["step"],
            unit=display["unit"],
            current=current,
        ))

    return sliders


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Recompute score, tier, and drivers for an account with the given
    feature overrides applied (live "what-if" inference).
    """
    if risk_model is None:
        raise HTTPException(status_code=503, detail="Live model not available")

    result = risk_model.predict(req.account_id, overrides=req.overrides)
    if result is None:
        raise HTTPException(status_code=404, detail="Account not found")

    score, drivers = result
    tier = assign_tier(score)

    return PredictResponse(
        id=req.account_id,
        score=round(score, 2),
        tier=tier,
        drivers=[Driver(**d) for d in drivers],
        action=Action(**ACTIONS[tier]),
    )
