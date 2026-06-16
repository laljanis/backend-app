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
import os
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

cors_allow_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

# One-tunnel ngrok usage is same-origin through Vite's /api proxy. If the
# backend is exposed separately, set CORS_ALLOW_ORIGINS to the frontend ngrok URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "demo_accounts.json"

with open(DATA_PATH) as f:
    ACCOUNTS = json.load(f)

with open(Path(__file__).parent / "tier_config.json") as f:
    _TIER_CONFIG = json.load(f)
NUDGE_THRESHOLD = _TIER_CONFIG["nudge_threshold"]
INTERVENE_THRESHOLD = _TIER_CONFIG["intervene_threshold"]

# Reassign tiers dynamically based on the configured thresholds to ensure consistency
for account in ACCOUNTS:
    account["tier"] = assign_tier(account["score"], NUDGE_THRESHOLD, INTERVENE_THRESHOLD)

ACCOUNTS_BY_ID = {a["id"]: a for a in ACCOUNTS}

# Live model for the "what-if" slider (Path B). If the required files
# aren't present, /api/predict and /api/sliders will return a 503 but
# the rest of the API keeps working off the static demo data.
try:
    risk_model = RiskModel(Path(__file__).parent)
except FileNotFoundError as e:
    risk_model = None
    print(f"Live model not loaded ({e}). /api/predict and /api/sliders disabled.")

# demo_accounts.json / the dashboard use "ACC-<SK_ID_CURR>" ids, but the live
# model's baseline vectors are keyed by the raw SK_ID_CURR. Strip the prefix
# before looking anything up against risk_model.
ACCOUNT_ID_PREFIX = "ACC-"


def to_raw_id(account_id: str) -> str:
    return account_id.removeprefix(ACCOUNT_ID_PREFIX)


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
    feature: Optional[str] = None
    shap: Optional[float] = None


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


class TierThresholds(BaseModel):
    nudge: float
    intervene: float


class PortfolioSummary(BaseModel):
    total_accounts: int
    tiers: dict[str, TierStat]
    thresholds: TierThresholds


class SliderDef(BaseModel):
    feature: str
    label: str
    min: float
    max: float
    step: float
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

    return PortfolioSummary(
        total_accounts=total,
        tiers=tiers,
        thresholds=TierThresholds(nudge=NUDGE_THRESHOLD, intervene=INTERVENE_THRESHOLD),
    )


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

    raw_id = to_raw_id(account_id)
    if raw_id not in risk_model.baselines:
        raise HTTPException(status_code=404, detail="Account not found")

    vector = risk_model.baselines[raw_id]["features"]

    sliders = []
    for slider in risk_model.slider_config:
        feature = slider["feature"]
        idx = risk_model.feature_index.get(feature)
        current = float(vector[idx]) if idx is not None else slider["default"]
        sliders.append(SliderDef(
            feature=feature,
            label=slider["label"],
            min=slider["min"],
            max=slider["max"],
            step=slider["step"],
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

    result = risk_model.predict(to_raw_id(req.account_id), overrides=req.overrides)
    if result is None:
        raise HTTPException(status_code=404, detail="Account not found")

    score, tier, drivers = result

    return PredictResponse(
        id=req.account_id,
        score=round(score, 4),
        tier=tier,
        drivers=[Driver(**d) for d in drivers],
        action=Action(**ACTIONS[tier]),
    )
