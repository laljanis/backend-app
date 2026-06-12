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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "pre-delinquency-risk-scoring",
        "accounts_loaded": len(ACCOUNTS),
        "endpoints": [
            "/api/portfolio/summary",
            "/api/accounts",
            "/api/accounts/{account_id}",
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
