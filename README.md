# Pre-Delinquency Risk Scoring — Backend API

FastAPI backend that serves account-level risk scores, tiers, and explainability drivers to the dashboard. The portfolio/account endpoints read from `demo_accounts.json` (itself generated from the live model); the what-if endpoints (`/api/sliders`, `/api/predict`) run the model live. Scoring uses a LightGBM booster with an isotonic calibrator (`calibrator.pkl`) on top, and Nudge/Intervene tier thresholds come from `tier_config.json` rather than being hardcoded.

## Requirements

- Python 3.10+

## Setup & Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API is available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

## API Reference

### `GET /`

Health check. Returns service name, number of accounts loaded, and available endpoints.

---

### `GET /api/portfolio/summary`

Returns account counts and percentages broken down by risk tier, plus the
tier thresholds the model is currently using (from `tier_config.json`),
so the dashboard can scale its own risk-segment coloring accordingly.

**Response**
```json
{
  "total_accounts": 60,
  "tiers": {
    "Watch":     { "count": 11, "pct": 18 },
    "Nudge":     { "count": 7,  "pct": 12 },
    "Intervene": { "count": 42, "pct": 70 }
  },
  "thresholds": {
    "nudge": 0.0457,
    "intervene": 0.0913
  }
}
```

---

### `GET /api/accounts`

Returns all accounts sorted by risk score (highest first).

**Query params**

| Param | Type   | Description                                  |
|-------|--------|----------------------------------------------|
| `tier`  | string | Filter by tier: `Watch`, `Nudge`, `Intervene` |

**Response** — array of:
```json
{
  "id": "ACC-293721",
  "score": 0.603,
  "tier": "Intervene",
  "trend": [0.603, 0.603, 0.603, 0.603, 0.603, 0.603],
  "top_driver": "Below-average external credit score"
}
```

---

### `GET /api/accounts/{account_id}`

Full detail for a single account including all SHAP drivers and the recommended action.

**Response**
```json
{
  "id": "ACC-293721",
  "score": 0.603,
  "tier": "Intervene",
  "trend": [0.603, 0.603, 0.603, 0.603, 0.603, 0.603],
  "drivers": [
    {
      "label": "Below-average external credit score",
      "impact": 100,
      "direction": "up",
      "feature": "EXT_MEAN",
      "shap": 0.788903
    }
  ],
  "action": {
    "title": "Escalate for outreach",
    "detail": "Route to a collections specialist for a call..."
  }
}
```

Returns `404` if the account ID is not found.

---

### `GET /api/sliders/{account_id}`

Slider definitions for the what-if panel (from `slider_config.json`), with
the account's current value for each feature pre-filled. Returns `503` if
the live model failed to load, `404` if the account ID is not found.

**Response** — array of:
```json
{
  "feature": "EXT_MEAN",
  "label": "External credit score (mean)",
  "min": 0.0,
  "max": 1.0,
  "step": 0.01,
  "current": 0.1195
}
```

---

### `POST /api/predict`

Recomputes score, tier, and drivers for an account with the given feature
overrides applied on top of its baseline vector (live "what-if" inference).
Returns `503` if the live model failed to load, `404` if the account ID is
not found.

**Request**
```json
{ "account_id": "ACC-293721", "overrides": { "EXT_MEAN": 0.6 } }
```

**Response** — same shape as `GET /api/accounts/{account_id}`, minus `trend`.

## Risk Tiers

Tier thresholds are data-driven (`tier_config.json`), not hardcoded — see
`GET /api/portfolio/summary` for the current cutoffs.

| Tier       | Recommended Action                              |
|------------|-------------------------------------------------|
| Watch      | Continue monitoring, re-score next period       |
| Nudge      | Send proactive SMS/email reminder               |
| Intervene  | Escalate to collections specialist              |
