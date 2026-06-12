# Pre-Delinquency Risk Scoring — Backend API

FastAPI backend that serves account-level risk scores, tiers, and explainability drivers to the dashboard. No live model inference — data is read from `demo_accounts.json` at startup.

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

Returns account counts and percentages broken down by risk tier.

**Response**
```json
{
  "total_accounts": 120,
  "tiers": {
    "Watch":     { "count": 60, "pct": 50 },
    "Nudge":     { "count": 36, "pct": 30 },
    "Intervene": { "count": 24, "pct": 20 }
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
  "id": "ACC-001",
  "score": 0.87,
  "tier": "Intervene",
  "trend": [0.62, 0.71, 0.79, 0.87],
  "top_driver": "Missed payment last cycle"
}
```

---

### `GET /api/accounts/{account_id}`

Full detail for a single account including all SHAP drivers and the recommended action.

**Response**
```json
{
  "id": "ACC-001",
  "score": 0.87,
  "tier": "Intervene",
  "trend": [0.62, 0.71, 0.79, 0.87],
  "drivers": [
    { "label": "Missed payment last cycle", "impact": 23, "direction": "up" }
  ],
  "action": {
    "title": "Escalate for outreach",
    "detail": "Route to a collections specialist for a call..."
  }
}
```

Returns `404` if the account ID is not found.

## Risk Tiers

| Tier       | Recommended Action                              |
|------------|-------------------------------------------------|
| Watch      | Continue monitoring, re-score next period       |
| Nudge      | Send proactive SMS/email reminder               |
| Intervene  | Escalate to collections specialist              |
