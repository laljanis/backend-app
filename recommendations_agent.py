"""
Generates "next best action" recommendations for an account using a
Google ADK agent (Gemini Flash-Lite), grounded in that account's real
score/tier/SHAP drivers. Results are cached in memory per account for
CACHE_TTL_SECONDS so repeat page views don't re-call Gemini.
"""

import json
import time
from typing import Literal, Optional

from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

CACHE_TTL_SECONDS = 30 * 60
APP_NAME = "pre-delinquency-recommendations"
USER_ID = "system"


class RecommendationItem(BaseModel):
    title: str = Field(description="Short actionable recommendation, under 12 words")
    priority: Literal["Critical", "High", "Medium", "Low"]
    reduction: str = Field(description="Estimated risk-score reduction range, e.g. '8-12%'")
    confidence: str = Field(description="Model confidence, e.g. '87%'")
    category: Literal["credit", "payment", "monitor", "outreach"]


class RecommendationsOutput(BaseModel):
    recommendations: list[RecommendationItem]


INSTRUCTION = """You are a credit-risk advisor for a collections team.
You will receive a JSON object with an account's risk score (0-1), its
tier (Watch/Nudge/Intervene), and its top SHAP risk drivers (label,
direction, relative impact). Return exactly 3 next-best-action
recommendations grounded in the actual drivers given -- do not invent
unrelated factors. Order by priority, descending. Priority should track
tier: Intervene accounts get Critical/High priority actions, Nudge
accounts get Medium/High, Watch accounts get Low/Medium. Respond with
JSON only, matching the response schema."""


class RecommendationAgent:
    def __init__(self):
        self._agent = Agent(
            name="recommendation_agent",
            model="gemini-flash-lite-latest",
            instruction=INSTRUCTION,
            output_schema=RecommendationsOutput,
            output_key="result",
        )
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self._agent,
            app_name=APP_NAME,
            session_service=self._session_service,
        )
        self._sessions_ready: set[str] = set()
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    async def _ensure_session(self, account_id: str) -> None:
        if account_id in self._sessions_ready:
            return
        await self._session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=account_id,
        )
        self._sessions_ready.add(account_id)

    async def get_recommendations(self, account: dict) -> list[dict]:
        cached = self._cache.get(account["id"])
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]

        items = await self._generate(account)
        self._cache[account["id"]] = (time.time(), items)
        return items

    async def _generate(self, account: dict) -> list[dict]:
        await self._ensure_session(account["id"])

        payload = {
            "score": account["score"],
            "tier": account["tier"],
            "drivers": account["drivers"][:5],
        }
        message = types.Content(
            role="user", parts=[types.Part(text=json.dumps(payload))],
        )

        result: Optional[RecommendationsOutput] = None
        async for event in self._runner.run_async(
            user_id=USER_ID, session_id=account["id"], new_message=message,
        ):
            if event.is_final_response() and event.content:
                result = RecommendationsOutput.model_validate_json(
                    event.content.parts[0].text
                )

        if result is None:
            raise RuntimeError("Recommendation agent returned no response")

        return [item.model_dump() for item in result.recommendations]
