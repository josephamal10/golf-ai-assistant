from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, get_args
import json
import logging
import re


logger = logging.getLogger(__name__)

Route = Literal['static', 'live', 'prediction', 'off_topic']
ROUTES: tuple[str, ...] = get_args(Route)
FALLBACK_ROUTE: Route = 'static'                           # Phase 1 behaviour, always safe

ROUTER_PROMPT = """You sort questions for a golf assistant into exactly one route.

static: golf knowledge that does not depend on today's date. Rules, terms, scoring,
  equipment, course design, the history of the game and of tournaments, past winners,
  and players' biographies, careers and records. Answered from an encyclopedia.
live: golf questions that need current or recent information. Today's, this week's or
  this season's events, latest results and leaderboards, current world rankings,
  current form, upcoming schedules and recent news. Words like "now", "current",
  "latest", "this week", "yesterday" or "right now" usually mean live.
prediction: golf questions about a future outcome or asking for a pick. Who will win,
  favourites, odds, betting, fantasy or tipping picks, "should I bet on".
off_topic: anything that is not about golf.

If a golf question could be static or live, choose static. Today's date is {today}.
Reply with JSON only: {{"route": "<one route>", "reason": "<under 15 words>"}}
"""
ROUTE_WORD_RE = re.compile(r'\b(static|live|prediction|off_topic|off-topic)\b', re.I)


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str = ''


@dataclass
class LiveAnswer:
    """An answer grounded in Google Search, shown to the user unmodified (Google's terms)."""

    text: str
    sources: list[dict[str, str]] = field(default_factory=list)   # [{title, url}], [n] = index + 1
    search_suggestions_html: str = ''                               # must be displayed with the answer
    queries: list[str] = field(default_factory=list)


def router_prompt(today: date | None = None) -> str:
    return ROUTER_PROMPT.format(today=(today or date.today()).isoformat())


def parse_route(text: str) -> RouteDecision:
    """JSON reply → decision. A stray route word still counts; anything else is static."""
    cleaned = (text or '').strip().removeprefix('```json').removeprefix('```').removesuffix('```')
    try:
        data = json.loads(cleaned)
        route = str(data.get('route', '')).strip().lower().replace('-', '_')
        if route in ROUTES:
            return RouteDecision(route, str(data.get('reason') or '')[:200])
    except (ValueError, AttributeError):
        pass
    found = ROUTE_WORD_RE.search(text or '')
    if found:
        return RouteDecision(found.group(1).lower().replace('-', '_'))
    logger.warning('Router reply had no route, using %s: %r', FALLBACK_ROUTE, (text or '')[:200])
    return RouteDecision(FALLBACK_ROUTE, 'router reply unreadable')
