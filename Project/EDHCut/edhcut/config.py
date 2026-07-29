"""Central configuration: commander slots, data paths, rate limits, User-Agent.

See plan `EDHCut_PLAN.md` §1 (commander roster), §2.1 (storage layout), §5 (ToS/rate-limit
notes per source).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

USER_AGENT = "EDHCut/0.1 (avivg2001@gmail.com)"

# Each slot is a list of 1 (single commander) or 2 (partner pair) card names,
# in plan §1 roster order.
COMMANDER_SLOTS: list[list[str]] = [
    ["Krenko, Mob Boss"],
    ["Kyler, Sigardian Emissary"],
    ["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"],
    ["Yenna, Redtooth Regent"],
    ["Orysa, Tide Choreographer"],
]


@dataclass(frozen=True)
class Paths:
    data_dir: Path = PACKAGE_ROOT / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "edhcut.db"

    @property
    def http_cache_path(self) -> Path:
        return self.data_dir / "http_cache.sqlite"

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def fixtures_dir(self) -> Path:
        return self.data_dir / "fixtures" / "my_decks"


@dataclass(frozen=True)
class RateLimit:
    """Per-source politeness settings. See plan §5 for the ToS basis of each value."""

    min_delay_seconds: float
    cache_expire_seconds: int


DAY = 60 * 60 * 24

RATE_LIMITS: dict[str, RateLimit] = {
    # Bulk file download, not a per-request API — delay is a formality.
    "scryfall": RateLimit(min_delay_seconds=0.1, cache_expire_seconds=DAY),
    # Both undocumented, per-request endpoints (plan §5): a wider margin than the ~1 req/s
    # floor the plan calls tolerated, and a shared two-week cache expiry so a re-fetch
    # picks up updated decks/stats reasonably promptly without hammering either site.
    "archidekt": RateLimit(min_delay_seconds=2.0, cache_expire_seconds=DAY * 14),
    "edhrec": RateLimit(min_delay_seconds=2.0, cache_expire_seconds=DAY * 14),
}


@dataclass(frozen=True)
class Config:
    commander_slots: list[list[str]] = field(default_factory=lambda: COMMANDER_SLOTS)
    user_agent: str = USER_AGENT
    paths: Paths = field(default_factory=Paths)
    rate_limits: dict[str, RateLimit] = field(default_factory=lambda: dict(RATE_LIMITS))
    decks_per_commander: int = 300


CONFIG = Config()
