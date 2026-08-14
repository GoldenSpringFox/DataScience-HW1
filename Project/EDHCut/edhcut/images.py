"""Card image lookup — resolve a card name to its Scryfall CDN image and fetch it.

`cards.image_uri` (task 5.2's ingest, `edhcut.ingest.scryfall._image_uri`) already stores each
card's own "normal"-size image URL from the bulk file, so this needs no fresh Scryfall API
call to find the URL — only to fetch the image bytes themselves, through the same
rate-limited/cached `scryfall` session every other ingest module uses (plan §5: hotlinking the
image CDN for occasional/UI lookups like this is explicitly fine per Scryfall's own
guidelines, distinct from the "don't hotlink at scale" caution aimed at high-volume/production
serving). `requests-cache` caches the image bytes too, so looking up the same card twice
doesn't re-fetch.
"""

from __future__ import annotations

import sqlite3
from io import BytesIO
from typing import Sequence
from urllib.parse import quote_plus

from PIL import Image

from edhcut.http import RateLimitedSession, get_session
from edhcut.ingest.scryfall import normalize_name


def get_card_image_uri(conn: sqlite3.Connection, name: str) -> str:
    """The stored Scryfall image URL for a card name (resolved via `card_names`, so any known
    alias/face name works, not just the primary name)."""
    row = conn.execute(
        """
        SELECT c.image_uri FROM card_names cn
        JOIN cards c ON c.oracle_id = cn.oracle_id
        WHERE cn.name_normalized = ?
        """,
        (normalize_name(name),),
    ).fetchone()
    if row is None:
        raise KeyError(f"Card {name!r} not found in card_names.")
    if row[0] is None:
        raise ValueError(
            f"Card {name!r} has no stored image_uri — re-run `python -m edhcut.ingest.scryfall` "
            "to backfill it."
        )
    return row[0]


def get_card_printing(conn: sqlite3.Connection, name: str) -> tuple[str, str]:
    """`(set_code, collector_number)` for a card name (resolved via `card_names`, so any known
    alias/face name works) -- the pair `scryfall_search_url` needs. Raises `KeyError` if the
    name isn't known, `ValueError` if the card is known but the pair hasn't been backfilled
    (pre-task-6.3 ingest; re-run `python -m edhcut.ingest.scryfall`)."""
    row = conn.execute(
        """
        SELECT c.set_code, c.collector_number FROM card_names cn
        JOIN cards c ON c.oracle_id = cn.oracle_id
        WHERE cn.name_normalized = ?
        """,
        (normalize_name(name),),
    ).fetchone()
    if row is None:
        raise KeyError(f"Card {name!r} not found in card_names.")
    set_code, collector_number = row
    if set_code is None or collector_number is None:
        raise ValueError(
            f"Card {name!r} has no stored set_code/collector_number — re-run "
            "`python -m edhcut.ingest.scryfall` to backfill it."
        )
    return set_code, collector_number


def get_card_image(
    conn: sqlite3.Connection, name: str, *, session: RateLimitedSession | None = None
) -> Image.Image:
    """Fetch and decode a card's image as a Pillow `Image`, ready to `.show()`, `.save(...)`,
    or display inline in a notebook."""
    uri = get_card_image_uri(conn, name)
    session = session or get_session("scryfall")
    response = session.get(uri)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def _build_scryfall_search_url_by_printing(printings: Sequence[tuple[str, str]], *, order: str) -> str:
    query = " or ".join(f"s:{set_code} cn:{collector_number}" for set_code, collector_number in printings)
    return f"https://scryfall.com/search?q={quote_plus(query)}&order={order}"


def scryfall_search_url(
    printings: Sequence[tuple[str, str]], *, order: str = "edhrec", max_url_length: int | None = None
) -> str:
    """A Scryfall search URL matching each `(set_code, collector_number)` pair exactly
    (`s:<set> cn:<cn> or s:<set> cn:<cn> ...`), e.g. for eyeballing a whole cluster/topic's cards
    on Scryfall's own grid view instead of fetching every card image inline. `order` is any
    Scryfall sort key ("edhrec" -- EDHREC rank, the default -- "name", "cmc", etc.).

    **Always the `s:`/`cn:` form, not `name:"..."` or `oracleid:<uuid>`** -- both fields are
    short, close-to-fixed-width codes (typically 2-5 characters each), so this is reliably the
    most compact per-card term: a full card name can run 30+ characters once quoted/encoded
    (worse for partner pairs, apostrophes, `//` DFC separators), and an oracle ID is a fixed
    36-character UUID either way. `cards.set_code`/`cards.collector_number` (task 5.2's Scryfall
    ingest, one representative printing per `oracle_id`) are the source for these -- resolve a
    card name to its pair via a query like
    `SELECT set_code, collector_number FROM cards WHERE oracle_id = ...` before calling this.

    Without `max_url_length`, every pair is included -- a large cluster's full member list can
    then build a URL of essentially unbounded length (a few hundred cards easily exceeds what a
    browser or Scryfall's own server will accept), so most real callers should set it. When
    given, pairs are kept **in the order passed in** until the next one would push the URL over
    the limit, then truncated there -- callers who want the highest-ranked cards in a capped
    link should sort `printings` by rank *before* calling this (e.g. `cards.edhrec_rank`
    ascending). A pair with a missing `set_code` or `collector_number` (`None`) is silently
    skipped -- can't build a valid `s:`/`cn:` term for it."""
    pairs = [(s, cn) for s, cn in printings if s is not None and cn is not None]
    if max_url_length is None:
        return _build_scryfall_search_url_by_printing(pairs, order=order)

    kept: list[tuple[str, str]] = []
    for pair in pairs:
        if len(_build_scryfall_search_url_by_printing(kept + [pair], order=order)) > max_url_length:
            break
        kept.append(pair)
    return _build_scryfall_search_url_by_printing(kept, order=order)
