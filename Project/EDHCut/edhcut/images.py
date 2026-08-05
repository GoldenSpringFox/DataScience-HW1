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
