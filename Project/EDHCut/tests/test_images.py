"""Card image and printing lookup: name resolution is case- and punctuation-insensitive, unknown
cards and un-backfilled image URIs fail with distinguishable errors, and the Scryfall search URLs
the notebooks link to are well-formed."""

from io import BytesIO

import pytest
from PIL import Image

from edhcut.db import connect
from edhcut.images import get_card_image, get_card_image_uri, get_card_printing, scryfall_search_url
from edhcut.ingest.scryfall import normalize_name


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, name, *, image_uri=None, set_code=None, collector_number=None):
    conn.execute(
        "INSERT INTO cards (oracle_id, name, image_uri, set_code, collector_number) VALUES (?, ?, ?, ?, ?)",
        (oracle_id, name, image_uri, set_code, collector_number),
    )
    conn.execute(
        "INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)",
        (normalize_name(name), oracle_id),
    )
    conn.commit()


def test_get_card_image_uri_resolves_by_name(db) -> None:
    _insert_card(db, "sol-ring-uid", "Sol Ring", image_uri="https://cards.scryfall.io/normal/sol-ring.jpg")
    assert get_card_image_uri(db, "Sol Ring") == "https://cards.scryfall.io/normal/sol-ring.jpg"


def test_get_card_image_uri_resolves_case_and_punctuation_insensitively(db) -> None:
    _insert_card(db, "urza-uid", "Urza's Saga", image_uri="https://cards.scryfall.io/normal/urza.jpg")
    assert get_card_image_uri(db, "URZA'S saga") == "https://cards.scryfall.io/normal/urza.jpg"


def test_get_card_image_uri_raises_key_error_for_unknown_card(db) -> None:
    with pytest.raises(KeyError):
        get_card_image_uri(db, "Not A Real Card")


def test_get_card_image_uri_raises_value_error_when_not_backfilled(db) -> None:
    _insert_card(db, "no-image-uid", "No Image Card", image_uri=None)
    with pytest.raises(ValueError):
        get_card_image_uri(db, "No Image Card")


class _FakeImageResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


class _FakeImageSession:
    def __init__(self, content: bytes):
        self._content = content
        self.requested_urls: list[str] = []

    def get(self, url, **kwargs):
        self.requested_urls.append(url)
        return _FakeImageResponse(self._content)


def _tiny_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buf, format="PNG")
    return buf.getvalue()


def test_get_card_image_fetches_and_decodes(db) -> None:
    _insert_card(db, "sol-ring-uid", "Sol Ring", image_uri="https://cards.scryfall.io/normal/sol-ring.jpg")
    session = _FakeImageSession(_tiny_png_bytes())

    image = get_card_image(db, "Sol Ring", session=session)

    assert isinstance(image, Image.Image)
    assert image.size == (2, 2)
    assert session.requested_urls == ["https://cards.scryfall.io/normal/sol-ring.jpg"]


def test_get_card_image_propagates_uri_resolution_errors(db) -> None:
    session = _FakeImageSession(_tiny_png_bytes())
    with pytest.raises(KeyError):
        get_card_image(db, "Not A Real Card", session=session)
    assert session.requested_urls == []  # never fetched -- failed before the network call


# --- get_card_printing ------------------------------------------------------------------------

def test_get_card_printing_resolves_by_name(db) -> None:
    _insert_card(db, "sol-ring-uid", "Sol Ring", set_code="msc", collector_number="211")
    assert get_card_printing(db, "Sol Ring") == ("msc", "211")


def test_get_card_printing_resolves_case_and_punctuation_insensitively(db) -> None:
    _insert_card(db, "urza-uid", "Urza's Saga", set_code="mh2", collector_number="251")
    assert get_card_printing(db, "URZA'S saga") == ("mh2", "251")


def test_get_card_printing_raises_key_error_for_unknown_card(db) -> None:
    with pytest.raises(KeyError):
        get_card_printing(db, "Not A Real Card")


def test_get_card_printing_raises_value_error_when_not_backfilled(db) -> None:
    _insert_card(db, "no-printing-uid", "No Printing Card")
    with pytest.raises(ValueError):
        get_card_printing(db, "No Printing Card")


# --- scryfall_search_url -----------------------------------------------------------------------

def test_scryfall_search_url_matches_known_example() -> None:
    printings = [("msc", "211"), ("otc", "176"), ("afc", "137")]
    expected = (
        "https://scryfall.com/search?q=s%3Amsc+cn%3A211+or+s%3Aotc+cn%3A176+or+"
        "s%3Aafc+cn%3A137&order=edhrec"
    )
    assert scryfall_search_url(printings) == expected


def test_scryfall_search_url_single_card() -> None:
    url = scryfall_search_url([("msc", "211")])
    assert url == "https://scryfall.com/search?q=s%3Amsc+cn%3A211&order=edhrec"


def test_scryfall_search_url_empty_list() -> None:
    assert scryfall_search_url([]) == "https://scryfall.com/search?q=&order=edhrec"


def test_scryfall_search_url_custom_order() -> None:
    url = scryfall_search_url([("msc", "211")], order="name")
    assert url.endswith("&order=name")


def test_scryfall_search_url_skips_pairs_missing_set_or_collector_number() -> None:
    printings = [("msc", "211"), (None, "5"), ("otc", None), ("afc", "137")]
    url = scryfall_search_url(printings)
    assert url == "https://scryfall.com/search?q=s%3Amsc+cn%3A211+or+s%3Aafc+cn%3A137&order=edhrec"


def test_scryfall_search_url_no_cap_includes_everything() -> None:
    printings = [(f"set{i:03d}", str(i)) for i in range(50)]
    url = scryfall_search_url(printings)
    for set_code, cn in printings:
        assert f"s%3A{set_code}+cn%3A{cn}" in url


def test_scryfall_search_url_respects_max_length() -> None:
    printings = [(f"set{i:04d}", f"{i:04d}") for i in range(200)]
    full_url = scryfall_search_url(printings)
    capped_url = scryfall_search_url(printings, max_url_length=500)

    assert len(capped_url) <= 500
    assert len(capped_url) < len(full_url)
    # pairs are kept in the given order, front-loaded, not an arbitrary subset
    assert "set0000" in capped_url
    assert "set0199" not in capped_url


def test_scryfall_search_url_max_length_too_small_for_even_one_pair() -> None:
    url = scryfall_search_url([("a-very-long-set-code", "9999")], max_url_length=10)
    assert url == "https://scryfall.com/search?q=&order=edhrec"


def test_scryfall_search_url_much_shorter_than_name_based_would_be() -> None:
    # The whole point of switching to s:/cn: -- a long, punctuated, partner-pair-style name vs.
    # its fixed-ish-width printing pair.
    long_name_term_len = len('name:"Kyler, Sigardian Emissary // Some Very Long Partner Name"')
    printing_term_len = len("s:otj cn:176")
    assert printing_term_len < long_name_term_len
