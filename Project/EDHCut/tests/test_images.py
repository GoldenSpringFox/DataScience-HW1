from io import BytesIO

import pytest
from PIL import Image

from edhcut.db import connect
from edhcut.images import get_card_image, get_card_image_uri
from edhcut.ingest.scryfall import normalize_name


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, name, *, image_uri=None):
    conn.execute(
        "INSERT INTO cards (oracle_id, name, image_uri) VALUES (?, ?, ?)",
        (oracle_id, name, image_uri),
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
