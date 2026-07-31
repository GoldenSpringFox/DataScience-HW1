"""Tests for the pure tag-hierarchy logic — no network, no gzip fakery (matches scryfall.py's
own test scope: `run()`'s download plumbing isn't unit-tested, only the risky logic is)."""

from edhcut.db import connect
from edhcut.ingest.tagger_bulk import (
    _write_card_tags,
    _write_tag_aliases,
    ancestor_slugs,
    expand_taggings,
    extract_aliases,
)


def _tag(tag_id, slug, parent_ids=None, taggings=None, aliases=None):
    return {
        "id": tag_id,
        "slug": slug,
        "parent_ids": parent_ids or [],
        "taggings": taggings or [],
        "aliases": aliases or [],
    }


def test_ancestor_slugs_walks_parent_chain() -> None:
    registry = {
        "removal": _tag("removal", "removal"),
        "boardwipe": _tag("boardwipe", "boardwipe", parent_ids=["removal"]),
        "boardwipe-creatures": _tag("boardwipe-creatures", "boardwipe-creatures", parent_ids=["boardwipe"]),
    }
    assert ancestor_slugs("boardwipe-creatures", registry) == {"boardwipe", "removal"}


def test_ancestor_slugs_root_tag_has_no_ancestors() -> None:
    registry = {"removal": _tag("removal", "removal")}
    assert ancestor_slugs("removal", registry) == set()


def test_ancestor_slugs_handles_multiple_parents_diamond() -> None:
    registry = {
        "root": _tag("root", "root"),
        "a": _tag("a", "a", parent_ids=["root"]),
        "b": _tag("b", "b", parent_ids=["root"]),
        "child": _tag("child", "child", parent_ids=["a", "b"]),
    }
    assert ancestor_slugs("child", registry) == {"a", "b", "root"}


def test_ancestor_slugs_ignores_dangling_parent_reference() -> None:
    registry = {"child": _tag("child", "child", parent_ids=["deleted-tag-id"])}
    assert ancestor_slugs("child", registry) == set()


def test_ancestor_slugs_is_cycle_safe() -> None:
    registry = {
        "a": _tag("a", "a", parent_ids=["b"]),
        "b": _tag("b", "b", parent_ids=["a"]),  # malformed cycle, shouldn't happen but don't hang
    }
    # Terminates rather than looping forever; walking the cycle from "a" reaches both nodes.
    assert ancestor_slugs("a", registry) == {"a", "b"}


def test_expand_taggings_includes_tag_and_ancestors() -> None:
    registry = {
        "removal": _tag("removal", "removal"),
        "boardwipe": _tag("boardwipe", "boardwipe", parent_ids=["removal"]),
        "boardwipe-creatures": _tag(
            "boardwipe-creatures", "boardwipe-creatures", parent_ids=["boardwipe"],
            taggings=[{"oracle_id": "wrath-uid"}],
        ),
    }
    result = expand_taggings(registry, known_oracle_ids={"wrath-uid"})
    assert result.rows == {
        ("wrath-uid", "boardwipe-creatures"),
        ("wrath-uid", "boardwipe"),
        ("wrath-uid", "removal"),
    }
    assert result.total_taggings == 1
    assert result.unresolved_oracle_ids == 0
    assert result.tagged_oracle_ids == {"wrath-uid"}


def test_expand_taggings_skips_oracle_ids_outside_known_set() -> None:
    registry = {
        "removal": _tag("removal", "removal", taggings=[{"oracle_id": "banned-uid"}]),
    }
    result = expand_taggings(registry, known_oracle_ids=set())
    assert result.rows == set()
    assert result.total_taggings == 1
    assert result.unresolved_oracle_ids == 1
    assert result.tagged_oracle_ids == set()


def test_expand_taggings_dedupes_shared_ancestor_across_multiple_tags() -> None:
    registry = {
        "removal": _tag("removal", "removal"),
        "boardwipe": _tag(
            "boardwipe", "boardwipe", parent_ids=["removal"], taggings=[{"oracle_id": "card-uid"}]
        ),
        "spot-removal": _tag(
            "spot-removal", "spot-removal", parent_ids=["removal"], taggings=[{"oracle_id": "card-uid"}]
        ),
    }
    result = expand_taggings(registry, known_oracle_ids={"card-uid"})
    # "removal" reached via two different tags on the same card — one row, not two.
    assert result.rows == {
        ("card-uid", "boardwipe"),
        ("card-uid", "spot-removal"),
        ("card-uid", "removal"),
    }
    assert result.tag_counts["removal"] == 1


def test_write_card_tags_replaces_source_wholesale_not_additively(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cards (oracle_id, name) VALUES (?, ?)", ("card-uid", "Some Card")
        )
        conn.execute(
            "INSERT INTO card_tags (oracle_id, tag, source) VALUES (?, ?, ?)",
            ("card-uid", "manual-tag", "manual"),
        )
        conn.commit()

        _write_card_tags(conn, {("card-uid", "removal")})
        _write_card_tags(conn, {("card-uid", "ramp")})  # re-run with a different tag set

        tagger_rows = {
            r[0] for r in conn.execute("SELECT tag FROM card_tags WHERE source = 'tagger_bulk'")
        }
        assert tagger_rows == {"ramp"}  # not {"removal", "ramp"} — replaced, not accumulated
        manual_rows = {
            r[0] for r in conn.execute("SELECT tag FROM card_tags WHERE source = 'manual'")
        }
        assert manual_rows == {"manual-tag"}  # untouched — delete is scoped by source


def test_extract_aliases_maps_normalized_alias_to_canonical_slug() -> None:
    registry = {
        "sweeper": _tag(
            "sweeper", "sweeper", aliases=["Wrath of God", "Boardwipe", "  Mass Removal  "]
        ),
    }
    assert extract_aliases(registry) == {
        "wrath of god": "sweeper",
        "boardwipe": "sweeper",
        "mass removal": "sweeper",
    }


def test_extract_aliases_skips_tags_without_a_slug() -> None:
    registry = {"broken": {"id": "broken", "slug": None, "aliases": ["whatever"]}}
    assert extract_aliases(registry) == {}


def test_write_tag_aliases_replaces_wholesale(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        _write_tag_aliases(conn, {"boardwipe": "sweeper", "wipe": "sweeper"})
        _write_tag_aliases(conn, {"ramp spell": "ramp"})  # re-run with a different alias set

        rows = dict(conn.execute("SELECT alias_normalized, tag FROM tag_aliases"))
        assert rows == {"ramp spell": "ramp"}  # replaced, not accumulated
