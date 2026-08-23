"""Schema creation is complete and idempotent, so re-running any ingest is safe."""

from pathlib import Path

from edhcut.db import TABLE_NAMES, connect


def test_schema_created_in_temp_file(tmp_path: Path) -> None:
    db_path = tmp_path / "edhcut.db"

    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

    assert db_path.exists()
    table_names = {row[0] for row in rows}
    for table in TABLE_NAMES:
        assert table in table_names


def test_schema_creation_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "edhcut.db"

    with connect(db_path):
        pass
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

    table_names = {row[0] for row in rows}
    for table in TABLE_NAMES:
        assert table in table_names
