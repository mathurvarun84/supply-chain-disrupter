import os
from pathlib import Path

from src.utils.etl_loader import replace_database_file


def test_replace_database_file_retries_on_windows_lock(tmp_path, monkeypatch):
    source = tmp_path / "incoming.db"
    source.write_bytes(b"new-data")

    target = tmp_path / "current.db"
    target.write_bytes(b"old-data")

    attempts = {"count": 0}
    real_replace = os.replace

    def flaky_replace(src_path: Path, dst_path: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            err = PermissionError("[WinError 32] The process cannot access the file")
            err.winerror = 32
            raise err
        real_replace(src_path, dst_path)

    monkeypatch.setattr("src.utils.etl_loader.os.replace", flaky_replace)

    replace_database_file(source, target)

    assert target.exists()
    assert target.read_bytes() == b"new-data"
    assert attempts["count"] == 3
