import json

import pytest

from autotrade.utils.atomic_io import atomic_write_json


def test_atomic_write_json_replaces_with_valid_payload(tmp_path):
    path = tmp_path / "overnight_state.json"
    path.write_text('{"old": true}', encoding="utf-8")

    atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}


def test_atomic_write_json_does_not_direct_write_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "overnight_state.json"
    path.write_text('{"old": true}', encoding="utf-8")

    def _fail_replace(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr("autotrade.utils.atomic_io.os.replace", _fail_replace)

    with pytest.raises(PermissionError):
        atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".overnight_state.json.*.tmp")) == []
