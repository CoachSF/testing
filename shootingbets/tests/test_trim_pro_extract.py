"""Tests for scripts/trim_pro_extract.py - throw-away folders in tmp_path only."""
import importlib.util
import json

import pytest

from conftest import ROOT

_spec = importlib.util.spec_from_file_location("trim_pro_extract", ROOT / "scripts" / "trim_pro_extract.py")
trim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trim)


def make_pro(base, name, layout):
    """layout: {date: [event ids]}; every event folder gets two market files; root files and a __pycache__ are added."""
    pro = base / name
    (pro / "__pycache__").mkdir(parents=True)
    (pro / "__pycache__" / "x.pyc").write_bytes(b"pyc")
    (pro / "manifest_2026-10.jsonl").write_text("{}\n")
    (pro / "_done_2026-10.json").write_text("{}")
    for day, events in layout.items():
        (pro / day).mkdir()
        (pro / day / "stray.txt").write_text("a plain file in a date dir")
        for ev in events:
            (pro / day / ev).mkdir()
            for m in ("1.1", "1.2"):
                (pro / day / ev / f"{m}.bz2").write_bytes(b"x" * 100)
    return pro


def csv_of(tmp_path, ids, header="match_event_id"):
    p = tmp_path / "enriched.csv"
    p.write_text(header + "\n" + "\n".join(ids) + "\n")
    return str(p)


@pytest.fixture
def two(tmp_path):
    a = make_pro(tmp_path, "pro_a", {"2026-10-01": ["111", "222"], "2026-10-02": ["333"], "2026-10-03": ["444"]})
    b = make_pro(tmp_path, "pro_b", {"2026-10-01": ["555", "111"], "2026-10-04": ["666"]})
    return a, b


def tree(*pros):
    return sorted(str(p.relative_to(pro.parent)) for pro in pros for p in pro.rglob("*"))


def test_dry_run_deletes_nothing_and_reports(two, tmp_path, capsys):
    before = tree(*two)
    assert trim.main(["--enriched", csv_of(tmp_path, ["111", "666"]), "--pro-dir", str(two[0]), str(two[1])]) == 0
    assert tree(*two) == before
    out = capsys.readouterr().out
    assert "DRY RUN - nothing deleted (add --commit)" in out and "keep 1 events" in out and "delete 3 events" in out
    assert not (two[0] / "_trimmed.json").exists()


def test_commit_deletes_exactly_the_unkept_events_across_two_dirs(two, tmp_path, capsys):
    a, b = two
    assert trim.main(["--enriched", csv_of(tmp_path, ["111", "666"]), "--pro-dir", f"{a}=Soccer", f"{b}=Tennis", "--commit"]) == 0
    assert "VERIFY remaining events: 3 (all kept)" in capsys.readouterr().out
    # kept: pro_a 111 (+ its files), pro_b 111 and 666; everything else is gone, emptied date dirs are removed
    assert sorted(p.name for p in (a / "2026-10-01").iterdir()) == ["111", "stray.txt"]
    assert sorted(p.name for p in (b / "2026-10-01").iterdir()) == ["111", "stray.txt"]
    assert (a / "2026-10-01" / "111" / "1.1.bz2").is_file() and (b / "2026-10-04" / "666" / "1.2.bz2").is_file()
    # a date dir holding only stray.txt is not "empty": the plain file stays and so does the dir
    assert not (a / "2026-10-02" / "333").exists() and (a / "2026-10-02" / "stray.txt").is_file()
    assert not (a / "2026-10-03" / "444").exists() and not (b / "2026-10-01" / "555").exists()
    for pro in (a, b):
        assert (pro / "manifest_2026-10.jsonl").is_file() and (pro / "_done_2026-10.json").is_file()
        assert (pro / "__pycache__" / "x.pyc").is_file()
    t = json.loads((a / "_trimmed.json").read_text())
    assert t["kept_event_ids"] == ["111"] and t["deleted_events"] == 3 and t["deleted_bytes"] == 3 * 200
    assert "manifests still list every market" in t["note"] and t["trimmed_utc"].endswith("+00:00")
    assert json.loads((b / "_trimmed.json").read_text())["kept_event_ids"] == ["111", "666"]
    assert json.loads((b / "_trimmed.json").read_text())["deleted_events"] == 1


def test_date_dir_left_empty_is_removed(tmp_path):
    pro = make_pro(tmp_path, "pro", {"2026-10-01": ["111"], "2026-10-02": ["222"]})
    (pro / "2026-10-02" / "stray.txt").unlink()
    assert trim.main(["--enriched", csv_of(tmp_path, ["111"]), "--pro-dir", str(pro), "--commit"]) == 0
    assert not (pro / "2026-10-02").exists() and (pro / "2026-10-01" / "111").is_dir()


def test_kept_id_missing_on_disk_aborts_with_3_and_deletes_nothing(two, tmp_path, capsys):
    before = tree(*two)
    assert trim.main(["--enriched", csv_of(tmp_path, ["111", "999"]), "--pro-dir", str(two[0]), str(two[1]), "--commit"]) == 3
    assert tree(*two) == before
    assert "999" in capsys.readouterr().out


def test_bad_input_is_exit_2_and_deletes_nothing(two, tmp_path):
    before = tree(*two)
    pro = [str(two[0]), str(two[1])]
    assert trim.main(["--enriched", csv_of(tmp_path, []), "--pro-dir", *pro, "--commit"]) == 2                 # empty keep set
    assert trim.main(["--enriched", csv_of(tmp_path, ["", ""]), "--pro-dir", *pro, "--commit"]) == 2          # only blanks
    assert trim.main(["--enriched", csv_of(tmp_path, ["111"], header="other"), "--pro-dir", *pro, "--commit"]) == 2
    assert trim.main(["--enriched", str(tmp_path / "nope.csv"), "--pro-dir", *pro, "--commit"]) == 2
    assert trim.main(["--enriched", csv_of(tmp_path, ["111"]), "--pro-dir", pro[0], str(tmp_path / "nodir"), "--commit"]) == 2
    assert tree(*two) == before


def test_float_rendered_id_keeps_the_integer_event_dir(tmp_path):
    pro = make_pro(tmp_path, "pro", {"2026-10-01": ["123", "1230", "456"]})
    assert trim.main(["--enriched", csv_of(tmp_path, ["123.0", "", " 1230 "]), "--pro-dir", f"{pro}=Soccer", "--commit"]) == 0
    assert sorted(p.name for p in (pro / "2026-10-01").iterdir() if p.is_dir()) == ["123", "1230"]


def test_verify_exit_4_when_an_unkept_event_survives_and_rmtree_errors_propagate(two, tmp_path, monkeypatch, capsys):
    args = ["--enriched", csv_of(tmp_path, ["111"]), "--pro-dir", str(two[0]), str(two[1]), "--commit"]
    monkeypatch.setattr(trim.shutil, "rmtree", lambda p: None)                    # a delete that silently did nothing
    assert trim.main(args) == 4 and "VERIFY FAILED" in capsys.readouterr().out
    def boom(p):
        raise PermissionError(str(p))
    monkeypatch.setattr(trim.shutil, "rmtree", boom)                              # no ignore_errors: the error is not swallowed
    with pytest.raises(PermissionError):
        trim.main(args)
