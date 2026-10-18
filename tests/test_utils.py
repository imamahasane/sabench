from __future__ import annotations

import pandas as pd
import pytest

from sabench.utils.io import PathSecurityError, atomic_write_csv, safe_join
from sabench.utils.seeding import perm_seed


def test_perm_seed_is_deterministic():
    assert perm_seed(42, 3) == perm_seed(42, 3)


def test_perm_seed_differs_across_index():
    assert perm_seed(42, 1) != perm_seed(42, 2)


def test_perm_seed_differs_across_base():
    assert perm_seed(1, 0) != perm_seed(2, 0)


def test_perm_seed_rejects_negative_index():
    with pytest.raises(ValueError):
        perm_seed(0, -1)


def test_safe_join_normal_relative_path(tmp_path):
    (tmp_path / "sub").mkdir()
    result = safe_join(tmp_path, "sub/file.wav")
    assert result == (tmp_path / "sub" / "file.wav").resolve()


def test_safe_join_rejects_absolute_path(tmp_path):
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "/etc/passwd")


def test_safe_join_rejects_parent_traversal(tmp_path):
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "../../etc/passwd")


def test_atomic_write_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "out.csv"
    atomic_write_csv(df, path)
    assert path.exists()
    back = pd.read_csv(path)
    assert back["a"].tolist() == [1, 2]


def test_atomic_write_csv_leaves_no_tempfile_on_success(tmp_path):
    df = pd.DataFrame({"a": [1]})
    atomic_write_csv(df, tmp_path / "out.csv")
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
