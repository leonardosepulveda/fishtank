import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fishtank as ft
sys.path.insert(0, str(Path(__file__).parent))
from test_stitching import PIXEL, _edge_error, _grid  # noqa: E402


def _run(tmp_path, monkeypatch, tiles, nominal, scale_factor=PIXEL):
    pd.DataFrame({"series": ["cells"], "405": ["DAPI"]}).to_csv(tmp_path / "color_usage.csv", index=False)
    nominal.to_csv(tmp_path / "positions.txt", sep="\t", header=False, index=False)
    monkeypatch.setattr(ft.io, "read_fov", lambda path, fov, **kwargs: (np.stack([tiles[fov]] * 2), {}))
    out = tmp_path / "stitch"
    common = dict(positions=tmp_path / "positions.txt", output=out)
    ft.scripts.stitch(input=tmp_path, overlap_px=80, **common)
    ft.scripts.stitch(solve=True, scale_factor=scale_factor, **common)
    return out


def test_stitch_recovers_positions(tmp_path, monkeypatch):
    tiles, nominal, true = _grid()
    out = _run(tmp_path, monkeypatch, tiles, nominal)
    solved = pd.read_csv(out / "positions_stitched.txt", sep="\t", header=None, names=["x", "y"])
    assert len(solved) == len(nominal)
    assert _edge_error(solved, true) < 0.5 * PIXEL
    report = json.loads((out / "stitch_report.json").read_text())
    assert report["n_fov"] == len(nominal) and report["micron_per_pixel"] == PIXEL


def test_stitch_stops_on_wrong_orientation(tmp_path, monkeypatch):
    tiles, nominal, _ = _grid()
    tiles = {fov: img[::-1, ::-1] for fov, img in tiles.items()}
    with pytest.raises(RuntimeError, match="oriented"):
        _run(tmp_path, monkeypatch, tiles, nominal)
