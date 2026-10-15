import logging

import geopandas as gpd
import pytest
import skimage as ski
from geopandas.testing import assert_geodataframe_equal

import fishtank as ft


@pytest.mark.slow
def test_cellpose_2d(caplog):
    parser = ft.scripts.cellpose_script.get_parser()
    args = parser.parse_args(
        [
            "--input",
            "./tests/data/merfish",
            "--output",
            "./tests/output/cellpose_2d",
            "--fov",
            "1",
            "--model",
            "cpsam",
            "--min_size",
            "100",
            "--z_slices",
            "3",
            "--filter",
            "unsharp_mask",
            "--channels",
            "DAPI",
            "--corrections",
            "./tests/data/corrections",
        ]
    )
    with caplog.at_level(logging.INFO):
        kwargs = vars(args)
        kwargs.pop("func")
        ft.scripts.cellpose(**kwargs)
    assert "Correcting illumination with tests/data/corrections" in caplog.text
    assert "Saving polygons to tests/output/cellpose_2d" in caplog.text


@pytest.mark.slow
def test_cellpose_3d(caplog):
    parser = ft.scripts.cellpose_script.get_parser()
    args = parser.parse_args(
        [
            "--input",
            "./tests/data/merfish",
            "--output",
            "./tests/output/cellpose_3d",
            "--fov",
            "1",
            "--model",
            "spsam",
            "--diameter",
            "20",
            "--downsample",
            "4",
            "--min_size",
            "500",
            "--filter",
            "deconwolf",
            "--channels",
            "DAPI",
            "--do_3D",
            "True",
        ]
    )
    with caplog.at_level(logging.INFO):
        kwargs = vars(args)
        kwargs.pop("func")
        ft.scripts.cellpose(**kwargs)
    assert "Applying deconwolf filter from fishtank" in caplog.text
    assert "Downsampling image by a factor of 4" in caplog.text
    assert "Saving polygons to tests/output/cellpose_3d" in caplog.text


class _ThresholdModel:
    """Stand-in for cellpose.models.CellposeModel: labels the first channel above its mean"""

    def __init__(self, pretrained_model=None, gpu=False):
        pass

    def eval(self, img, **kwargs):
        return (ski.measure.label(img[0] > img[0].mean()),)


def _run_cellpose(output, fov):
    ft.scripts.cellpose(
        input="./tests/data/merfish", fov=fov, channels=["DAPI", "DAPI"], output=output, downsample=4, do_3D=True, min_size=0
    )


def test_cellpose_batch_matches_single(tmp_path, monkeypatch):
    import cellpose.models

    monkeypatch.setattr(cellpose.models, "CellposeModel", _ThresholdModel)
    for fov in [0, 1]:
        _run_cellpose(tmp_path / "single", fov)
    _run_cellpose(tmp_path / "batch", [0, 1])
    for fov in [0, 1]:
        single = gpd.read_file(tmp_path / "single" / f"polygons_{fov}.json")
        batch = gpd.read_file(tmp_path / "batch" / f"polygons_{fov}.json")
        assert len(single) > 0
        assert_geodataframe_equal(batch, single)
    assert not list((tmp_path / "batch").glob("*.tmp"))


def test_cellpose_batch_skips_existing(tmp_path, monkeypatch, caplog):
    import cellpose.models

    monkeypatch.setattr(cellpose.models, "CellposeModel", _ThresholdModel)
    _run_cellpose(tmp_path, [0])
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _run_cellpose(tmp_path, [0, 1])
    assert "Skipping 1 FOVs with existing output: [0]" in caplog.text
    assert "[FOV 0] Loading image" not in caplog.text
    assert (tmp_path / "polygons_1.json").exists()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
