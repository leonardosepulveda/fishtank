import numpy as np
import pytest

import fishtank as ft


def test_correct_illumination(img, channels, corr_path):
    corrected = ft.correct.illumination(img, colors=channels["color"], corr_path=corr_path)
    assert corrected.shape == img.shape
    assert corrected.dtype == img.dtype
    assert corrected.max() == 65313


def test_correct_illumination_shapes(img, corr_path):
    corrected = ft.correct.illumination(img[3], colors=[405], corr_path=corr_path)
    assert corrected.shape == img[3].shape
    corrected = ft.correct.illumination(img[3, 0], colors=405, corr_path=corr_path)
    assert corrected.shape == img[3, 0].shape


@pytest.fixture
def ffc_path(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "ffc_field_405.npy", rng.uniform(0.2, 1.5, (6, 8)))
    np.save(tmp_path / "ffc_field_650.npy", rng.uniform(0.2, 1.5, (6, 8)))
    np.save(tmp_path / "dark_frame.npy", np.full((6, 8), 100.0))
    return tmp_path


def test_correct_illumination_ffc(ffc_path):
    img = np.random.default_rng(1).integers(0, 65535, (2, 3, 6, 8)).astype(np.uint16)
    img[0, 0, 0, 0] = 0  # below the dark frame
    corrected = ft.correct.illumination(img, [405, 650], ffc_path, transpose=False, flip=False, ffc=True)
    field = np.load(ffc_path / "ffc_field_650.npy").astype(np.float32)
    expected = np.clip((img[1].astype(np.float32) - np.float32(100)) / field, 0, None)
    assert corrected.dtype == np.float32
    np.testing.assert_array_equal(corrected[1], expected)
    assert corrected[0, 0, 0, 0] == 0
    assert corrected.max() > 65535  # no wrap back to uint16


def test_correct_illumination_ffc_orientation_and_no_dark(ffc_path):
    (ffc_path / "dark_frame.npy").unlink()
    img = np.ones((8, 6), dtype=np.uint16)
    corrected = ft.correct.illumination(img, 405, ffc_path, ffc=True)  # default turn: transpose + flip
    field = np.load(ffc_path / "ffc_field_405.npy").astype(np.float32)
    np.testing.assert_array_equal(corrected, 1 / np.flip(field.T, axis=-2))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
