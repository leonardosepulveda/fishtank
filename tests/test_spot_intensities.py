import numpy as np
import pytest

import fishtank as ft
from fishtank.decode.spot_intensities import _circle_mask, _spot_intensity


def test_spot_intensities_3d(img):
    intensities = ft.decode.spot_intensities(img, x=[10, 20], y=[10, 20], z=[0, 1], radius=5)
    assert intensities.shape == (2, 7)
    assert intensities[0, 0] == 22401
    assert intensities.dtype == np.float64


def test_spot_intensities_2d(img):
    intensities = ft.decode.spot_intensities(img[:, 0], x=[10, 20], y=[10, 20], radius=5)
    assert intensities.shape == (2, 7)
    assert intensities[0, 0] == 22401
    assert intensities.dtype == np.float64


def _reference_spot_intensities(img, x, y, z=None, radius=5, agg=np.max):
    """The original per-spot loop, kept as the reference for exact equality."""
    if z is None:
        img = img[np.newaxis, ...] if img.ndim == 2 else img
    else:
        img = img[np.newaxis, ...] if img.ndim == 3 else img
    mask = _circle_mask(radius, radius, radius, (2 * radius + 1, 2 * radius + 1))
    out = np.zeros((len(x), img.shape[0]))
    for i in range(len(x)):
        out[i] = _spot_intensity(
            img, x[i], y[i], None if z is None else z[i], radius, mask, agg
        )
    return out


def _edge_spots(shape, radius, n=500, seed=0):
    """Random spots plus every edge case: on, near and past each border."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    x = list(rng.integers(0, nx, n))
    y = list(rng.integers(0, ny, n))
    for v in [-radius - 2, -1, 0, 1, radius - 1, radius, radius + 1]:
        for u in [v, nx - 1 - v, nx + v] if v < 0 else [v, nx - 1 - v, nx - v]:
            x += [u, u, u]
            y += [ny // 2, 0, ny]
    for v in [-radius - 2, -1, 0, 1, radius - 1, radius, radius + 1, ny - 1, ny, ny + 1]:
        x += [nx // 2]
        y += [v]
    return np.array(x), np.array(y)


@pytest.mark.parametrize("agg", [np.max, np.sum, np.mean])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("ndim,use_z", [(2, False), (3, False), (3, True), (4, True)])
def test_spot_intensities_matches_reference(agg, dtype, ndim, use_z):
    rng = np.random.default_rng(1)
    radius = 5
    shape = {2: (40, 50), 3: (3, 40, 50), 4: (3, 4, 40, 50)}[ndim]
    img = (rng.random(shape) * 60000).astype(dtype)
    x, y = _edge_spots(shape[-2:], radius)
    z = rng.integers(-1, shape[-3] + 1, len(x)) if use_z else None
    expected = _reference_spot_intensities(img, x, y, z, radius, agg)
    result = ft.decode.spot_intensities(img, x, y, z, radius, agg)
    assert np.isnan(expected).any()
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


if __name__ == "__main__":
    pytest.main(["-v", __file__])
