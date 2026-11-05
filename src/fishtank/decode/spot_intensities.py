from collections.abc import Callable

import numpy as np

_CHUNK_SIZE = 10000  # spots gathered per agg call; bounds memory use


def _circle_mask(x, y, radius, shape):
    """Create a circular mask."""
    y, x = np.ogrid[-y : shape[0] - y, -x : shape[1] - x]
    mask = x**2 + y**2 <= radius**2
    return mask


def _spot_intensity(img, x, y, z, radius, mask, agg=np.max):
    """Get intensity vector for a spot."""
    y_min, y_max = max(0, y - radius), min(img.shape[-2], y + radius + 1)
    x_min, x_max = max(0, x - radius), min(img.shape[-1], x + radius + 1)
    if y_max - y_min != mask.shape[-2] or x_max - x_min != mask.shape[-1]:
        if y > img.shape[-2] or y < 0 or x > img.shape[-1] or x < 0:
            return np.ones(img.shape[0]) * np.nan
        mask = _circle_mask(x - x_min, y - y_min, radius, (y_max - y_min, x_max - x_min))
    if z is None:
        intensity = agg(img[:, y_min:y_max, x_min:x_max][:, mask], axis=-1)
    else:
        if z >= img.shape[-3] or z < 0:
            return np.ones(img.shape[0]) * np.nan
        intensity = agg(img[:, z, y_min:y_max, x_min:x_max][:, mask], axis=-1)
    return intensity


def spot_intensities(
    img: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None, radius: int = 5, agg: Callable = np.max
) -> np.ndarray:
    """Get spot intensities.

    Parameters
    ----------
    img
        A (Y,X), (C,Y,X) or (C,Z,Y,X) image.
    x
        A numpy array of x coordinates.
    y
        A numpy array of y coordinates.
    z
        A numpy array of z coordinates. If None, the image is assumed to be 2D.
    radius
        The radius of the spot in pixels.
    agg
        The aggregation function to use. Called with an (N, C, P) array of
        disc pixels and ``axis=-1``.

    Returns
    -------
    intensities
        a N x C numpy array of intensities where N is the number of spots and C is the number of channels.
    """
    # Setup
    if z is None:
        if img.ndim == 4:
            raise ValueError("z must be specified for images with 4 dimensions")
        if img.ndim == 2:
            img = img[np.newaxis, ...]
    else:
        if img.ndim == 2:
            raise ValueError("z must be None for images with 2 dimensions")
        elif img.ndim == 3:
            img = img[np.newaxis, ...]
    x, y = np.asarray(x), np.asarray(y)
    z = None if z is None else np.asarray(z)
    intensities = np.zeros((len(x), img.shape[0]))
    circle_mask = _circle_mask(radius, radius, radius, (2 * radius + 1, 2 * radius + 1))
    # Spots whose full disc fits in the image are gathered and reduced in chunks
    inside = (y - radius >= 0) & (y + radius < img.shape[-2]) & (x - radius >= 0) & (x + radius < img.shape[-1])
    full = inside
    if z is not None:
        z_ok = (z >= 0) & (z < img.shape[-3])
        intensities[inside & ~z_ok] = np.nan
        full = inside & z_ok
    dy, dx = np.nonzero(circle_mask)
    dy, dx = dy - radius, dx - radius
    idx = np.flatnonzero(full)
    for start in range(0, len(idx), _CHUNK_SIZE):
        chunk = idx[start : start + _CHUNK_SIZE]
        yy = y[chunk, None] + dy
        xx = x[chunk, None] + dx
        if z is None:
            pixels = img[:, yy, xx]
        else:
            pixels = img[:, z[chunk, None], yy, xx]
        # (N, C, P) view with C fastest in memory, the layout the per-spot mask indexing gives,
        # so float sums accumulate in the same order
        intensities[chunk] = agg(pixels.transpose(1, 0, 2), axis=-1)
    # Spots at the image edge use a per-spot clipped mask
    for i in np.flatnonzero(~inside):
        intensities[i] = _spot_intensity(img, x[i], y[i], None if z is None else z[i], radius, circle_mask, agg)
    return intensities
