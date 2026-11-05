from pathlib import Path

import numpy as np

from logging import getLogger

def illumination(
    img: np.ndarray,
    colors: list | np.ndarray | int,
    corr_path: str | Path,
    transpose: bool = True,
    flip: bool = True,
    ffc: bool = False,
) -> np.ndarray:
    """Correct image illumination.

    Parameters
    ----------
    img
        A (Y,X), (C,Y,X) or (C,Z,Y,X) image.
    colors
        A list of channel colors. Must be the same length as the number of channels in the image.
    corr_path
        The path to the illumination correction file.
    transpose
        Whether to transpose the correction matrix.
    flip
        Whether to flip the correction matrix.
    ffc
        Apply a MERlin CreateFfc flat-field correction instead: ``(img - dark) / field``,
        with ``field`` read from ``ffc_field_<color>.npy`` and ``dark`` from
        ``dark_frame.npy`` (skipped if that file is absent), both in ``corr_path``.
        ``transpose`` and ``flip`` turn the field and dark frame as they do the default
        correction. CreateFfc stores them in MERlin's image frame, so pass
        ``transpose=False, flip=False`` when the image is read in that frame (e.g. with a
        microscope file that matches MERlin's).

    Returns
    -------
    corrected
        a numpy array of the same shape and dtype as the input image. With ``ffc``,
        a float32 array clipped at 0 instead, so nothing wraps or overflows.
    """
    if ffc:
        return _flat_field(img, colors, Path(corr_path), transpose, flip)
    # Setup
    dtype = img.dtype
    img = img.astype(np.uint32)
    corrected = np.zeros_like(img)
    unique_colors = np.unique(colors)
    shape = img.shape
    # Check inputs
    if isinstance(colors, int):
        colors = [colors]
        
    logger = getLogger("fishtank.correct.illumination")
    logger.info(f"Illumination correction called with img shape: {img.shape}, colors: {colors}, corr_path: {corr_path}, transpose: {transpose}, flip: {flip}")
    print("**", img.shape, colors)
    
    if len(colors) > 1 and len(colors) != shape[0]:
        raise ValueError("The length of colors must equal the number channels in the image.")
    # Correct illumination
    for color in unique_colors:
        color_corr = np.load(corr_path / f"illumination_correction_{color}_{shape[-2]}x{shape[-1]}.npy")
        if transpose:
            color_corr = color_corr.T
        if flip:
            color_corr = np.flip(color_corr, axis=-2)
        color_idx = np.where(colors == color)[0]
        if len(colors) > 1:
            corrected[color_idx, :] = img[color_idx, :] / color_corr
        else:
            corrected = img / color_corr
    return corrected.astype(dtype)


def _flat_field(img, colors, corr_path, transpose, flip):
    """(img - dark) / field per color, in float32, clipped at 0."""

    def load(name):
        arr = np.load(corr_path / name).astype(np.float32)
        if transpose:
            arr = arr.T
        if flip:
            arr = np.flip(arr, axis=-2)
        return arr

    colors = np.atleast_1d(colors)
    if len(colors) > 1 and len(colors) != img.shape[0]:
        raise ValueError("The length of colors must equal the number channels in the image.")
    dark = load("dark_frame.npy") if (corr_path / "dark_frame.npy").exists() else np.float32(0)
    corrected = np.empty(img.shape, dtype=np.float32)
    for color in np.unique(colors):
        field = load(f"ffc_field_{color}.npy")
        if len(colors) > 1:
            idx = np.where(colors == color)[0]
            corrected[idx] = (img[idx].astype(np.float32) - dark) / field
        else:
            corrected[:] = (img.astype(np.float32) - dark) / field
    return np.clip(corrected, 0, None, out=corrected)
