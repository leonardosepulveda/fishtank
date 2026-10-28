import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import fishtank as ft

from ._utils import parse_index, parse_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("stitch")


def get_parser():
    """Get parser for stitch script"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", type=parse_path, default=None, help="Input file directory (band mode)")
    parser.add_argument("-o", "--output", type=parse_path, default="./stitch", help="Output directory")
    parser.add_argument("--positions", type=parse_path, required=True, help="TSV file with nominal FOV positions in microns (no header, columns: x, y)")
    parser.add_argument("--solve", action="store_true", help="Solve positions from the saved bands instead of extracting bands")
    parser.add_argument("--fovs", type=parse_index, default=None, help="FOVs to extract bands for (e.g. 1 or 1,2,3 or 1:20); default all")
    parser.add_argument("--channel", type=str, default="DAPI", help="Bit of the registration channel in the color usage file, imaged in the reference round")
    parser.add_argument("--file_pattern", type=str, default="{series}/Conv_zscan_{fov}.dax", help="Naming pattern for image files")
    parser.add_argument("--color_usage", type=str, default="{input}/color_usage.csv", help="Path to color usage file")
    parser.add_argument("--z_slices", type=parse_index, default=None, help="Z-slices to project (e.g. 1 or 1,2,3 or 1:20:5); default all")
    parser.add_argument("--overlap_px", type=int, default=345, help="Width in pixels of the overlap bands (nominal overlap plus margin)")
    parser.add_argument("--microscope", type=parse_path, default=None, help="YAML file of microscope settings (transpose, flip_horizontal, flip_vertical, micron_per_pixel) that override the image XML")
    parser.add_argument("--scale_factor", type=float, default=None, help="Microns per pixel; default is micron_per_pixel of --microscope (solve mode)")
    parser.set_defaults(func=stitch)
    return parser


def stitch(
    positions: str | Path,
    output: str | Path = "./stitch",
    input: str | Path | None = None,
    solve: bool = False,
    fovs: list[int] | None = None,
    channel: str = "DAPI",
    file_pattern: str = "{series}/Conv_zscan_{fov}.dax",
    color_usage: str = "{input}/color_usage.csv",
    z_slices: list[int] | None = None,
    overlap_px: int = 345,
    microscope: str | Path | None = None,
    scale_factor: float | None = None,
    **kwargs,
):
    """Stitch a FOV grid from the overlaps between neighbouring FOVs.

    Two steps, so that it scales to a SLURM array plus one job.

    Band mode (default): fishtank stitch -i input --positions positions.txt --fovs 5 -o out

    Reads the DAPI (or other ``channel``) image of each FOV, takes the maximum over z, and saves its four
    overlap bands to ``output/bands/bands_{fov}.npz``.

    Solve mode: fishtank stitch --solve --positions positions.txt -o out --microscope scope.yaml

    Registers every pair of neighbouring FOVs on the saved bands and solves the positions. Writes
    ``output/positions_stitched.txt`` (microns, no header, one row per FOV: the file format of ``--positions``),
    ``output/edges.csv`` and ``output/stitch_report.json``. Stops if the median Pearson correlation of the registered bands is below 0.3, which
    happens when the images are not oriented along the stage axes (check the microscope settings).

    Parameters
    ----------
    positions
        Path to a two-column (no header) TSV file with the nominal x and y position (microns) of each FOV.
    output
        Output directory.
    input
        Input file directory (band mode).
    solve
        Solve positions from the saved bands instead of extracting bands.
    fovs
        FOVs to extract bands for. Default is every FOV in ``positions``.
    channel
        Bit of the registration channel in the color usage file.
    file_pattern
        Naming pattern for image files.
    color_usage
        Path to color usage file.
    z_slices
        Z-slices to project. Default is all.
    overlap_px
        Width in pixels of the overlap bands.
    microscope
        YAML file of microscope settings that override the image XML (see ``fishtank.io.read_microscope``).
    scale_factor
        Microns per pixel. Default is ``micron_per_pixel`` of ``microscope``.
    """
    output = Path(output)
    nominal = pd.read_csv(positions, sep=None, engine="python", header=None, names=["x", "y"])
    nominal.index.name = "fov"
    if solve:
        _solve(nominal, output, microscope, scale_factor)
    else:
        if input is None:
            raise ValueError("input is required to extract bands")
        _extract(nominal, output, input, fovs, channel, file_pattern, color_usage, z_slices, overlap_px, microscope)


def _extract(nominal, output, input, fovs, channel, file_pattern, color_usage, z_slices, overlap_px, microscope):
    """Save the overlap bands of each FOV."""
    (output / "bands").mkdir(parents=True, exist_ok=True)
    if "{input}" in color_usage:
        color_usage = color_usage.format(input=input)
    channels = ft.io.read_color_usage(color_usage)
    channels = channels.set_index("bit").loc[[channel], :].reset_index()
    for fov in nominal.index if fovs is None else fovs:
        img, _ = ft.io.read_fov(
            input, fov, channels=channels, file_pattern=file_pattern, z_slices=z_slices, microscope=microscope
        )
        bands = ft.correct.stitching.overlap_bands(img.max(axis=0), overlap_px)
        np.savez(output / "bands" / f"bands_{fov:04d}.npz", **{_key(side): band for side, band in bands.items()})
        logger.info(f"[FOV {fov}] saved bands from an image of shape {img.shape}")


def _key(side):
    return side.replace("+", "p").replace("-", "m")


def _solve(nominal, output, microscope, scale_factor):
    """Register neighbouring FOVs on the saved bands and solve the positions."""
    stitching = ft.correct.stitching
    if scale_factor is None:
        if microscope is None:
            raise ValueError("scale_factor or microscope is required to solve positions")
        scale_factor = ft.io.read_microscope(microscope)["micron_per_pixel"]
    cache = {}

    def load_bands(fov):
        if fov not in cache:
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
            with np.load(output / "bands" / f"bands_{fov:04d}.npz") as z:
                cache[fov] = {side: z[_key(side)] for side in stitching.DIRECTIONS}
        return cache[fov]

    first = load_bands(nominal.index[0])
    frame_shape = (first["+x"].shape[0], first["+y"].shape[1])
    overlap_px = first["+x"].shape[1]
    edges = stitching.find_edges(nominal)
    logger.info(f"{len(nominal)} FOVs, {len(edges)} edges")
    edges = stitching.register_edges(edges, load_bands, frame_shape, overlap_px, scale_factor)
    keep = stitching.filter_edges(edges, min_threshold=scale_factor)
    if edges.pearson.median() < 0.3:
        raise RuntimeError(
            f"The overlap bands of neighbouring FOVs barely correlate (median Pearson {edges.pearson.median():.2f}). "
            "The images are probably not oriented along the stage axes: check transpose and flips in the "
            "microscope settings (see fishtank.io.read_microscope)."
        )
    solved, affine, info = stitching.solve_positions(edges[keep], nominal)
    errors = stitching.cross_validate(edges[keep], nominal)
    report = {
        "n_fov": len(nominal),
        "n_edges": len(edges),
        "n_rejected": int((~keep).sum()),
        "micron_per_pixel": scale_factor,
        "overlap_px": overlap_px,
        "affine": affine.tolist(),
        "heldout_median_um": {k: float(np.median(v)) for k, v in errors.items()},
        "heldout_p95_um": {k: float(np.percentile(v, 95)) for k, v in errors.items()},
        **info,
    }
    edges.assign(kept=keep).to_csv(output / "edges.csv", index=False)
    solved.loc[nominal.index, ["x", "y"]].to_csv(output / "positions_stitched.txt", sep="\t", header=False, index=False)
    (output / "stitch_report.json").write_text(json.dumps(report, indent=1, default=float))
    logger.info(f"Held-out median edge error (um): {report['heldout_median_um']}")
