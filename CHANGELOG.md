# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Support for reading `.zarr` movies via a new `read_zarr` function, wired into `read_img`/`read_fov`. Reads the Zhuang-lab HAL / `storm-control` Zarr format (a directory store with a single `(F, Y, X)` uint16 array named `data`) and returns frames in the same orientation as `.dax`.
- Add `--on_the_fly` flag to `detect-spots` for incremental/streaming analysis of experiments whose rounds are processed while still being acquired. Spots are detected once from the reference (and common-bit) series and their positions are frozen; series whose image files are not yet on disk are skipped instead of raising `FileNotFoundError`, and re-running the same command as more rounds arrive resumes from the existing `spots_{fov}.csv` / `channels_{fov}.csv`, quantifying only the newly-arrived series and appending their bit columns. Defaults to `False`, preserving the original all-rounds-present behavior.
- `fishtank.correct.stitching`: stitch a FOV grid from neighbour overlaps (phase correlation per edge, per-direction outlier rejection, displacement affine, least-squares positions with an affine prior, held-out validation). Ported from MERlin's `globalpositions`.
- Microscope settings file: `read_microscope` reads a YAML file with `transpose`, `flip_horizontal`, `flip_vertical` and `micron_per_pixel`, which override the image XML in `read_img`, `read_fov` and `read_mosaic`, and via `--microscope` in `detect-spots`, `cellpose` and `mosaic`. Use it when the XML's orientation or pixel size is wrong for a microscope.

### Changed

- `masks_to_polygons` traces each cell on its bounding box instead of scanning the whole volume per cell. Output is unchanged; about 9x faster on a 300-cell 3D volume.

### Fixes

- `determine_fov_format` raises `FileNotFoundError` instead of `ValueError` when no file matches, so a missing image is reported the same way whether or not the pattern fixes the FOV width.
- `decode-spots` drops spots with NaN intensities in the decoding bits before decoding (one NaN made every normalized intensity NaN), and raises a clear error when a decoding bit has no intensity column.
- `fishtank cellpose` with `gpu=True` raises `RuntimeError` when no working CUDA or MPS device is found, instead of silently running on the CPU while holding the GPU.
- Boolean script flags (`cellpose --gpu/--do_3D/--clear_border`, `aggregate-polygons --save_union/--flip_*`, `align-experiments --skip_fine`, `assign-spots --map_z`) used `type=bool`, so `False` turned them on. They now use `parse_bool`, which also raises on values that are not true/false, 1/0, yes/no, t/f or y/n instead of reading them as `False`.

## [0.3.7]

### Fixes

- Fix `IndexError` when reading ImageJ hyperstacks stored as a single physical TIFF page (e.g. contiguous multi-frame files from some microscope acquisitions). Falls back to `tif.asarray()` when `tif.pages` has fewer entries than the requested frame index.

## [0.1.1]

### Added

### Changed

-  Strategy is now optional for decode_spots

### Fixes

- aggregate_polygons now works when no FOVs overlap

## [0.1.0]

### Added

-   Support for Zhuang lab XML format
-   Switched to Cellpose>=4.0.0 to use SAM model

### Changed

-  "nuclei" and "cyto" models are no longer supported, use "cpsam" instead

### Fixes

## [0.0.1]

### Added

-   Basic tool, preprocessing and plotting functions
