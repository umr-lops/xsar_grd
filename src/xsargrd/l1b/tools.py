import os
import logging
from typing import Tuple
from datetime import datetime
import xarray as xr
import numpy as np
from tqdm import tqdm
from xsarslc.processing.xspectra import get_bright_target_mask

logger = logging.getLogger(__name__)

def build_l1b_fileout(
    fullpath: str,
    dirout: str,
    res: str,
    *,
    config_id: str | None = None,
    version: str | None = None,
    level: str = "l1b",
) -> Tuple[str, str]:
    """
    Build output directory and filename for L1B products.

    If no config_id is provided, a generic identifier ('a00') is used.
    """

    safename = os.path.basename(fullpath)
    parts = safename.split("_")

    # --- Resolve configuration identifier ---
    cfg_id = config_id.lower() if config_id is not None else "a00"

    # --- Mission handling ---
    mission = parts[0]
    if "S1" in mission:
        date_str = parts[4][:8]
        mode = parts[1].lower()
    elif "RCM" in mission:
        date_str = parts[5]
        mode = parts[4].lower()
    elif "RS2" in mission:
        date_str = parts[5]
        mode = parts[4].lower()
    else:
        raise ValueError(f"Unsupported mission type: {mission}")

    year = date_str[:4]
    doy = f"{datetime.strptime(date_str, '%Y%m%d').timetuple().tm_yday:03d}"

    # --- Filename prefix ---
    prefix = os.path.splitext(safename.lower())[0]

    # --- Filename ---
    file_out = f"{prefix}_{cfg_id}"

    # if version is not None:
    #     file_out += f"_v{version.replace('.', '')}"

    file_out += ".nc"

    # --- Output directory ---
    path_out = os.path.join(
        dirout,
        mission.lower(),
        mode,
        level.lower(),
        year,
        doy,
        safename,
        f"res{res}",
    )

    return path_out, file_out


def is_ocean_only(tile):
    """
    Returns True if the tile is only ocean (no land).

    Parameters
    ----------
    tile : xr.DataArray
        Input tile DataArray containing a land mask.

    Returns
    -------
    bool
        True if the tile is only ocean, False otherwise.
    """
    return not (tile.land_mask == True).any().values

def get_ocean_mask(tiles):
    indexes = tiles.tile
    ocean_only = [is_ocean_only(tiles.sel(tile=i)) for i in tqdm(indexes, total=indexes.size, desc="Create ocean-only mask", unit="tile")]
    return xr.DataArray(
        ocean_only,
        dims="tile",
        name="ocean_only",
        attrs={
            "long_name": "Tile is fully ocean",
            "flag_meanings": "True=Ocean False=Land"
        },
    )


def check_tile_mask(tile_mask, nt):
    if isinstance(tile_mask, xr.DataArray):
        # check if tile_mask is a DataArray of booleans
        if not np.issubdtype(tile_mask.dtype, np.bool_):
            raise ValueError(f'tile_mask is a DataArray containing {tile_mask.dtype}. Must be a DataArray of booleans.')
        # check if tile_mask has dimension along tile
        if 'tile' not in tile_mask.dims:
            raise ValueError(f'tile_mask must have a dimension along "tile".')
        # check if tile_mask length is similar to number of tiles (nt)
        if tile_mask.sizes['tile'] != nt:
            raise ValueError(f'tile_mask length ({tile_mask.sizes["tile"]}) must match number of tiles ({nt}).')
        tile_mask.attrs['comment'] = 'Used as tile_mask.'
    else:
        raise ValueError(f'Unsupported tile_mask type: {type(tile_mask)}. Must be a DataArray of booleans.')


def fill_bright_targets(da: xr.DataArray, bright_mask: xr.DataArray) -> xr.DataArray:
    """
    Fill bright targets in a DataArray with the mean value of non-bright pixels.
    Fully supports the presence or absence of a 'pol' dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array, potentially with a 'pol' dimension.
    bright_mask : xr.DataArray
        Boolean mask indicating bright targets (True = bright pixel).
        May or may not include a 'pol' dimension.

    Returns
    -------
    xr.DataArray
        Same array with bright pixels replaced by the mean of non-bright pixels,
        computed per polarization if applicable.
    """

    # ------------------------------------------------------------------
    # Ensure the mask is broadcastable to the DataArray
    # ------------------------------------------------------------------
    try:
        mask, _ = xr.broadcast(bright_mask, da)
    except Exception:
        # Fallback: manually expand dims if 'pol' is missing
        if "pol" in da.dims and "pol" not in bright_mask.dims:
            mask = bright_mask.expand_dims({"pol": da["pol"]})
            mask = mask.transpose(*da.dims)
        else:
            raise  # Mask shape truly incompatible

    # ------------------------------------------------------------------
    # Compute mean of non-bright pixels
    # If 'pol' exists → mean computed per polarization
    # If no 'pol' → mean computed globally
    # ------------------------------------------------------------------
    mean_da = da.where(~mask).mean(skipna=True, dim=[d for d in da.dims if d != "pol"])

    # If mean_da is missing 'pol' but da has it, broadcast it
    if "pol" in da.dims and "pol" not in mean_da.dims:
        mean_da = mean_da.expand_dims({"pol": da["pol"]})
        mean_da = mean_da.transpose(*da.dims)

    # ------------------------------------------------------------------
    # Replace bright pixels by the corresponding mean
    # ------------------------------------------------------------------
    return da.where(~mask, mean_da)


def remove_bright_targets(
        tiles : xr.Dataset,
        varnames : list = ['digital_number', 'sigma0', 'nesz'],
        targetsize : dict = {'line':10, 'sample':10},
        guardsize : dict = {'line':350, 'sample':350},
        cluttersize : dict = {'line':1000, 'sample':1000},
        nstddev : float = 10,
        nstddev_neigh : float = 3.5,
        itermax : int = 10
):
    """
    Remove bright targets from each tile in the dataset.

    Parameters
    ----------
    tiles : xr.Dataset
        Input Dataset containing the tiles.
    varnames : list
        List of variable names to process for bright target removal.
    targetsize : dict
        Size of the bright target in 'line' and 'sample' dimensions.
    guardsize : dict
        Size of the guard window around the target.
    cluttersize : dict
        Size of the clutter window.
    nstddev : float
        Threshold for bright target detection.
    nstddev_neigh : float
        Threshold for bright target neighborhood detection.
    itermax : int
        Maximum number of iterations to enhance detection.

    Returns
    -------
    tiles : xr.Dataset
        Updated tiles with bright targets removed.
    bright_mask_all : xr.DataArray
        Bright target masks (per tile and per polarization).
    bright_hist_all : xr.DataArray
        Bright target histograms (per tile and per polarization).
    """

    bright_mask_all = []
    bright_hist_all = []

    indexes = tiles.tile.values

    for i in tqdm(indexes, total=indexes.size, desc="Bright target detection & removal", unit="tile"):
        # Select current tile
        tile = tiles.sel(tile=i)

        # Extract spatial spacing
        spacing = {
            "sample": tile.sampleSpacing.item(),
            "line": tile.lineSpacing.item()
        }

        # Compute NRCS 
        nrcs = np.sqrt(tile.sigma0)

        # Compute bright mask and histogram (per polarization)
        bright_mask, bright_hist = get_bright_target_mask(
            nrcs,
            targetsize, guardsize, cluttersize,
            spacing,
            nstddev=nstddev,
            nstddev_neigh=nstddev_neigh,
            itermax=itermax
        )

        # Remove temporary coordinates if they exist
        bright_mask = bright_mask.drop_vars(
            [v for v in ["__line", "__sample"] if v in bright_mask.coords]
        )

        # Store per-polarization mask and histogram for this tile
        bright_mask_all.append(bright_mask.expand_dims({"tile": [int(i)]}))
        bright_hist_all.append(bright_hist.expand_dims({"tile": [int(i)]}))

        # --- UNION OF POLARIZATIONS ---
        # A pixel is considered bright if it is bright in at least one polarization
        if "pol" in bright_mask.dims:
            mask_union = bright_mask.any(dim="pol")
        else:
            mask_union = bright_mask

        # Apply the union mask to each variable to be cleaned
        for varname in varnames:
            var = tile[varname]

            # If variable has polarization dimension but mask_union does not,
            # expand mask_union to match the variable's pol dimension
            if "pol" in var.dims and "pol" not in mask_union.dims:
                mask_apply = mask_union.expand_dims({"pol": var["pol"]})
                mask_apply = mask_apply.transpose(*var.dims)

            else:
                # Otherwise broadcast mask and variable so they match
                try:
                    mask_apply, _ = xr.broadcast(mask_union, var)
                except Exception:
                    # Fallback: try to align dims manually
                    if set(mask_union.dims).issubset(set(var.dims)):
                        mask_apply = mask_union.transpose(
                            *[d for d in var.dims if d in mask_union.dims]
                        )
                        mask_apply, _ = xr.broadcast(mask_apply, var)
                    else:
                        # Last fallback: use mask_union as is (will error if incompatible)
                        mask_apply = mask_union

            # Replace bright target pixels using the union mask
            tile[varname] = fill_bright_targets(var, mask_apply)

        # Update tiles dataset with the cleaned tile
        tiles.loc[dict(tile=i)] = tile

    # Concatenate results over tile dimension
    bright_mask_all = xr.concat(bright_mask_all, dim="tile")
    bright_hist_all = xr.concat(bright_hist_all, dim="tile")

    # Set attributes and variable names
    bright_mask_all = (
        bright_mask_all
        .drop_attrs(deep=True)
        .assign_attrs({"long_name": "bright targets mask (per polarization)"})
    )
    bright_mask_all.name = "bt_mask"
    bright_hist_all.name = "bt_hist"

    return tiles, bright_mask_all, bright_hist_all


def standardize_tile_coords(tiles: xr.Dataset) -> xr.Dataset:
    """
    Standardize a tiles dataset to use canonical coordinates/dimensions.

    Parameters
    ----------
    tiles : xr.Dataset
        Input dataset containing a 'tile' dimension and 2D tile grids with
        line/sample dimensions (either 'line'/'sample' or 'tile_line'/'tile_sample').

    Returns
    -------
    xr.Dataset
        A (view-like) dataset with:
          - coord 'tile' = 0..nt-1 (attrs: long_name/description/units),
          - dims 'line' and 'sample' as 0-based integer indices,
    """
    ds = tiles.copy(deep=False)

    # Validation
    if "tile" not in ds.dims:
        raise ValueError("Missing required 'tile' dimension.")

    nt = ds.sizes["tile"]
    nl = ds.sizes["tile_line"]
    ns = ds.sizes["tile_sample"]    

    # Rename original line/sample dims to temporary names
    ds = ds.rename({"line": "__line", "sample": "__sample"})

    # Create canonical coordinates:
    ds = ds.assign_coords(
        tile=("tile", np.arange(nt)),
        line=("tile_line", np.arange(nl)),
        sample=("tile_sample", np.arange(ns)),
    )

    # Swap dims to use 'line'/'sample' as primary dims
    ds = ds.swap_dims({"tile_line": "line", "tile_sample": "sample"})
   
    return ds

def sanitize_attrs(ds):
    _ds = ds.copy()
    def fix(attrs):
        for k, v in list(attrs.items()):
            if isinstance(v, bool):
                attrs[k] = str(v)
    fix(_ds.attrs)
    return _ds