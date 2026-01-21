import os
import sys
from glob import glob
import logging
from pathlib import Path
import xarray as xr
from collections import defaultdict
from grdtiler import tiling_prod
import xsargrd
from xsargrd.l1b.spectra import compute_spectra
from xsargrd.l1b.cwave import compute_cwave_parameters
from xsargrd.l1b.scatt import compute_scatt_coeffs
from xsargrd.l1b.tools import (
    build_l1b_fileout,
    get_ocean_mask,
    remove_bright_targets,
    standardize_tile_coords,
)

# --- version ---
version = xsargrd.__version__

# --- Configure logging ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,  
    format="%(message)s", 
)
logger = logging.getLogger(__name__)


def generate_l1b(
        fullpath : str = None,
        dirout : str = None,
        res : str = None,
        tile_size : int = 17000,
        periodo_width : dict = {'sample': 2000., 'line': 2000.},
        periodo_overlap : dict = {'sample': 1000., 'line': 1000.},
        lowpass_width : dict = {'line': 4750., 'sample': 4750.},
        scatt_mode : str = None,
        norient : int = 8,
        save : bool = True,
        config_id : str = None,
        istest : bool = False,
)->None:
    """
    Generate Level 1B Ground Range Detected (GRD) products.

    Parameters
    ----------
    fullpath : str
        Full path to the SAR repository.
    dirout : str
        Directory where the output file will be saved.
    res : str
        Resolution of the product (e.g., '10m').
    tile_size : int
        Size of the tiles to be processed (default is 17000).
    periodo_width : dict
        Width of the periodogram to compute the cospectrum (default is {'sample': 2000., 'line': 2000.}).
    periodo_overlap : dict
        Overlap of the periodogram to compute the cospectrum (default is {'sample': 1000., 'line': 1000.}).
    lowpass_width : dict
        Width in meters for low-pass filtering during modulation computation,
        keyed by dimension names. Default is {'line': 4750., 'sample': 4750.}.
    scatt_mode : str
        Scattering transform mode. If None, scattering transform is not applied (default is None).
    norient : int
        Number of orientations for the scattering transform (default is 8).
    save : bool
        If True, save the output to a NetCDF file (default is True).
    istest : bool
        If True, save the output in the specified output directory without creating subdirectories (default is False).

    Returns
    -------
    None
    """
    # --- Step 1a: Tiling and select polarization ---
    logger.info("[Tile] Generating tiles from input product...")
    centering = True
    side = 'left'
    to_keep_var = ['digital_number',
                   'sigma0',
                   'nesz',
                   'incidence',
                   'land_mask',
                   'longitude',
                   'latitude',
                   'ground_heading',
                   'sampleSpacing',
                   'lineSpacing',
                   ]
    
    _, tiles = tiling_prod(
        path=fullpath,
        tile_size=tile_size,
        resolution=res,
        detrend=False,
        noverlap=0,
        centering=centering,
        side=side,  
        save=False,
        to_keep_var=to_keep_var,
    )

    logger.info(f"[Tile] Tiles generated.")

    # --- Step 1b: Get attrs & change of dimensions and add tile coordinate ---
    tiles_attrs = tiles.attrs
    line_attrs = tiles.line.attrs
    sample_attrs = tiles.sample.attrs
    pol_attrs = tiles.pol.attrs
    tiles = standardize_tile_coords(tiles)
    tiles_index = tiles.tile

    # --- Step 2a: Define which tile to process ---
    logger.info("[Tile] Checking ocean-only mask...")
    ocean_only = get_ocean_mask(tiles)
    tiles = tiles.where(ocean_only, drop=True).load()
    logger.info(f"[Tile] Ocean-only mask applied. {tiles.tile.size} tiles remain after filtering.")
    logger.info(f"[Tile] Loaded {tiles.tile.size} tiles.")
    to_merge = [ocean_only]

    # --- Step 2b: Remove bright targets ---
    logger.info("[Tile] Removing bright targets from tiles...")
    tiles, bright_targets_mask, bright_targets_hist = remove_bright_targets(tiles)
    logger.info("[Tile] Bright targets removed.")
    to_merge.extend([bright_targets_mask, bright_targets_hist])

    # --- Step 3: Compute Spectra & Cwave ---
    logger.info("[Spectra] Computing periodograms...")
    spec = compute_spectra(
        tiles,
        periodo_width=periodo_width,
        periodo_overlap=periodo_overlap,
        lowpass_width=lowpass_width
    )
    logger.info("[Spectra] Computing C-wave parameters...")
    cwave = compute_cwave_parameters(spec.spectra)
    to_merge.extend([spec, cwave.T])

    # --- Step 4: Scattering coefficients ---
    if scatt_mode is not None:
        logger.info("[Scatt] Computing Scattering coefficients...")
        scatt_data = compute_scatt_coeffs(
            tiles,
            var_name='sigma0',
            res=res,
            tile_size=tile_size,
            #scatt_mode=scatt_mode,
            norient=norient,
        )
        to_merge.append(scatt_data)
        J = scatt_data.j1.size
        L = scatt_data.l1.size
    else:
        J = None
        L = None

    # --- Step 5: Merging ---
    logger.info("[Merge] Combining all datasets...")
    to_merge = [da.reindex(tile=tiles_index) for da in to_merge]
    out = xr.merge(to_merge)

    # --- Step 6: Add & update attributes ---
    out.coords["tile"].attrs.update({
        "long_name":   "Tile index",
        "description": "Index of the tile in the dataset",
    })
    out.coords["pol"].attrs.update(pol_attrs)
    out.coords["line"].attrs.update(line_attrs)
    out.coords["sample"].attrs.update(sample_attrs)
    out.attrs.update(tiles_attrs)
    out.attrs.update({'main_footprint':str(out.attrs['main_footprint'])})
    out.attrs['processing_version'] = version
    out.attrs['processing_level'] = 'L1B'
    # TODO add config_id in attributes 
       
    # --- Step 7: Saving ---
    if save:
        logger.info("[IO] Saving dataset to NetCDF...")
        pathout, fileout = build_l1b_fileout(
            fullpath=fullpath,
            dirout=dirout,
            res=res,
            config_id=config_id,
            version=version
        )

        if istest:
            out.to_netcdf(os.path.join(dirout, fileout))
            logger.info(f"[IO] Test file saved at: {os.path.join(dirout, fileout)}")
        else:
            # Build the output directory path (ensures consistent handling)
            pathout = Path(pathout)             # directory
            file_path = pathout / fileout       # final file path

            # Create output directory if it doesn't exist (equivalent to `mkdir -p`)
            pathout.mkdir(parents=True, exist_ok=True)

            # Remove existing file to avoid overwriting corrupted or partial outputs
            if file_path.exists():
                logger.warning(f"[IO] Existing output removed: {file_path}")
                file_path.unlink()  # safely delete the old file

            # Write NetCDF file with error handling
            try:
                out.to_netcdf(file_path)
                logger.info(f"[IO] File written to: {file_path}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to write NetCDF to {file_path}: {e}")
                raise  # re-raise to fail fast if needed

    else:
        logger.info("[IO] Returning dataset (not saved).")
        return out


