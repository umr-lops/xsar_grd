import sys
import logging
from pathlib import Path
import xsargrd
from xsargrd.l1c.pipeline import enrich_l1b
from xsargrd.l1c.tools import build_l1c_fileout, save_l1c_dataset

# --- version ---
version = xsargrd.__version__

# --- Configure logging ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,  
    format="%(message)s", 
)

# --- logger ---
logger = logging.getLogger(__name__)


def generate_l1c(
    fullpath_l1b: str,
    res: str,
    ancillary_list: dict,
    overwrite: bool = False,
    save: bool = True,
) -> tuple | None:
    """
    Generate Level-1C products from Level-1B NetCDF files.

    Parameters
    ----------
    fullpath_l1b : str
        Path to the L1B product directory.
    res : str
        Product resolution (e.g. '100m').
    ancillary_list : dict
        Dictionary describing ancillary datasets to ingest.
    overwrite : bool
        If True, overwrite existing L1C files.
    save : bool
        If True, save L1C product to NetCDF.

    Returns
    -------
    (xr.Dataset, dict) or None
        L1C dataset and ancillary flags if save=False or single file processed.
    """

    logger.info("[L1C] Starting L1C generation")
    logger.info(f"[IO] Ancillary inputs: {list(ancillary_list.keys())}")

    # --- Locate L1B files ---
    run_directory = Path(fullpath_l1b) / f"res{res}"
    files = sorted(run_directory.glob("*.nc"))

    logger.info(f"[IO] Searching L1B files in: {run_directory}")
    logger.info(f"[IO] Number of L1B files found: {len(files)}")

    if len(files) == 0:
        logger.warning("[ERROR] No L1B NetCDF files found. Aborting.")
        return None

    if len(files) > 1:
        logger.warning(
            "[WARNING] Multiple L1B files found. "
            "Processing them sequentially, but only one is usually expected."
        )

    last_ds = None
    last_flags = None

    # --- Loop on L1B files (robust but explicit) ---
    for l1b_fullpath in files:

        logger.info(f"[L1C] Processing L1B file: {l1b_fullpath.name}")

        # --- Build output path ---
        l1c_fullpath = build_l1c_fileout(str(l1b_fullpath))

        if Path(l1c_fullpath).exists() and not overwrite:
            logger.info(f"[IO] L1C already exists, skipping: {l1c_fullpath}")
            continue

        # --- Enrichment step ---
        logger.info("[L1C] Enriching L1B with ancillary data...")
        l1c_ds, flag_ancillaries_added = enrich_l1b(
            str(l1b_fullpath),
            ancillary_list=ancillary_list,
        )

        # --- Update attributes ---
        l1c_ds.attrs.update({
            "processing_level": "L1C",
            "processing_version": version,
            "source_l1b": str(l1b_fullpath),
            "ancillaries_added": str(flag_ancillaries_added),
        })

        # --- Saving ---
        if save:
            logger.info("[IO] Saving L1C NetCDF...")
            save_l1c_dataset(
                l1c_ds,
                l1c_fullpath,
                overwrite=overwrite,
            )

        last_ds = l1c_ds
        last_flags = flag_ancillaries_added

    logger.info("[L1C] Processing completed")

    return last_ds, last_flags