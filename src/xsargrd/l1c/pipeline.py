import logging
import xarray as xr
from xsargrd.l1c.ancillaries import append_ancillary_field

# --- logger ---
logger = logging.getLogger(__name__)


def enrich_l1b(
    l1b_fullpath: str,
    *,
    ancillary_list: dict | None = None,
) -> tuple[xr.Dataset, dict]:
    """
    Enrich a Level-1B dataset with ancillary geophysical fields.

    Parameters
    ----------
    l1b_fullpath : str
        Path to the L1B NetCDF file.
    ancillary_list : dict, optional
        Dictionary describing ancillary datasets to ingest.

    Returns
    -------
    ds : xr.Dataset
        Enriched L1B dataset (L1C content).
    flag_ancillaries : dict
        Dictionary indicating which ancillary fields were added.
    """
    logger.info(f"[Ancillary] Opening L1B dataset: {l1b_fullpath}")

    if ancillary_list is None:
        ancillary_list = {}

    ds = xr.open_dataset(l1b_fullpath)

    flag_ancillaries = {}

    for ancillary_name, ancillary_conf in ancillary_list.items():
        logger.info(f"[Ancillary] Processing: {ancillary_name}")

        ds, added = append_ancillary_field(ds, ancillary_conf)
        flag_ancillaries[ancillary_name] = added

        if added:
            logger.info(f"[Ancillary] {ancillary_name} successfully added")
        else:
            logger.warning(f"[Ancillary] {ancillary_name} not found or not added")

    return ds, flag_ancillaries