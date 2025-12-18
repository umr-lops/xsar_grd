from glob import glob
import logging
from shapely import wkt
from datetime import datetime
import xarray as xr
from xsargrd.l1c.raster_readers import resource_strftime, ecmwf_0100_1h, ww3_global_yearly_3h
from xsargrd.l1c.coloc import raster_cropping_in_polygon_bounding_box, coloc_tiles_from_l1bgroup_with_raster

# --- logger ---
logger = logging.getLogger(__name__)


def append_ancillary_field(
    ds: xr.Dataset,
    ancillary: dict,
) -> tuple[xr.Dataset, bool]:
    """
    Append one ancillary dataset to an L1B dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input L1B dataset.
    ancillary : dict
        Ancillary configuration dictionary.

    Returns
    -------
    ds : xr.Dataset
        Dataset with ancillary fields appended if available.
    added : bool
        True if ancillary fields were added.
    """
    sar_date = datetime.strptime(
        ds.attrs["start_date"].split(".")[0],
        "%Y-%m-%d %H:%M:%S",
    )

    closest_date, filename = resource_strftime(
        ancillary["pattern"],
        step=int(ancillary["step"]),
        date=sar_date,
    )

    matches = glob(filename)
    if len(matches) != 1:
        logger.debug(f"[Ancillary] No matching file for pattern: {filename}")
        return ds, False

    logger.info(f"[Ancillary] Using file: {matches[0]}")

    raster_ds = _load_ancillary_raster(ancillary, matches[0], closest_date)

    footprint = wkt.loads(ds.attrs["main_footprint"])
    raster_ds = raster_cropping_in_polygon_bounding_box(footprint, raster_ds)
    raster_ds.attrs["name"] = ancillary["name"]

    colocated = coloc_tiles_from_l1bgroup_with_raster(
        ds,
        raster_ds,
        apply_merging=False,
    )

    ds = xr.merge([ds, colocated])
    ds.attrs[f"{ancillary['name']}_pattern"] = filename

    return ds, True


def _load_ancillary_raster(ancillary: dict, filename: str, date: datetime) -> xr.Dataset:
    """
    Load an ancillary raster dataset based on its configuration.
    """
    name = ancillary["name"]

    if name == "ecmwf_0100_1h":
        return ecmwf_0100_1h(filename)

    if name == "ww3_global_yearly_3h":
        return ww3_global_yearly_3h(filename, date)

    raise NotImplementedError(f"Ancillary '{name}' not implemented")