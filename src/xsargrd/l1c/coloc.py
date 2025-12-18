import logging
import xarray as xr
import numpy as np

# --- logger ---
logger = logging.getLogger(__name__)


def raster_cropping_in_polygon_bounding_box(
    poly,
    raster_ds: xr.Dataset,
    *,
    enlarge: bool = True,
    step: int = 1,
) -> xr.Dataset:
    """
    Crop a raster dataset to the bounding box of a polygon.
    """
    lon1, lat1, lon2, lat2 = poly.exterior.bounds

    raster_ds = raster_ds.transpose("y", "x")

    for coord in ("x", "y"):
        if raster_ds[coord][0] > raster_ds[coord][-1]:
            raster_ds = raster_ds.reindex({coord: raster_ds[coord][::-1]})

    ilon = np.searchsorted(raster_ds.x.values, [lon1, lon2])
    ilat = np.searchsorted(raster_ds.y.values, [lat1, lat2])

    if enlarge:
        ilon = [ilon[0] - step, ilon[1] + step]
        ilat = [ilat[0] - step, ilat[1] + step]

    logger.debug("[Ancillary] Cropping raster to SAR footprint bounding box")

    return raster_ds.isel(x=slice(*ilon), y=slice(*ilat))


def coloc_tiles_from_l1bgroup_with_raster(
    ds: xr.Dataset,
    raster_ds: xr.Dataset,
    *,
    apply_merging: bool = True,
    drop_vars: tuple[str, ...] = ("forecast_hour",),
) -> xr.Dataset:
    """
    Colocate raster ancillary fields onto L1B tile centers.

    Raster fields defined on a regular lon/lat grid are interpolated
    onto the SAR tile center coordinates (center_longitude, center_latitude).

    Parameters
    ----------
    ds : xr.Dataset
        L1B dataset containing tile center coordinates.
    raster_ds : xr.Dataset
        Ancillary raster dataset cropped over the SAR footprint.
    apply_merging : bool, optional
        If True, merge colocated fields with the input dataset.
        If False, return only the colocated raster fields.
    drop_vars : tuple of str, optional
        Raster variables to ignore during colocation.

    Returns
    -------
    xr.Dataset
        Dataset containing colocated raster fields, optionally merged with ds.
    """

    logger.debug("[Coloc] Starting raster → SAR tile colocation")

    lons = ds.center_longitude
    lats = ds.center_latitude

    mapped_fields = []

    for var_name, raster_da in raster_ds.data_vars.items():

        if var_name in drop_vars:
            logger.debug(f"[Coloc] Skipping variable: {var_name}")
            continue

        logger.debug(f"[Coloc] Interpolating variable: {var_name}")

        # Ensure float type for interpolation safety
        raster_da = raster_da.astype(float)

        projected = (
            raster_da
            .interp(
                x=lons,
                y=lats,
                assume_sorted=False,
            )
            .drop_vars(("x", "y"), errors="ignore")
        )

        projected.name = var_name
        projected.attrs["source"] = raster_ds.attrs.get("name", "unknown")

        mapped_fields.append(projected)

    if not mapped_fields:
        logger.warning("[Coloc] No raster variables were colocated")
        return ds if apply_merging else xr.Dataset()

    raster_mapped = xr.merge(mapped_fields)

    if apply_merging:
        logger.debug("[Coloc] Merging colocated fields with L1B dataset")
        return xr.merge([ds, raster_mapped])

    logger.debug("[Coloc] Returning colocated raster fields only")
    return raster_mapped