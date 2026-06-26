import numpy as np
import xarray as xr
from shapely import wkt
from shapely import contains_xy
import logging

logger = logging.getLogger(__name__)


def extract_wind_stats_per_tile(wind_ds, wave_ds):
    """
    For each tile in wave_ds, extract the wind grid points falling inside
    the tile's polygon (WKT) and compute mean, std, min, max, and count.
    Results are added as new variables to wave_ds, aligned on the 'tile' dimension.
    """
    # Drop the residual time dimension if present (wind_ds has a single timestep)
    if 'time' in wind_ds.sizes:
        wind_ds = wind_ds.squeeze('time', drop=True)

    # Apply the quality flag, acceptable when qc >= 3:
    qc = wind_ds['quality_level']
    wind_da = wind_ds['wind_speed'].where(qc >= 3)

    # Keep the original wind_speed attributes as a base for the new stat variables
    base_attrs = wind_ds['wind_speed'].attrs

    # Check coordinate ordering once (assumed constant for the whole grid)
    lat_incr = bool(wind_ds['lat'][0] < wind_ds['lat'][-1])
    lon_incr = bool(wind_ds['lon'][0] < wind_ds['lon'][-1])

    n_tiles = wave_ds.sizes['tile']
    means = np.full(n_tiles, np.nan)
    stds = np.full(n_tiles, np.nan)
    mins = np.full(n_tiles, np.nan)
    maxs = np.full(n_tiles, np.nan)
    meds = np.full(n_tiles, np.nan)
    counts = np.zeros(n_tiles, dtype=int)

    for i in range(n_tiles):
        polygon_str = wave_ds['tile_footprint'].values[i]

        # Skip missing or empty footprints
        if polygon_str is None:
            continue
        if isinstance(polygon_str, float) and np.isnan(polygon_str):
            continue
        if isinstance(polygon_str, str) and polygon_str.strip() == "":
            logger.warning(f"Tile {i}: empty tile_footprint, skipping.")
            continue

        try:
            polygon = wkt.loads(polygon_str)
        except Exception as e:
            logger.warning(f"Tile {i}: failed to parse tile_footprint WKT ({e!r}), skipping.")
            continue

        minx, miny, maxx, maxy = polygon.bounds

        # Subset on bounding box first (fast, vectorized), accounting for coordinate order
        if lat_incr and lon_incr:
            wind_sub = wind_da.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
        elif (not lat_incr) and lon_incr:
            wind_sub = wind_da.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))
        elif lat_incr and (not lon_incr):
            wind_sub = wind_da.sel(lon=slice(maxx, minx), lat=slice(miny, maxy))
        else:
            wind_sub = wind_da.sel(lon=slice(maxx, minx), lat=slice(maxy, miny))

        # Skip if the tile falls outside the wind grid domain
        if wind_sub.sizes.get('lon', 0) == 0 or wind_sub.sizes.get('lat', 0) == 0:
            continue

        lon_sub = wind_sub['lon'].values
        lat_sub = wind_sub['lat'].values
        lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)

        # Vectorized point-in-polygon test (no Python-level loop over points)
        mask = contains_xy(polygon, lon2d, lat2d)

        if not mask.any():
            continue

        values = wind_sub.values[mask]
        values = values[~np.isnan(values)]

        if values.size == 0:
            continue

        means[i] = values.mean()
        stds[i] = values.std()
        mins[i] = values.min()
        maxs[i] = values.max()
        meds[i] = np.median(values)
        counts[i] = values.size

    # Attach results to wave_ds, aligned on the 'tile' dimension
    wave_ds_out = wave_ds.copy()

    quality_note = "Computed from wind_speed points with quality_level >= 3 falling inside the tile footprint."

    wave_ds_out['SAR_wind_speed_mean'] = ('tile', means, {
        'long_name': f"Mean {base_attrs.get('long_name', 'wind speed')}",
        'units': base_attrs.get('units', 'm/s'),
        'description': base_attrs.get('description', ''),
        'comment': f"Mean wind speed per SAR tile. {quality_note}",
    })

    wave_ds_out['SAR_wind_speed_std'] = ('tile', stds, {
        'long_name': f"Standard deviation of {base_attrs.get('long_name', 'wind speed')}",
        'units': base_attrs.get('units', 'm/s'),
        'description': base_attrs.get('description', ''),
        'comment': f"Standard deviation of wind speed per SAR tile. {quality_note}",
    })

    wave_ds_out['SAR_wind_speed_min'] = ('tile', mins, {
        'long_name': f"Minimum {base_attrs.get('long_name', 'wind speed')}",
        'units': base_attrs.get('units', 'm/s'),
        'description': base_attrs.get('description', ''),
        'comment': f"Minimum wind speed per SAR tile. {quality_note}",
    })

    wave_ds_out['SAR_wind_speed_max'] = ('tile', maxs, {
        'long_name': f"Maximum {base_attrs.get('long_name', 'wind speed')}",
        'units': base_attrs.get('units', 'm/s'),
        'description': base_attrs.get('description', ''),
        'comment': f"Maximum wind speed per SAR tile. {quality_note}",
    })

    wave_ds_out['SAR_wind_speed_med'] = ('tile', meds, {
        'long_name': f"Median {base_attrs.get('long_name', 'wind speed')}",
        'units': base_attrs.get('units', 'm/s'),
        'description': base_attrs.get('description', ''),
        'comment': f"Median wind speed per SAR tile. {quality_note}",
    })

    wave_ds_out['SAR_wind_speed_count'] = ('tile', counts, {
        'long_name': "Number of valid wind speed points used per SAR tile",
        'units': '1',
        'comment': quality_note,
    })

    # Global attributes: trace back to the source SAR wind product and document the method
    try:
        sar_wind_source = wind_ds.encoding["source"]
    except KeyError:
        sar_wind_source = "unknown"
    wave_ds_out.attrs['SAR_wind_product'] = sar_wind_source

    wave_ds_out.attrs['comment'] = (
        "SAR_wind_speed_mean/std/min/max/count were computed by extracting all SAR wind grid points "
        "whose coordinates fall inside each tile's footprint (tile_footprint, WKT polygon), "
        "keeping only points with quality_level >= 3. wind_speed_count reports the number of "
        "valid points used per tile; tiles with count = 0 had no matching wind point and are set to NaN."
    )

    return wave_ds_out