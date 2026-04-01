
import numpy as np
import xarray as xr
import logging
from datetime import datetime, timedelta
from xsargrd.l1c.tools import url_get

logger = logging.getLogger(__name__)

def resource_strftime(resource, **kwargs):
    """
    From a resource string like '%Y/%j/myfile_%Y%m%d%H%M.nc' and a date like 'Timestamp('2018-10-13 06:23:22.317102')',
    returns a tuple composed of the closer available date and string like '/2018/286/myfile_201810130600.nc'
    If ressource string is an url (ie 'ftp://ecmwf/%Y/%j/myfile_%Y%m%d%H%M.nc'), fsspec will be used to retreive the file locally.

    Parameters
    ----------
    resource: str
        resource string, with strftime template
    date: datetime
        date to be used
    step: int
        hour step between 2 files
    Returns
    -------
    tuple : (datetime,str)
    """

    date = kwargs["date"]
    step = kwargs["step"]

    delta = timedelta(hours=step) / 2
    date = date.replace(
        year=(date + delta).year,
        month=(date + delta).month,
        day=(date + delta).day,
        hour=int((date + delta).hour // step * step),
        minute=0,
        second=0,
        microsecond=0,
    )
    if "%(Y+1)" in resource:
        resource = resource.replace(
            "%(Y+1)", (date + datetime.timedelta(days=366)).strftime("%Y")
        )
    return date, url_get(date.strftime(resource))


def _to_lon180(ds):
    # roll [0, 360] to [-180, 180] on dim x
    ds = ds.roll(x=-np.searchsorted(ds.x, 180), roll_coords=True)
    ds["x"] = xr.where(ds["x"] >= 180, ds["x"] - 360, ds["x"])
    return ds


def ecmwf_0100_1h(fname, use_dask=False):
    """
    ecmwf 0.1 deg 1h reader (ECMWF_FORECAST_0100_202109091300_10U_10V.nc)

    Parameters
    ----------
    fname: str

        hwrf filename
    Returns
    -------
    xarray.Dataset
    """
    ecmwf_ds = open_netcdf_fallback(
        fname,
        decode_timedelta=True
    ).isel(time=0)

    ecmwf_ds.attrs["time"] = datetime.fromtimestamp(
        ecmwf_ds.time.item() // 1000000000
    )
    ecmwf_ds = ecmwf_ds.drop_vars("time").rename(
        {"Longitude": "x", "Latitude": "y", "10U": "U10", "10V": "V10"}
    )
    ecmwf_ds.attrs = {k: ecmwf_ds.attrs[k] for k in ["title", "institution", "time"]}

    # dataset is lon [0, 360], make it [-180,180]
    ecmwf_ds = _to_lon180(ecmwf_ds)

    ecmwf_ds.rio.write_crs("EPSG:4326", inplace=True)

    if use_dask is False:
        for var in ecmwf_ds:
            ecmwf_ds[var] = ecmwf_ds[var].compute()

    return ecmwf_ds


def ecmwf_0125_1h(fname):
    """
    ecmwf 0.125 deg 1h reader (ecmwf_201709071100.nc)

    Parameters
    ----------
    fname: str

        hwrf filename
    Returns
    -------
    xarray.Dataset
    """
    ecmwf_ds = xr.open_dataset(fname, chunks={"longitude": 1000, "latitude": 1000})

    ecmwf_ds = (
        ecmwf_ds.rename({"longitude": "x", "latitude": "y"})
        .rename({"Longitude": "x", "Latitude": "y", "U": "U10", "V": "V10"})
        .set_coords(["x", "y"])
    )

    ecmwf_ds["x"] = ecmwf_ds.x.compute()
    ecmwf_ds["y"] = ecmwf_ds.y.compute()

    # dataset is lon [0, 360], make it [-180,180]
    ecmwf_ds = _to_lon180(ecmwf_ds)

    ecmwf_ds.attrs["time"] = datetime.fromisoformat(ecmwf_ds.attrs["date"])

    ecmwf_ds.rio.write_crs("EPSG:4326", inplace=True)

    return ecmwf_ds


def ww3_global_yearly_3h(fname, date):
    str_time = datetime.strftime(date, "%Y-%m-%d-T%H:%M:%S.000000000")

    ds = open_netcdf_fallback(
        fname,
        decode_timedelta=True
    )

    dst = (
        ds.sel(time=str_time)
        .drop_vars("time")
        .rename({"longitude": "x", "latitude": "y"})
    )

    dst.rio.write_crs("EPSG:4326", inplace=True)

    return dst


def open_netcdf_fallback(
    fname,
    engines=("h5netcdf", "netcdf4"),
    **open_kwargs,
):
    """
    Open a NetCDF file using several xarray engines as fallback.

    Parameters
    ----------
    fname : str or Path
        Path to the NetCDF file.
    engines : tuple/list
        Engines to try in order.
    **open_kwargs :
        Additional kwargs passed to xr.open_dataset().

    Returns
    -------
    xr.Dataset
    """

    last_error = None

    for engine in engines:
        try:
            ds = xr.open_dataset(
                fname,
                engine=engine,
                **open_kwargs
            )
            logger.info(f"[IO] Opened {fname} with engine='{engine}'")
            return ds

        except Exception as e:
            last_error = e
            logger.info(f"[IO] Engine '{engine}' failed for {fname}: {e}")

    raise RuntimeError(
        f"Could not open {fname} with engines {engines}"
    ) from last_error