import aiohttp
import fsspec
import logging
import xarray as xr
from pathlib import Path

logger = logging.getLogger(__name__)


def build_l1c_fileout(l1b_fullpath)->str:
    """
    Transform a Level-1B GRD path into a Level-1C GRD

    Args:
        l1b_fullpath: str .nc level-1B full path "S1...SAFE.nc"

    Returns:
        str : fullpath of l1c product
    """
    return l1b_fullpath.replace('l1b','l1c')


def save_l1c_dataset(
    ds: xr.Dataset,
    fileout: str,
    *,
    overwrite: bool = False,
    extra_attrs: dict | None = None,
) -> None:
    """
    Save a Level-1C dataset to NetCDF with standardized attributes.

    Parameters
    ----------
    ds : xr.Dataset
        L1C dataset to save.
    fileout : str
        Full output path to NetCDF file.
    overwrite : bool
        If True, overwrite existing file.
    extra_attrs : dict, optional
        Additional global attributes to attach before saving.
    """
    file_path = Path(fileout)

    # --- Directory handling ---
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        if overwrite:
            logger.warning(f"[IO] Overwriting existing file: {file_path}")
            file_path.unlink()
        else:
            logger.info(f"[IO] Output already exists, skipping: {file_path}")
            return

    # --- Attributes ---
    if extra_attrs is not None:
        ds.attrs.update(extra_attrs)

    # --- Writing ---
    try:
        ds.to_netcdf(file_path)
        logger.info(f"[IO] L1C file written to: {file_path}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to write NetCDF: {file_path}")
        raise


def url_get(url):
    """
    Get fil from url, using caching.
    Parameters
    ----------
    url: str
    cache_dir: str
        Cache dir to use. default to `os.path.join(config['data_dir'], 'fsspec_cache')`
    Raises
    ------
    FileNotFoundError
    Returns
    -------
    filename: str
        The local file name
    Notes
    -----
    Due to fsspec, the returned filename won't match the remote one.
    """

    if "://" in url:
        with fsspec.open(
            "filecache::%s" % url,
            https={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
        ) as f:
            fname = f.name
    else:
        fname = url

    return fname