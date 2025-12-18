#from tools import *
import numpy as np
import xarray as xr
from tqdm import tqdm
from xsargrd.l1b.tools import check_tile_mask
from xsarslc.tools import xtiling, xndindex
from xsarslc.processing.xspectra import compute_modulation, compute_normalized_variance


def get_tiles_width(tile_size, tile_spacing):
    func = lambda x, y: np.int16(x*y)
    return xr.apply_ufunc(func, tile_size, tile_spacing)


def get_tiles_mean_value(da, dims, long_name=None):
    """
    Compute the mean of a DataArray over specified dimensions, 
    preserving attributes and assigning a new name.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.
    dims : str or list of str
        Dimensions over which to compute the mean.
    long_name : str
        A descriptive long name for the resulting DataArray.

    Returns
    -------
    xr.DataArray
        The mean DataArray with updated name and attributes.
    """
    return (
        da.mean(dim=dims, keep_attrs=True)
          .assign_attrs({
              "long_name": long_name or f"Mean {da.name} over tile." if da.name else "Mean value over the tile.",
              "units": da.attrs.get("units", ""),
          })
          .rename(f"mean_{da.name}")
    )

def get_tiles_center_value(da, long_name=None, units=None):
    """
    Extract the value of a DataArray at the center of a tile and 
    return it as a scalar DataArray with metadata.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray containing dimensions 'line' and 'sample'.
    long_name : str, optional
        Descriptive long name for the result. If None, a default is generated.
    units : str, optional
        Units of the result. If None, taken from the input attrs if available.

    Returns
    -------
    xr.DataArray
        Scalar DataArray with the value at the center and attributes set.
    """
    center_line   = int(da.sizes["line"] / 2)
    center_sample = int(da.sizes["sample"] / 2)

    return xr.DataArray(
        da.isel(line=center_line, sample=center_sample),
        name=f"center_{da.name}" if da.name else "center_value",
        attrs={
            "long_name": long_name or f"{da.name} at tile center" if da.name else "value at tile center",
            "units": units or da.attrs.get("units", "")
        }
    )

def compute_modulation_spectrum(
    mod: xr.DataArray,
    ground_range_spacing: float,
    azimuth_spacing: float,
    range_dim: str = 'sample',
    azimuth_dim: str = 'line',
    nperseg: dict[str, int] = None,
    noverlap: dict[str, int] = None,
    **kwargs
) -> xr.DataArray:
    """
    Compute the co-spectrum using a 2D Welch method (periodograms).

    Parameters
    ----------
    mod : xarray.DataArray
        Modulation signal
    ground_range_spacing : float
        Ground spacing in the range (sample) direction [meters].
    azimuth_spacing : float
        Spacing in the azimuth (line) direction [meters].
    range_dim : str, optional
        Name of the range dimension (default is 'sample').
    azimuth_dim : str, optional
        Name of the azimuth dimension (default is 'line').
    nperseg : dict[str, int], optional
        Number of points per segment for the periodogram, as a dictionary keyed by dimension names.
        Defaults to {'sample': 512, 'line': 512} if not provided.
    noverlap : dict[str, int], optional
        Number of overlapping points per segment for the periodogram, as a dictionary keyed by dimension names.
        Defaults to {'sample': 256, 'line': 256} if not provided.
    **kwargs
        Additional keyword arguments passed to the underlying spectral computation function.

    Returns
    -------
    xarray.DataArray
        Concatenated co-spectra
    """
    if nperseg is None:
        nperseg = {range_dim: 512, azimuth_dim: 512}
    if noverlap is None:
        noverlap = {range_dim: 256, azimuth_dim: 256}

    freq_rg_dim = 'freq_' + range_dim
    freq_azi_dim = 'freq_' + azimuth_dim

    periodo_slices = xtiling(mod, nperseg=nperseg, noverlap=noverlap, prefix='periodo_')

    periodo = mod[periodo_slices]  # .swap_dims({'__'+d:d for d in [range_dim, azimuth_dim]})
    periodo_sizes = {d: k for d, k in periodo.sizes.items() if 'periodo_' in d}

    dims_to_expand = ['periodo_'+range_dim, 'periodo_'+azimuth_dim]

    if 'pol' in mod.sizes:
        periodo_sizes.update({'pol':mod.sizes.get('pol')})
        dims_to_expand+=['pol']

    # if 'tile' in mod.sizes:
    #     periodo_sizes.update({'tile':mod.sizes.get('tile')})
    #     dims_to_expand+=['tile']
    
    out = list()
    for i in xndindex(periodo_sizes):
        image = periodo[i].swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
        image = (image - image.mean()) / image.mean()
        
        #
        cspecs = xr.DataArray(abs(np.fft.fft2(image))**2,
                              dims=['freq_' + d for d in image.dims])
        cspecs = cspecs[{freq_rg_dim: slice(None, cspecs.sizes[freq_rg_dim] // 2 + 1)}]  # keeping only half of the wavespectrum (positive wavenumbers)
        cspecs.data = np.fft.fftshift(cspecs.data,
                                      axes=cspecs.get_axis_num(freq_azi_dim))  # fftshifting azimuthal wavenumbers
        #
        periodo_coords = {k:(v if k not in image.coords else image[k].item()) for k,v in i.items()}
        cspecs = cspecs.assign_coords(periodo_coords)
        out.append(cspecs)

    out = xr.combine_by_coords([x.expand_dims(dims_to_expand) for x in out], combine_attrs='drop_conflicts').rename('spectra')

    # dealing with wavenumbers
    k_rg = xr.DataArray(np.fft.rfftfreq(nperseg[range_dim], ground_range_spacing / (2 * np.pi)), dims=freq_rg_dim,
                        name='k_rg', attrs={'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    k_az = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[azimuth_dim], azimuth_spacing / (2 * np.pi))),
                        dims='freq_' + azimuth_dim, name='k_az',
                        attrs={'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    
    out = out/(out.sizes['freq_line']*out.sizes['freq_sample'])
    out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az})
    out.attrs.update({'periodogram_nperseg_' + range_dim: nperseg[range_dim],
                      'periodogram_nperseg_' + azimuth_dim: nperseg[azimuth_dim],
                      'periodogram_noverlap_' + range_dim: noverlap[range_dim],
                      'periodogram_noverlap_' + azimuth_dim: noverlap[azimuth_dim]})
    return out

def compute_spectra(
    tiles: xr.Dataset,
    periodo_width: dict[str, float] = {'sample': 2000., 'line': 2000.},
    periodo_overlap: dict[str, float] = {'sample': 1000., 'line': 1000.},
    lowpass_width: dict[str, float] = {'line': 4750., 'sample': 4750.},
) -> xr.Dataset:
    """
    Compute GRD (Ground Range Detected) spectra for all SAR tiles in parallel
    using xarray vectorization.

    Parameters
    ----------
    tiles : xarray.Dataset
        Input dataset containing SAR tiles with dimensions (tile, pol, tile_line, tile_sample).
        Must include variables necessary for spectral and statistical calculations.
    periodo_width : dict of float, optional
        Window size in meters for periodogram segments, keyed by dimension names.
        Default is {'sample': 2000., 'line': 2000.}.
    periodo_overlap : dict of float, optional
        Overlap size in meters between periodogram segments, keyed by dimension names.
        Default is {'sample': 1000., 'line': 1000.}.
    lowpass_width : dict of float, optional
        Width in meters for low-pass filtering during modulation computation,
        keyed by dimension names. Default is {'line': 4750., 'sample': 4750.}.

    Returns
    -------
    xarray.Dataset
        Dataset containing spectral, statistical, and descriptive variables
        computed per tile.
    """
    # --- Center values ---
    nl = tiles.line.size
    ns = tiles.sample.size
    center_line           = tiles.__line.isel(line=nl//2).rename('center_line').assign_attrs(long_name='line value at tile center')
    center_sample         = tiles.__sample.isel(sample=ns//2).rename('center_sample').assign_attrs(long_name='sample value at tile center')
    center_incidence      = get_tiles_center_value(tiles.incidence)
    center_longitude      = get_tiles_center_value(tiles.longitude)
    center_latitude       = get_tiles_center_value(tiles.latitude)
    center_ground_heading = get_tiles_center_value(tiles.ground_heading)

    # --- Initiate variables ---
    nv_f = []
    nv_lpf = []
    specs_m = []
    specs_v = []
    variables_list = []   

    # --- Tiles to process ---
    indexes = tiles.tile.values

    # --- Loop over tiles ---
    for i in tqdm(indexes, total=indexes.size, desc="Spectrum calculation", unit="tile"):

        # --- Select tile ---
        tile = tiles.sel(tile=i)

        # --- Tile spacing ---
        lineSpacing   = tile.lineSpacing.item()
        sampleSpacing = tile.sampleSpacing.item()
        spacing       = {'sample': sampleSpacing, 'line': lineSpacing}

        # --- Digital number ---
        dn = tile.digital_number
        
        # --- Modulation ---
        mod, _nv_lpf = compute_modulation(
            np.abs(dn),
            lowpass_width=lowpass_width,
            spacing=spacing
        )
        nv_lpf.append(_nv_lpf.expand_dims(tile=[i]))
        nv_f.append(compute_normalized_variance(mod).expand_dims(tile=[i]))

        # --- Spectrum ---
        nperseg_periodo = {d: int(np.rint(periodo_width[d] / spacing[d])) for d in periodo_width}
        noverlap_periodo = {d: int(np.rint(periodo_overlap[d] / spacing[d])) for d in periodo_overlap}
        specs = compute_modulation_spectrum(
            mod**2,
            float(sampleSpacing), float(lineSpacing),
            range_dim='sample', azimuth_dim='line',
            nperseg=nperseg_periodo, noverlap=noverlap_periodo
        )
        specs_m.append(
            specs.mean(dim=['periodo_line', 'periodo_sample'], keep_attrs=True)
            .expand_dims(tile=[i])
        )
        specs_v.append(
            specs.var(dim=['periodo_line', 'periodo_sample'],keep_attrs=True)
                .rename('var_' + specs.name)
                .expand_dims(tile=[i])
        )

    # --- Compute mean values ---
    mean_sigma0 = get_tiles_mean_value(tiles.sigma0,('sample','line'))
    mean_nesz   = get_tiles_mean_value(tiles.nesz,('sample','line'))

    # --- concatenate along tile ---
    nv_f    = xr.concat(nv_f,    dim='tile')
    nv_lpf  = xr.concat(nv_lpf,  dim='tile')
    specs_m = xr.concat(specs_m, dim='tile')
    specs_v = xr.concat(specs_v, dim='tile')
    
    # --- Build output ---
    variables_list += [tiles.tile_footprint,
                       center_line,
                       center_sample,
                       center_incidence,
                       center_latitude,
                       center_longitude,
                       center_ground_heading,
                       mean_sigma0,
                       mean_nesz,
                       nv_f,
                       nv_lpf,
                       specs_m,
                       specs_v,
                       ]
    
    out = xr.Dataset({da.name: da for da in variables_list})
    
    # --- Formatting spectra before  ---    
    Nfreqs = [out[x].sizes['freq_sample'] if 'freq_sample' in out[x].dims else np.nan for x in out if 'freq_sample' in out[x].dims]

    if np.any(np.isfinite(Nfreqs)):
        # Returned xspecs have different shape in range (to keep same dk).
        # Lines below only select common portions of xspectra.
        Nfreq_min = min(Nfreqs)
        out = out.sel(freq_sample=slice(None, Nfreq_min))
    
    # --- Remove some coordinates ---
    out = out.drop_vars(('line','sample','__line','__sample'))

    return out