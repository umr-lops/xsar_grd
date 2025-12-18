import sys
import numpy as np
import xarray as xr
sys.path.append('/home1/datahome/egauvrit/Libraries/xsar_slc')
from xsarslc.tools import netcdf_compliant
from xsarslc.processing.xspectra import symmetrize_xspectrum, xs_formatting

def gegenbauer_polynoms(x, nk, lbda=3 / 2.):
    """

    Args:
        x: np.ndarray
        nk: int
        lbda: float

    Returns:
        Cnk : np.ndarray
    """
    C0 = 1
    if (nk == 0):
        return C0 + x * 0
    C1 = 3 * x
    if (nk == 1):
        return C1 + x * 0

    Cnk = (1 / nk) * (2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda) - (
            nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda))
    Cnk = (1 / nk) * (2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda) - (
            nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda))

    return Cnk


def harmonic_functions(x, nphi):
    """

    Args:
        x: np.ndarray
        nphi: int

    Returns:
        Fn : np.ndarray
    """
    if nphi == 1:
        Fn = np.sqrt(1 / np.pi) + x * 0
        return Fn

    # Even & Odd case
    if nphi % 2 == 0:
        Fn = np.sqrt(2 / np.pi) * np.sin((nphi) * x)
    else:
        Fn = np.sqrt(2 / np.pi) * np.cos((nphi - 1) * x)

    return Fn


def compute_kernel(krg, kaz, kmin, kmax, Nk=4, Nphi=5, save_cwave_kernel=False):
    """
    Compute CWAVE kernels
    Args:
        krg (xarray.DataArray) : spectrum wavenumbers in range direction
        kaz (xarray.DataArray) : spectrum wavenumbers in azimuth direction
        
    Keywords Args:
        Nk (int) : 
        Nphi (int) : 
        save_cwave_kernel (bool, optional) : save CWAVE kernel on disk in working directory

    Return:
        (xarray.Dataset) : kernels
    """    
    # Kernel Computation
    #
    coef = lambda nk : (nk + 3 / 2.) / ((nk + 2.) * (nk + 1.))
    nu = lambda x : np.sqrt(1 - x ** 2.)
    
    gamma = 2
    a1 = (gamma ** 2 - np.power(gamma, 4)) / (gamma ** 2 * kmin ** 2 - kmax ** 2)
    a2 = (kmax ** 2 - np.power(gamma, 4) * kmin ** 2) / (kmax ** 2 - gamma ** 2 * kmin ** 2)
    tmp = a1 * np.power(kaz, 4) + a2 * kaz ** 2 + krg ** 2
    # alpha k
    alpha_k = 2 * ((np.log10(np.sqrt(tmp)) - np.log10(kmin)) / (np.log10(kmax) - np.log10(kmin))) - 1
    # alpha phi
    alpha_phi = np.arctan2(krg, kaz).rename(None)
    # eta
    eta = np.sqrt((2. * tmp) / ((kaz ** 2 + krg ** 2) * tmp * np.log10(kmax / kmin)))

    Gnk = xr.combine_by_coords([gegenbauer_polynoms(alpha_k, ik - 1, lbda=3 / 2.) * coef(ik - 1) * nu(
        alpha_k).assign_coords({'k_gp': ik}).expand_dims('k_gp') for ik in np.arange(Nk) + 1])
    Fnphi = xr.combine_by_coords(
        [harmonic_functions(alpha_phi, iphi).assign_coords({'phi_hf': iphi}).expand_dims('phi_hf') for iphi in
         np.arange(Nphi) + 1])

    Kernel = Gnk * Fnphi * eta
    Kernel.k_gp.attrs.update({'long_name': 'Gegenbauer polynoms dimension'})
    Kernel.phi_hf.attrs.update({'long_name': 'Harmonic functions dimension (odd number)'})

    _Kernel = Kernel.rename('cwave_kernel').to_dataset()
    if 'pol' in _Kernel:
        _Kernel = _Kernel.drop_vars('pol')
    _Kernel['cwave_kernel'].attrs.update({'long_name': 'CWAVE Kernel'})

    ds_G = Gnk.rename('Gegenbauer_polynoms').to_dataset()
    ds_F = Fnphi.rename('Harmonic_functions').to_dataset()
    ds_eta = eta.rename('eta').to_dataset()
    if 'pol' in ds_G:
        ds_G = ds_G.drop_vars('pol')
        ds_F = ds_F.drop_vars('pol')
        ds_eta = ds_eta.drop_vars('pol')

    Kernel = xr.merge([_Kernel, ds_G, ds_F, ds_eta])

    if (save_cwave_kernel):
        Kernel.to_netcdf('cwaves_kernel.nc')

    return Kernel


def compute_cwave_parameters(cs, kmin=2*np.pi/600, kmax = 2*np.pi/25, **kwargs):
    """
    Compute CWAVE parameters from a spectrum dataset.

    Parameters
    ----------
    cs : xarray.DataArray or xarray.Dataset
        Spectrum data containing at least coordinates
        `k_rg` (range wavenumber), `k_az` (azimuth wavenumber),
        and dimensions `freq_line` and `freq_sample`.
    kmin : float, optional
        Lower cutoff for the radial wavenumber filter (default: 2π/600).
    kmax : float, optional
        Upper cutoff for the radial wavenumber filter (default: 2π/25).
    **kwargs :
        Additional keyword arguments passed to sub-functions.

    Returns
    -------
    xarray.DataArray
        The computed CWAVE parameter field named ``'cwave'`` with an
        attribute ``long_name`` describing the content.
    """

    cs = symmetrize_xspectrum(cs)

    # Cross-Spectra Frequency Filtering  
    kk = np.sqrt((cs.k_rg) ** 2. + (cs.k_az) ** 2.)
    ccs = cs.where((kk > kmin) & (kk < kmax)) # drop=True here would remove other dimensions we may want to keep
    ccs = ccs.dropna(dim='freq_line', how='all').dropna(dim='freq_sample', how='all') # removing only along the chosen dimensions

    # Cross-Spectra normalization
    ccsm = np.abs(ccs)
    dkrg = ccs.k_rg.diff(dim='freq_sample').mean(dim='freq_sample') # works even if there are other dimensions than expected
    dkaz = ccs.k_az.diff(dim='freq_line').mean(dim='freq_line') # works even if there are other dimensions than expected
    ccsmn = ccsm /(ccsm.sum(dim=['freq_line', 'freq_sample']) * dkrg * dkaz)

    # XS decomposition Kernel
    kernel = compute_kernel(krg=ccs.k_rg, kaz=ccs.k_az, kmin=kmin, kmax=kmax)

    # CWAVE paremeters computation
    cwave_parameters = ((kernel.cwave_kernel * ccsmn) * dkrg * dkaz).sum(dim=['freq_sample', 'freq_line'], skipna=True, min_count=1)

    cwave_parameters = cwave_parameters.rename('cwave')
    cwave_parameters.attrs['long_name'] = "CWAVE parameters"

    return cwave_parameters