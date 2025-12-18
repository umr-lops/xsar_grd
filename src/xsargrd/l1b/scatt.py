import warnings
import numpy as np
import xarray as xr
import pywst as pw
from xsargrd.l1b.tools import check_tile_mask
import logging

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,  # niveau par défaut (INFO = affiché, DEBUG = plus verbeux, WARNING/ERROR moins)
    format="%(message)s",  # format simple sans timestamp
)

logger = logging.getLogger(__name__)

def closest_pow2(x: int):
    """
    Return the largest power of 2 less than or equal to x,
    and its exponent.

    Parameters
    ----------
    x : int
        Input positive integer.

    Returns
    -------
    M : int
        Largest power of 2 <= x.
    J : int
        Exponent such that M = 2**J.
    """
    # Convert floats to int
    if not isinstance(x, (int, np.integer)):
        x = int(x)

    if x < 1:
        raise ValueError("x must be >= 1")
    
    # bit_length trick
    J = (x.bit_length() - 1)   # index of highest set bit
    M = 1 << J                 # 2**J
    return M, J

def get_operators(
        M: int,
        J: int,
        L: int,
        *,
        OS: int = 0,
        use_rwst: bool = True):
    """
    Create WST (and optionally RWST) operators.

    Parameters
    ----------
    M : int
        Input size (signal/image side length).
    J : int
        Number of dyadic scales.
    L : int
        Number of orientations.
    OS : int, optional
        Oversampling parameter (default=0).
    use_rwst : bool, optional
        If True (default), also create the Reduced WST operator.
        If False, only the WST operator is returned.

    Returns
    -------
    (op,) or (op, rop)
        If `use_rwst=False`, returns a 1-tuple `(op,)`.
        If `use_rwst=True`, returns `(op, rop)`.
    """
    op = pw.WSTOp(M, M, J, L, OS)
    if use_rwst:
        rop = pw.RWSTOp(M, M, J, L, OS, wst_op=op)
        return op, rop
    else:
        return (op,)


def apply_operators(x, op, rop=None):
    """
    Apply the wavelet scattering transform (WST) and optionally
    the reduced wavelet scattering transform (RWST).

    Parameters
    ----------
    x : ndarray
        Input signal or image (depending on the operator definition).
    op : pw.WSTOp
        Wavelet scattering transform operator created with `pw.WSTOp`.
    rop : pw.RWSTOp, optional
        Reduced scattering operator created with `pw.RWSTOp`.
        If provided, RWST is attempted on the normalized WST.

    Returns
    -------
    (wst_data,) or (wst_data, rwst_data)
        Always returns at least a 1-tuple containing `wst_data`.
        If RWST succeeds, returns `(wst_data, rwst_data)`.
        If RWST fails to converge or raises, returns `(wst_data,)`
        and attaches the exception to `wst_data.rwst_error`.

    Notes
    -----
    - WST is always applied first with `crop=0`.
    - WST coefficients are converted to log2 scale and normalized
      before computing RWST.
    - Downstream, you can call:
        `coeffs_to_dataset(wst_data, rwst_data, fill_missing_rwst=True)`
      so that when RWST failed (i.e. `rwst_data` is absent),
      RWST variables are created with NaN and shapes consistent with WST.
    """
    # 1) WST
    wst_data = op.apply(x, crop=0)
    wst_data.to_log2()
    wst_data.normalize()

    # 2) RWST (optional, failure-tolerant)
    if rop is None:
        return (wst_data,)

    try:
        rwst_data = rop.apply(wst_data)
        return wst_data, rwst_data
    except RuntimeError as e:
        # Typical SciPy curve_fit failure (e.g., maxfev reached)
        warnings.warn(
            f"RWST optimization failed: {e}. "
            "Proceeding without RWST; downstream may fill NaNs.",
            RuntimeWarning,
        )
        setattr(wst_data, "rwst_error", e)
        return (wst_data,)
    except Exception as e:
        # Any unexpected RWST error → same fallback
        warnings.warn(
            f"RWST failed with unexpected error: {e}. "
            "Proceeding without RWST; downstream may fill NaNs.",
            RuntimeWarning,
        )
        setattr(wst_data, "rwst_error", e)
        return (wst_data,)


def extract_subtiles(x, M):
    """
    Extract 9 subtiles of shape (M, M) from a square tile x of shape (N, N).
    Works on input of shape (N, N).
    """
    N = x.shape[0]
    R = (N - M) // 2
    X = np.zeros((9, M, M), dtype=x.dtype)

    # 3x3 subtiles
    X[0] = x[:M, :M]                                # top-left
    X[1] = x[:M, R:R+M]                             # top-center
    X[2] = x[:M, -M:]                               # top-right
    X[3] = x[R:R+M, :M]                             # middle-left
    X[4] = x[(N-M)//2:(N+M)//2, (N-M)//2:(N+M)//2]  # center
    X[5] = x[R:R+M, -M:]                            # middle-right
    X[6] = x[-M:, :M]                               # bottom-left
    X[7] = x[-M:, R:R+M]                            # bottom-center
    X[8] = x[-M:, -M:]                              # bottom-right

    return X

def subtile_da(da, M):
    """
    Apply the extract_subtiles function to a DataArray of tiles.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with dimensions ('tile', 'tile_line', 'tile_sample').
    M : int
        Size of the subtiles to extract (M x M).

    Returns
    -------
    xr.DataArray
        Output DataArray with dimensions ('tile', 'subtile', 'x_sub', 'y_sub').
        Each tile is split into 9 subtiles of size M x M.
    """
    result = xr.apply_ufunc(
        extract_subtiles,
        da,
        kwargs={"M": M},
        input_core_dims=[["tile_line", "tile_sample"]],
        output_core_dims=[["subtile", "x_sub", "y_sub"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )
    return result

def center_crop(da, M, dims=('line', 'sample')):
    """
    Extract the central MxM patch from each tile in a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input with dimensions ('tile', 'line', 'sample').
    M : int
        Size of the central crop.

    Returns
    -------
    xr.DataArray
        Cropped DataArray with dimensions ('tile', 'y_sub', 'x_sub').
    """
    N = da.sizes[dims[0]]  # assume square tiles: tile_line == tile_sample
    R = (N - M) // 2

    cropped = da.isel(
        **{dims[0]: slice(R, R + M),
           dims[1]: slice(R, R + M)}
    )

    return cropped

def coeffs_to_dataset(wst_data, rwst_data=None, *, tiles_index=None, pol_index=None):
    """
    Transform scattering coefficients (WST + optional RWST) into an xarray.Dataset.

    Parameters
    ----------
    wst_data : object
        PyWST object exposing get_coeffs() -> (coefs, index, ...)
        index rows encode [layer, j1, l1, j2, l2].
    rwst_data : object, optional
        PyWST object exposing get_coeffs(varname) -> ndarray
        Shapes:
          - First-order: (j1, tile)
          - Second-order: (j1, j2, tile)

    Returns
    -------
    xr.Dataset
        With S0, S1, S2 and (if provided) RWST fields:
        S1Iso, S1Aniso, ThetaRef1, S2Iso1, S2Iso2, S2Aniso1, S2Aniso2, ThetaRef2.
    """

    # =========== WST ===========
    # Extract coefficients and index arrays
    coefs, index, *_ = wst_data.get_coeffs()

    layer = index[0, :].astype(int)   # 0 = S0, 1 = S1, 2 = S2
    j1 = index[1, :].astype(int)
    l1 = index[2, :].astype(int)
    j2 = index[3, :].astype(int)
    l2 = index[4, :].astype(int)

    # check coefs shape & define number of tiles :
    dim = len(coefs.shape)
    if dim==1:
        nt = 1
    elif dim==2:
        nt = coefs.shape[1]
    else:
        raise ValueError(f"Unexpected coeffs ndim={coefs.ndim}, shape={coefs.shape}")
    
    # If pol_index is provided, nt corresponds to tile * pol
    if pol_index is not None:
        pol_vals = pol_index.values
        pol_attrs = pol_index.attrs if hasattr(pol_index, "attrs") else {}
        npol = pol_index.size
        if nt % npol != 0:
            raise ValueError(
                f"Inconsistent shapes: nt={nt} is not divisible by npol={npol}"
            )
        nt = nt // npol

    else:
        # No pol information provided → assume single polarization
        pol_vals = np.array(["pol0"])
        pol_attrs = {"long_name": "Synthetic polarization (none provided)"}
        npol = 1
    
    # check tiles_index
    if tiles_index is not None:
        # tiles_index must be xr dataarray of length nt
        if not isinstance(tiles_index, xr.DataArray):
            raise ValueError(f"tiles_index must be an xarray.DataArray, got {type(tiles_index)}")
        if tiles_index.sizes.get('tile', None) != nt:
            raise ValueError(f"tiles_index length {tiles_index.sizes.get('tile', None)} does not match number of tiles {nt}")
        tiles_index = tiles_index.values
        tiles_attrs = tiles_index.attrs if hasattr(tiles_index, 'attrs') else {}
    else:
        tiles_index = np.arange(nt)
        tiles_attrs = {"long_name": "Tile index", "comment": "Assigned during scattering coefficients extraction."}

    # --- Define coordinates once ---
    coords = {
        "tile": xr.DataArray(
            tiles_index,
            dims=("tile",),
            attrs=tiles_attrs,
        ),
        "pol": xr.DataArray(
            pol_vals,
            dims=("pol",),
            attrs=pol_attrs,
        ),
        "j1": xr.DataArray(
            np.unique(j1[layer == 1]),
            dims=("j1",),
            attrs={
                "long_name": "First-order scale index",
                "description": "Wavelet scale index for the first convolution (order 1).\nWavelet scales are defined as 2^j1 pix, with j1 ∈ [0, J-1].",
            },
        ),
        "l1": xr.DataArray(
            np.unique(l1[layer == 1]),
            dims=("l1",),
            attrs={
                "long_name": "First-order orientation index",
                "description": "Orientation index for the first convolution (order 1).\nWavelet angles are defined as θ1 = π/L × l1, with l1 ∈ [0, L-1]",
            },
        ),
        "j2": xr.DataArray(
            np.unique(j2[layer == 2]),
            dims=("j2",),
            attrs={
                "long_name": "Second-order scale index",
                "description": "Wavelet scale index for the second convolution (order 2).\nWavelet scales are defined as 2^j2 pix, with j2 ∈ [1, J-1].",
            },
        ),
        "l2": xr.DataArray(
            np.unique(l2[layer == 2]),
            dims=("l2",),
            attrs={
                "long_name": "Second-order orientation index",
                "description": "Orientation index for the second convolution (order 2).\nWavelet angles are defined as θ2 = π/L × l2, with l2 ∈ [0, L-1]",
            },
        ),
    }

    # --- S0 ---
    mask_s0 = layer == 0
    S0 = coefs[mask_s0].reshape(nt, npol)

    da_S0 = xr.DataArray(
        S0,
        dims=("tile", "pol"),
        coords={"tile": coords["tile"], "pol": coords["pol"]},
        attrs={
            "long_name": "Zero-order scattering coefficient",
            "description": "Mean of the input signal",
        },
    )

    # --- S1 ---
    mask_s1 = layer == 1
    j1_vals = coords["j1"].values
    l1_vals = coords["l1"].values

    S1 = np.full((nt, npol, len(j1_vals), len(l1_vals)), np.nan)
    for i, jj in enumerate(j1_vals):
        for k, ll in enumerate(l1_vals):
            idx = mask_s1 & (j1 == jj) & (l1 == ll)
            if np.any(idx):
                S1[..., i, k] = coefs[idx].reshape(nt, npol)

    da_S1 = xr.DataArray(
        S1,
        dims=("tile", "pol", "j1", "l1"),
        coords={
            "tile": coords["tile"], "pol": coords["pol"],
            "j1": coords["j1"], "l1": coords["l1"]
        },
        attrs={
            "long_name": "First-order scattering coefficients",
            "description": "S1 = ⟨|X * ψ_{j1,l1}|⟩, i.e. convolution with a wavelet family at (j1,l1), modulus, then averaging.",
        },
    )

    # --- S2 ---
    mask_s2 = layer == 2
    j1_vals = coords["j1"].values
    l1_vals = coords["l1"].values
    j2_vals = coords["j2"].values
    l2_vals = coords["l2"].values

    S2 = np.full((nt, npol, len(j1_vals), len(l1_vals), len(j2_vals), len(l2_vals)), np.nan)
    for i, jj in enumerate(j1_vals):
        for k, ll1 in enumerate(l1_vals):
            for m, jj2 in enumerate(j2_vals):
                for n, ll2 in enumerate(l2_vals):
                    idx = mask_s2 & (j1 == jj) & (l1 == ll1) & (j2 == jj2) & (l2 == ll2)
                    if np.any(idx):
                        S2[..., i, k, m, n] = coefs[idx].reshape(nt, npol)

    da_S2 = xr.DataArray(
        S2,
        dims=("tile", "pol", "j1", "l1", "j2", "l2"),
        coords={
            "tile": coords["tile"],
            "pol": coords["pol"],
            "j1": coords["j1"],
            "l1": coords["l1"],
            "j2": coords["j2"],
            "l2": coords["l2"],
        },
        attrs={
            "long_name": "Normalized Second-order scattering coefficients",
            "description": "S2 = ⟨||X * ψ_{j1,l1}| * ψ_{j2,l2}|⟩ / S1, i.e. successive convolutions at (j1,l1) and (j2,l2), modulus, then averaging. S2 coefficients only for j2>j1.",
        },
    )
    
    data_vars = {"S0": da_S0, "S1": da_S1, "S2": da_S2}

    # =========== RWST ===========    
    if rwst_data is not None:
        # 1st order: 
        for name, long_name in {
            "S1Iso": "RWST first-order isotropic component",
            "S1Aniso": "RWST first-order anisotropic component",
            "ThetaRef1": "RWST first-order reference angle",
        }.items():
            arr = np.asarray(rwst_data.get_coeffs(name)).T # (j1, ...) → (..., j1)
            arr = arr.reshape(nt, npol, len(j1_vals)) # (..., j1) → (nt, npol, j1)
            data_vars[name] = xr.DataArray(
                arr, 
                dims=("tile", "pol", "j1"),
                coords={"tile": coords["tile"], "pol": coords["pol"], "j1": coords["j1"]},
                attrs={"long_name": long_name},
            )

        # 2nd order: 
        for name, long_name in {
            "S2Iso1": "RWST second-order isotropic component (1)",
            "S2Iso2": "RWST second-order isotropic component (2)",
            "S2Aniso1": "RWST second-order anisotropic component (1)",
            "S2Aniso2": "RWST second-order anisotropic component (2)",
            "ThetaRef2": "RWST second-order reference angle",
        }.items():
            arr = np.moveaxis(np.asarray(rwst_data.get_coeffs(name)), -1, 0)[..., 1:] # (j2+1, j1, ...) → (..., j1, j2)
            arr = arr.reshape(nt, npol, len(j1_vals), len(j2_vals)) # (..., j1, j2) → (nt, npol, j1, j2)
            data_vars[name] = xr.DataArray(
                arr,
                dims=("tile", "pol", "j1", "j2"),
                coords={
                    "tile": coords["tile"], "pol": coords["pol"],
                    "j1": coords["j1"], "j2": coords["j2"]
                },
                attrs={"long_name": long_name},
            )

    elif hasattr(wst_data, "rwst_error"):
        # RWST a échoué → créer NaN placeholders

        def _nan3d():
            return xr.DataArray(
                np.full((nt, npol, len(j1_vals)), np.nan),
                dims=("tile", "pol", "j1"),
                coords={"tile": coords["tile"], "pol": coords["pol"], "j1": coords["j1"]},
            )

        def _nan4d():
            return xr.DataArray(
                np.full((nt, npol, len(j1_vals), len(j2_vals)), np.nan),
                dims=("tile", "pol", "j1", "j2"),
                coords={
                    "tile": coords["tile"], "pol": coords["pol"],
                    "j1": coords["j1"], "j2": coords["j2"]
                },
            )

        data_vars.update({
            "S1Iso": _nan3d(),
            "S1Aniso": _nan3d(),
            "ThetaRef1": _nan3d(),
            "S2Iso1": _nan4d(),
            "S2Iso2": _nan4d(),
            "S2Aniso1": _nan4d(),
            "S2Aniso2": _nan4d(),
            "ThetaRef2": _nan4d(),
        })

    ds = xr.Dataset(data_vars, coords=coords)

    # Propage le message d’erreur si présent
    if hasattr(wst_data, "rwst_error"):
        ds.attrs["rwst_error"] = str(wst_data.rwst_error)

    return ds


def compute_scatt_coeffs(
    tiles : xr.Dataset,
    var_name : str,
    res : str,
    tile_size : int,
    #scatt_mode : str,
    norient : int = 8,
) -> xr.Dataset:
    
    #logger.info(f"[Scatt] Starting scattering transform in mode='{scatt_mode}'")

    # get image size 
    sampling_space = int(res[:res.find('m')])
    N = int(tile_size/sampling_space)
    logger.info(f"[Scatt] Tile size in pixels: {N}x{N}")

    # define operators
    M, J = closest_pow2(N)
    logger.info(f"[Scatt] Building scattering operators... (M={M}, J={J}, L={norient})")
    operators = get_operators(M, J, norient)

    # prepare data
    data = center_crop(tiles[var_name], M).values 
    logger.info(f"[Scatt] Data cropped at center.")

    # !!! Need to change the shape to combine the 'pol' dimension !!!
    if 'pol' in tiles.dims:
        npol = tiles.pol.size
        nt = tiles.tile.size
        data = data.reshape(-1,M,M)

    # apply Scattering operators 
    logger.info("[Scatt] Applying scattering operators...")
    scatt_data = apply_operators(data, *operators)
    logger.info("[Scatt] Scattering operators applied successfully.")

    # transform Scattering coefficient matrix to dataset
    scatt_data = coeffs_to_dataset(*scatt_data, tiles_index=tiles.tile, pol_index=tiles.pol)
    logger.info("[Scatt] Converted coefficients to xarray.Dataset.")
    
    return scatt_data