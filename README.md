# xsargrd

**xsargrd** is a Python library designed to process SAR images acquired in **GRD mode** and to generate intermediate and analysis-ready products for oceanographic and geophysical applications.

The library is applicable to **Sentinel-1**, **RCM (RADARSAT Constellation Mission)** and **RADARSAT-2 (RS2)** SAR missions, with Sentinel-1 currently being the most maturely supported.

---

## 🚀 Features & Processing Levels

### Level-1B (L1B)

L1B products are derived from SAR GRD images and provide a physically and statistically rich description of the SAR signal.  
They include:

- Spatial tiling of SAR GRD images
- Computation of **power spectra**
- Estimation of **C-wave parameters**
- Computation of **Scattering Transform coefficients**:
  - Multi-scale
  - Multi-orientation
- Storage as self-described **NetCDF datasets**

L1B products are designed to capture both the spectral content and the multi-scale statistical structure of the SAR signal.

### Level-1C (L1C)

L1C products extend L1B datasets by adding geophysical context through ancillary data.  
They include:

- Ingestion and colocation of external ancillary datasets (e.g. wind & wave products)
- Raster-to-tile interpolation onto SAR tile centers
- Generation of homogeneous, analysis-ready datasets

L1C products are intended for direct use in statistical analysis, machine learning workflows, or physical interpretation.

---

## 🛰️ Supported Missions

- **Sentinel-1** (primary target, fully supported)
- **RCM (RADARSAT Constellation Mission)** (supported, some features under development)
- **RADARSAT-2 (RS2)** (supported, some features under development)

Some mission-specific configurations may raise `NotImplementedError` where support is incomplete.

---

## 📦 Installation

### Using pip (editable mode, recommended for development)

```bash
git clone https://github.com/Egauvrit/xsar_grd.git
cd xsar_grd
pip install -e .
```

---

## ⚙️ Example Usage 

### Configuration System

Processing parameters are defined using YAML configuration files (e.g. `l1b_config.yaml`, `l1c_config.yaml`).

### Generate a Level-1B product

```python
from xsargrd import generate_l1b, load_config

# --- load L1B configuration ---
config_id = "J01"
c = load_config()[config_id]

# --- produce L1B ---
generate_l1b(
    fullpath="/path/to/SAR/GRD/product",
    dirout=c["dirout"],
    res=c["res"],
    tile_size=c["tile_size"],
    periodo_width=c["periodo_width"],
    periodo_overlap=c["periodo_overlap"],
    lowpass_width=c["lowpass_width"],
    scatt_mode=c["scatt_mode"],
    norient=c["norient"],
)
```

### Generate a Level-1C product from L1B

```python
from xsargrd import generate_l1c, load_config

# --- load L1C configuration ---
config_id = "J01"
c = load_config()[config_id]

# --- produce L1C ---
generate_l1c(
    fullpath_l1b="/path/to/L1B/product",
    res=c["res"],
    ancillary_list=c["ancillary_list"],
)
```

