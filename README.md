# xsargrd

**xsargrd** is a Python library designed to process SAR images acquired in **GRD mode** and to generate intermediate and analysis-ready products for oceanographic and geophysical applications.

The library focuses primarily on **Sentinel-1**, with planned or partial support for **RCM (RADARSAT Constellation Mission)** and **RADARSAT-2 (RS2)** data.

---

## 🚀 Main Features

- Ingestion of SAR GRD products
- Generation of **Level-1B (L1B)** products:
  - Tiling of SAR images
  - Spectral representations
  - Statistical descriptors derived from **Scattering Transform** representations
  - Multi-resolution processing
- Generation of **Level-1C (L1C)** products:
  - Ingestion and colocation of ancillary geophysical datasets
  - Raster-to-tile interpolation
  - Homogenized, analysis-ready datasets
- Modular and configurable processing pipelines
- Designed for **large-scale production** as well as **interactive scientific analysis**

---

## 📦 Processing Levels

### Level-1B (L1B)

L1B products are derived from SAR GRD images and include:
- Spatial tiling of the SAR image
- Computation of spectral quantities
- Computation of statistical features based on **Scattering Transform representations**
- Multi-scale and multi-orientation descriptors
- Products stored as NetCDF datasets

These products are intended to capture both the spectral content and the multi-scale statistical structure of SAR backscatter.

---

### Level-1C (L1C)

L1C products extend L1B datasets by:
- Adding colocated ancillary fields (e.g. wind, wave or model-based products)
- Interpolating raster datasets onto SAR tile centers
- Providing consistent geophysical context for each SAR tile

L1C products are designed to be directly usable for statistical analysis, machine learning or physical interpretation.

---

## 🛰️ Supported Missions

- **Sentinel-1** (primary target, fully supported)
- **RCM (RADARSAT Constellation Mission)** (partial / under development)
- **RADARSAT-2 (RS2)** (partial / under development)

Some missions or configurations may raise `NotImplementedError` where support is incomplete.

---

## 🗂️ Package Structure

xsargrd/
├── l1b/
│ ├── generate.py # L1B generation pipeline
│ ├── tools.py # L1B helper functions
│ ├── cwave.py
│ ├── scatt.py
│ └── spectra.py
│
├── l1c/
│ ├── generate.py # L1C generation pipeline
│ ├── pipeline.py # L1C enrichment logic
│ ├── ancillaries.py # Ancillary handling
│ ├── coloc.py # Raster-to-tile colocation
│ ├── raster_readers.py
│ └── tools.py
│
├── config.py # Configuration loading utilities
└── init.py



---
## 📦 Installation

### Using pip (editable mode, recommended for development)

```bash
git clone https://github.com/<your-username>/xsargrd.git
cd xsargrd
pip install -e .
```

---
## ⚙️ Configuration System

Processing parameters are defined using YAML configuration files (e.g. `l1b_config.yaml`, `l1c_config.yaml`).

This allows:
- Reproducible large-scale production
- Easy tracking of processing configurations
- Flexible usage outside production (e.g. notebooks, experiments)

---

## 🧪 Example Usage

```python
from xsargrd import generate_l1b, load_config

config = load_config("l1b")
c = config["A01"]

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
