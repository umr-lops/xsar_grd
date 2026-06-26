import csv
import logging
import xarray as xr
import argparse
import resource
from pathlib import Path
from xsargrd.l1c import sarwind

# --- Logger setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # --- Parse args ---
    parser = argparse.ArgumentParser(description="Add SAR wind statistics to a wave L1C SAFE")
    parser.add_argument("path_l1c_wave", help="Path to the .nc file found inside the L1C SAFE")
    parser.add_argument("path_l2m_wind", help="Path to the L2M SAR wind .nc file")
    parser.add_argument("--output_dir", default="/home1/datahome/egauvrit/Libraries/xsar_grd/scripts/add_sarwind_reports", help="Directory to save the per-task report")
    args = parser.parse_args()

    path_l1c_wave = Path(args.path_l1c_wave)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Identify SAFE name for reporting ---
    safe_name = path_l1c_wave.parent.parent.name

    logger.info(f"Starting processing for {safe_name}")

    # --- Output netcdf: same directory as the L1C wave file, with '_sarwind' suffix ---
    out_path = path_l1c_wave.with_name(path_l1c_wave.stem + "_sarwind" + path_l1c_wave.suffix)

    status = "OK"
    error_msg = ""

    try:
        # --- Open datasets ---
        logger.info(f"Opening datasets: {path_l1c_wave.name}, {Path(args.path_l2m_wind).name}")
        wave_ds = xr.open_dataset(args.path_l1c_wave)
        wind_ds = xr.open_dataset(args.path_l2m_wind)
        logger.info("ncfiles open !")

        # --- Add wind stats ---
        ds_out = sarwind.extract_wind_stats_per_tile(wind_ds, wave_ds)
        logger.info("SAR wind stats added !")

        # --- Save netcdf next to the source L1C wave file ---
        ds_out.to_netcdf(out_path)
        logger.info(f"Saved output to {out_path}")

    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        logger.error(f"Failed processing {safe_name}: {error_msg}")

    # --- Write a small per-task CSV report (status, errors, memory) ---
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report_path = output_dir / f"{safe_name}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['safe_name', 'path_l1c_wave', 'path_l2m_wind', 'output_path', 'status', 'error', 'peak_memory_mb'])
        writer.writerow([safe_name, str(path_l1c_wave), args.path_l2m_wind, str(out_path), status, error_msg, f"{peak_mb:.1f}"])

    logger.info(f"[{status}] {safe_name} -> {out_path} - peak memory usage: {peak_mb:.1f} MB")