import csv
import argparse
import resource
from pathlib import Path
from xsargrd import generate_l1b, load_config
script_dir = Path(__file__).resolve().parent

if __name__ == "__main__":

    # --- Parse args ---
    parser = argparse.ArgumentParser(
        description="Produce L1B products from a listing file."
    )
    parser.add_argument(
        "--path_file",
        required=True,
        help="Path to the input listing (.txt)",
    )
    parser.add_argument(
        "--config_id",
        default="J01",
        help="Configuration ID to use (default: J01)",
    )
    parser.add_argument(
        "--report_dir",
        default=script_dir / "l1c_production_reports",
        help="Directory to store per-SAFE production CSV reports",
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config()
    if args.config_id not in config:
        raise ValueError(
            f"Unknown config_id '{args.config_id}'. "
            f"Available configs: {list(config.keys())}"
        )

    c = config[args.config_id]

    # --- Create report directory ---
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV report path ---
    safe_name = Path(args.path_file).name.replace("/", "_")
    csv_path = report_dir / f"l1b_{safe_name}.csv"

    # --- Generate L1B ---
    try:
        generate_l1b(
            fullpath=args.path_file,
            dirout=c["dirout"],
            res=c["res"],
            tile_size=c["tile_size"],
            periodo_width=c["periodo_width"],
            periodo_overlap=c["periodo_overlap"],
            lowpass_width=c["lowpass_width"],
            scatt_mode=c["scatt_mode"],
            norient=c["norient"],
            save=True,
            config_id=args.config_id
        )
        comment = "OK"

    except Exception as e:
        flags = {}
        comment = f"FAIL: {e}"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "safe_path",
            "res",
            "config_id",
            "comment"
        ])
        writer.writerow([
            args.path_file,
            c['res'],
            args.config_id,
            comment,
        ])

    # --- Run info ---
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"Peak memory usage: {peak_mb:.1f} MB")
