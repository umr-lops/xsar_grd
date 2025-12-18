import csv
import argparse
import resource
from pathlib import Path
from xsargrd import generate_l1c, load_config

if __name__ == "__main__":

    # --- Parse args ---
    parser = argparse.ArgumentParser(
        description="Produce L1C from a listing of L1B SAFE products."
    )
    parser.add_argument(
        "--path_file",
        required=True,
        help="Path to the L1B SAFE listing (.txt)",
    )
    parser.add_argument(
        "--config_id",
        default="A01",
        help="L1C configuration identifier",
    )
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--report_dir",
        default=script_dir / "l1c_production_reports",
        help="Directory to store per-SAFE production CSV reports",
    )
    args = parser.parse_args()

    # --- Load configuration ---
    config = load_config("l1c")
    cfg = config[args.config_id]
    ancillary_list = cfg["auxilliary_dataset"]
    ancillary_names = list(ancillary_list.keys())

    # --- Fixed resolution for now ---
    res = "100m"

    # --- Create report directory ---
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV report path ---
    safe_name = Path(args.path_file).name.replace("/", "_")
    csv_path = report_dir / f"l1c_{safe_name}.csv"

    # --- Generate L1C ---
    try:
        _, flags = generate_l1c(
            fullpath_l1b=args.path_file,
            res=res,
            ancillary_list=ancillary_list,
            save=True,
            overwrite=True,
        )
        comment = "OK"

    except Exception as e:
        flags = {}
        comment = f"FAIL: {e}"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "safe_path", "res", "config_id",
            *ancillary_names, "comment"
        ])
        writer.writerow([
            args.path_file,
            res,
            args.config_id,
            *[flags.get(n, False) for n in ancillary_names],
            comment,
        ])

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak memory usage: {peak_mb:.1f} MB")