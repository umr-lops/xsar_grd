import argparse
import resource
from xsargrd import generate_l1b, load_config

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
        default="A01",
        help="Configuration ID to use (default: A01)",
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

    # --- Process ---
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

    # --- Run info ---
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"Peak memory usage: {peak_mb:.1f} MB")
