import argparse
import resource
from xsargrd import generate_l1c

if __name__ == "__main__":

    # --- Parse args ---
    parser = argparse.ArgumentParser(description="Produce L1C from a path file of L1B product.")
    parser.add_argument("--path_file", required=True, help="Path to the L1B listing in .txt")
    args = parser.parse_args()

    # --- Process ---
    generate_l1c(args.path_file)

    # --- Run info ---
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"peak memory usage: {peak_mb:.1f} MB")