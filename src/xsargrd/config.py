# xsargrd/config.py
from pathlib import Path
import yaml
import xsargrd

def load_config(name: str) -> dict:
    """
    Load a YAML configuration file shipped with xsargrd.

    Parameters
    ----------
    name : str
        Name of the configuration file. ('l1b' or 'l1c')
    
    Returns
    -------
    config : dict
        Configuration parameters loaded from the YAML file.
    """
    config_path = Path(xsargrd.__file__).parent / f"{name}_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
