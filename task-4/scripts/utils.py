import json
import logging
from pathlib import Path

# ---------------------------------------
# Logging Configuration
# ---------------------------------------

def setup_logger(log_file="../logs/benchmark.log"):

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

# ---------------------------------------
# JSON Export
# ---------------------------------------

def save_json(data, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)