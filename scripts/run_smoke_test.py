from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import run_smoke_test
from src.utils.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/smoke_test.yaml")
    print(run_smoke_test(load_config(parser.parse_args().config)))
