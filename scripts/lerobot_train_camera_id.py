"""Register the repository ACT extension, then enter the standard LeRobot trainer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policies.act_camera_id.configuration_act_camera_id import ACTCameraIdConfig
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
