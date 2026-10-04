"""ACT config retaining the ordinary ACT hyperparameters plus camera identity."""
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_camera_id")
@dataclass
class ACTCameraIdConfig(ACTConfig):
    camera_order: list[str] = field(default_factory=lambda: [
        "observation.images.robot0_agentview_left_image",
        "observation.images.robot0_agentview_right_image",
        "observation.images.robot0_eye_in_hand_image",
    ])
    camera_embedding_init_std: float = 0.02

    def validate_features(self):
        super().validate_features()
        if list(self.image_features) != self.camera_order:
            raise ValueError("ACT camera input order must match saved camera_order")
        if len(set(self.camera_order)) != len(self.camera_order) or len(self.camera_order) < 2:
            raise ValueError("Camera identity requires distinct multiple camera keys")
        if self.camera_embedding_init_std <= 0:
            raise ValueError("Camera embedding initialization std must be positive")
