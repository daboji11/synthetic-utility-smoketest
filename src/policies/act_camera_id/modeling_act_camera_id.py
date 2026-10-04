"""Extend installed LeRobot ACT without modifying its package or action path."""
import torch
from torch import nn
from lerobot.policies.act.modeling_act import ACTPolicy

from .configuration_act_camera_id import ACTCameraIdConfig


class ACTCameraIdPolicy(ACTPolicy):
    config_class = ACTCameraIdConfig
    name = "act_camera_id"

    def __init__(self, config: ACTCameraIdConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Keep this module inside model so it is optimized and saved with normal ACT weights.
        self.model.camera_identity_embed = nn.Embedding(len(config.camera_order), config.dim_model)
        nn.init.normal_(self.model.camera_identity_embed.weight,
                        mean=0.0, std=config.camera_embedding_init_std)
        # ACT builds [latent, state?, environment_state?, camera0 pixels, camera1 pixels, ...].
        # Add camera identity to the existing visual positional encoding only.
        # The hook is recreated by this constructor on checkpoint reload.
        self.model.encoder.register_forward_pre_hook(self._add_camera_identity, with_kwargs=True)

    def _add_camera_identity(self, module, inputs, kwargs):
        tokens = inputs[0]
        positions = kwargs.get("pos_embed")
        if positions is None:
            raise ValueError("Installed ACT encoder no longer passes pos_embed by keyword")
        prefix = 1 + int(bool(self.config.robot_state_feature)) + int(bool(self.config.env_state_feature))
        visual_count = tokens.shape[0] - prefix
        cameras = len(self.config.camera_order)
        if visual_count <= 0 or visual_count % cameras:
            raise ValueError("ACT visual token count cannot be assigned to saved cameras")
        pixels_per_camera = visual_count // cameras
        ids = torch.arange(cameras, device=tokens.device).repeat_interleave(pixels_per_camera)
        visual_identity = self.model.camera_identity_embed(ids).to(dtype=positions.dtype).unsqueeze(1)
        identity = torch.cat((positions.new_zeros((prefix, 1, positions.shape[-1])),
                              visual_identity), dim=0)
        updated = dict(kwargs)
        updated["pos_embed"] = positions + identity
        return inputs, updated
