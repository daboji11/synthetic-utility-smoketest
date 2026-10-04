"""Camera identity, gradient routing, and checkpoint compatibility checks."""
import torch
import pytest

pytest.importorskip("lerobot")
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from src.policies.act_camera_id.configuration_act_camera_id import ACTCameraIdConfig
from src.policies.act_camera_id.modeling_act_camera_id import ACTCameraIdPolicy


def make_policy():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    config = ACTCameraIdConfig(
        device="cpu", pretrained_backbone_weights=None, use_vae=False,
        dim_model=64, n_heads=4, dim_feedforward=128, n_encoder_layers=1,
        n_decoder_layers=1, chunk_size=4, n_action_steps=4, dropout=0.0,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))
           for key in config.camera_order},
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))}
    assert get_policy_class("act_camera_id") is ACTCameraIdPolicy
    return ACTCameraIdPolicy(config).eval()


def make_batch(policy):
    return {
        "observation.state": torch.randn(1, 16),
        **{key: torch.rand(1, 3, 128, 128, requires_grad=True)
           for key in policy.config.camera_order},
        "action": torch.randn(1, 4, 12),
        "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
    }


def test_camera_identity_and_all_view_gradients():
    policy = make_policy()
    batch = make_batch(policy)
    loss, _ = policy.forward(batch)
    loss.backward()
    assert policy.model.camera_identity_embed.weight.grad.abs().sum() > 0
    for key in policy.config.camera_order:
        assert batch[key].grad is not None and batch[key].grad.abs().sum() > 0
    baseline = policy.predict_action_chunk(batch)
    changed = dict(batch)
    left, right = policy.config.camera_order[:2]
    changed[left], changed[right] = batch[right], batch[left]
    swapped = policy.predict_action_chunk(changed)
    assert (baseline - swapped).abs().max() > 1e-6
    # Without identities, the installed ACT architecture recovers permutation invariance.
    with torch.no_grad():
        policy.model.camera_identity_embed.weight.zero_()
    plain = policy.predict_action_chunk(batch)
    plain_swapped = policy.predict_action_chunk(changed)
    torch.testing.assert_close(plain, plain_swapped, rtol=1e-5, atol=1e-6)


def test_saved_camera_identity_and_processors_roundtrip(tmp_path):
    policy = make_policy()
    batch = make_batch(policy)
    stats = {
        "observation.state": {"mean": torch.zeros(16), "std": torch.ones(16)},
        "action": {"mean": torch.zeros(12), "std": torch.ones(12)},
        **{key: {"mean": torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1),
                  "std": torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)}
           for key in policy.config.camera_order},
    }
    pre, post = make_pre_post_processors(policy.config, dataset_stats=stats)
    expected = post(policy.predict_action_chunk(pre(dict(batch))))
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    restored = ACTCameraIdPolicy.from_pretrained(tmp_path, local_files_only=True, strict=True)
    restored_pre, restored_post = make_pre_post_processors(restored.config, pretrained_path=str(tmp_path))
    actual = restored_post(restored.predict_action_chunk(restored_pre(dict(batch))))
    assert restored.config.type == "act_camera_id"
    assert restored.config.camera_order == policy.config.camera_order
    torch.testing.assert_close(restored.model.camera_identity_embed.weight,
                               policy.model.camera_identity_embed.weight)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
