"""Reuse exactly the standard ACT observation and action processors."""
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_act_camera_id_pre_post_processors(config, dataset_stats=None):
    return make_act_pre_post_processors(config, dataset_stats)
