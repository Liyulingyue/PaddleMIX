# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

""" MiniCPMV model configuration"""
from typing import Union

from paddlenlp.transformers import PretrainedConfig, Qwen2Config

from paddlemix.utils.log import logger

from .modeling_navit_siglip import SigLipVisionConfig


class MiniCPMVSliceConfig(PretrainedConfig):
    model_type = "minicpmv"

    def __init__(self, patch_size=14, max_slice_nums=9, scale_resolution=448, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == "minicpmv":
            config_dict = config_dict["slice_config"]
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        return cls.from_dict(config_dict, **kwargs)


class MiniCPMVConfig(Qwen2Config):
    model_type = "minicpmv"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 980,
        "intermediate_size": 4304,
        "model_type": "siglip",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    def __init__(
        self,
        use_cache=True,
        query_num=64,
        image_size=448,
        drop_vision_last_layer=True,
        batch_vision_input=True,
        slice_config=None,
        vision_config=None,
        use_image_id=True,
        **kwargs
    ):
        self.use_cache = use_cache
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        if slice_config is None:
            self.slice_config = MiniCPMVSliceConfig(max_slice_nums=1)
        else:
            self.slice_config = MiniCPMVSliceConfig(**slice_config)
        self.slice_mode = True
        if vision_config is None:
            self.vision_config = SigLipVisionConfig(**self.default_vision_config)
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = SigLipVisionConfig(**vision_config)
        elif isinstance(vision_config, SigLipVisionConfig):
            self.vision_config = vision_config
        self.patch_size = self.vision_config.patch_size
        super().__init__(**kwargs)
