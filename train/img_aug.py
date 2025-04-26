import random
import torch
import torch.nn as nn
import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F 

class ImageTransformConfig:
    def __init__(self, weight, type, kwargs):
        self.weight = weight
        self.type = type
        self.kwargs = kwargs
        
# transform_configs = {
#     "brightness": ImageTransformConfig(1.0, "ColorJitter", {"brightness": (0.8, 1.2)}),
#     "contrast": ImageTransformConfig(1.0, "ColorJitter", {"contrast": (0.8, 1.2)}),
#     "saturation": ImageTransformConfig(1.0, "ColorJitter", {"saturation": (0.5, 1.5)}),
#     "hue": ImageTransformConfig(1.0, "ColorJitter", {"hue": (-0.05, 0.05)}),
#     "sharpness": ImageTransformConfig(1.0, "SharpnessJitter", {"sharpness": (0.5, 1.5)}),
#     "crop_resize": ImageTransformConfig(1.0, "RandomResizedCrop", {"size": (256, 256), "scale": (0.9, 0.95), "ratio": (1.0, 1.0)}),
#     "rotate": ImageTransformConfig(1.0, "RandomRotate", {"degrees": (-5, 5)}),
# }

class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpnesss values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)

class RandomSubsetWeightedTransform(nn.Module):
    def __init__(
        self,
        transform_configs: dict[str, ImageTransformConfig],
        n_subset: int = 3,
        random_order: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(transform_configs, dict):
            raise TypeError("transform_configs should be a dict of {name: ImageTransformConfig}")

        if not (1 <= n_subset <= len(transform_configs)):
            raise ValueError(f"n_subset should be in [1, {len(transform_configs)}]")

        self.transform_configs = transform_configs
        self.n_subset = n_subset
        self.random_order = random_order

        self.names = list(transform_configs.keys())
        self.configs = list(transform_configs.values())
        weights = torch.tensor([cfg.weight for cfg in self.configs], dtype=torch.float)
        self.p = weights / weights.sum()  # 归一化权重

        self.transforms = [self.build_transform(cfg) for cfg in self.configs]

    def build_transform(self, cfg: ImageTransformConfig):
        if cfg.type == "ColorJitter":
            return v2.ColorJitter(**cfg.kwargs)
        elif cfg.type == "SharpnessJitter":
            return SharpnessJitter(**cfg.kwargs)
        elif cfg.type == "RandomResizedCrop":
            return v2.RandomResizedCrop(**cfg.kwargs)
        elif cfg.type == "RandomRotate":
            return v2.RandomRotation(**cfg.kwargs)
        else:
            raise ValueError(f"Unknown transform type: {cfg.type}")

    def forward(self, inputs):

        selected_indices = torch.multinomial(self.p, self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        selected_transforms = [self.transforms[i] for i in selected_indices]

        outputs = inputs
        for transform in selected_transforms:
            outputs = transform(outputs)

        return outputs



    