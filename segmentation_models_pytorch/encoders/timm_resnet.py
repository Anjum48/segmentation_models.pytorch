from ._base import EncoderMixin
from timm.models.resnet import ResNet, Bottleneck, BasicBlock
import torch.nn as nn


class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.global_pool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.act1),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)


resnet_weights = {
    "resnet34d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth",
    },
    "resnet50d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth",
    },
    "resnet101d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth",
    },
    "resnet200d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth",
    },
    "seresnet152d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet152d_ra2-04464dd2.pth",
    },
}

pretrained_settings = {}
for model_name, sources in resnet_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }


timm_resnet_encoders = {
    "resnet34d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet34d"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
        },
    },
    "resnet50d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet50d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
        },
    },
    "resnet101d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet101d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
        },
    },
    "resnet200d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet200d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 24, 36, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
        },
    },
    "seresnet152d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["seresnet152d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "block_args": dict(attn_layer="se"),
        },
    },
}
