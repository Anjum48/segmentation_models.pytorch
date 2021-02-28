from ._base import EncoderMixin
from timm.models.nfnet import NormFreeNet, model_cfgs
import torch.nn as nn


class NFNetEncoder(NormFreeNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.head.fc
        del self.head.global_pool

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
        state_dict.pop("head.fc.bias")
        state_dict.pop("head.fc.weight")
        super().load_state_dict(state_dict, **kwargs)


nfnet_weights = {
    "dm_nfnet_f1": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f1-fc540f82.pth",
    },
}

pretrained_settings = {}
for model_name, sources in nfnet_weights.items():
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


timm_nfnet_encoders = {
    "dm_nfnet_f1": {
        "encoder": NFNetEncoder,
        "pretrained_settings": pretrained_settings["dm_nfnet_f1"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1536, 3072),
            "cfg": model_cfgs["dm_nfnet_f1"],
        },
    },
}
