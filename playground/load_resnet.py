import logging

import torch as th

from models.components.backbones.resnet import ResNet

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    resnet18 = ResNet.create_resnet18(pretrained=True)
    resnet50 = ResNet.create_resnet50(pretrained=True)
    resnet50d = ResNet.create_resnet50d(pretrained=True)
    resnext50d_32x4d = ResNet.create_resnext50d_32x4d(
        pretrained=True,
        out_features=[
            "conv1",
            "layer1",
            "layer2",
            "layer3",
            "layer4"
        ]
    )
    seresnext50_32x4d = ResNet.create_seresnext50_32x4d(pretrained=True)
    dummy_out = resnext50d_32x4d(th.randn(1, 3, 224, 224).float())
    print(dummy_out.keys())
