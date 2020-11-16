import logging

from models.components.backbones.resnet import ResNet

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    resnet18 = ResNet.create_resnet18(pretrained=True)
    resnet50 = ResNet.create_resnet50(pretrained=True)
    resnet50d = ResNet.create_resnet50d(pretrained=True)
    resnext50d_32x4d = ResNet.create_resnext50d_32x4d(pretrained=True)
    seresnext50_32x4d = ResNet.create_seresnext50_32x4d(pretrained=True)
