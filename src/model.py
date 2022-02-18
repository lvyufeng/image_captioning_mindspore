import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from .backbone import resnet101
from .layers import AdaptiveAvgPool2d

class Encoder(nn.Cell):
    def __init__(self, encoded_image_size=14):
        super().__init__()
        self.encoded_image_size = encoded_image_size
        resnet = resnet101(pretrained=True)

        modules = list(resnet.cells())[:-2]
        self.resnet = nn.SequentialCell(*modules)

        self.adaptive_pool = AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def construct(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.transpose(0, 2, 3, 1)
        return out