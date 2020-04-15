import torch.nn as nn
import torch
from ResNet import resnet50
from adjust_layer import AdjustAllLayer
from corr import xcorr_depthwise
from Head import CARHead
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        use_layers = {'used_layers': [2, 3, 4]}
        self.backbone = resnet50(**use_layers)
        # Adjust all layers to 256
        self.adj = AdjustAllLayer([512, 1024, 2048], [256, 256, 256])
        self.head = CARHead(in_channels= 256)
    def forward(self, template, search):
        template = self.adj.forward(self.backbone.forward(template))
        search = self.adj.forward(self.backbone.forward(search))
        features = xcorr_depthwise(search[0], template[0])
        for i in range(len(template) - 1):
            feature_new = xcorr_depthwise(search[i + 1], template[i + 1])
            features = torch.cat([features, feature_new], 1)
        Down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
        # In the code, the author use 1*1 transpose convolution...
        features = Down(features)
        logits, mask_reg, centerness = self.head.forward(features)

        return logits, mask_reg, centerness

model = Mymodel()
template = torch.randn(5,3,127,127)
search = torch.randn(5,3,255,255)
cls, mask, centerness = model.forward(template, search)
print('cls shape: {} mask shape: {} centerness shape: {}'.format(cls.shape, mask.shape, centerness.shape))
