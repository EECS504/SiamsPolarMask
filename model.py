import torch.nn as nn
import torch
from ResNet import resnet50
from adjust_layer import AdjustAllLayer
from corr import xcorr_depthwise
from Head import CARHead
from torch.nn import init
from loss_fun import My_loss
def init_weights(net, init_type='normal', mean = 0.0, init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, mean)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, mean)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        use_layers = {'used_layers': [2, 3, 4]}
        self.backbone = resnet50(**use_layers)

        pretrained_resnet50 = torch.load('pretrained_resnet50.pt')
        assert len(self.backbone.state_dict()) == len(pretrained_resnet50)
        pretrained_Weights = []
        for key in pretrained_resnet50:
            pretrained_Weights.append(pretrained_resnet50[key])
        for i, key in enumerate(self.backbone.state_dict()):
            self.backbone.state_dict()[key] = pretrained_Weights[i]
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Adjust all layers to 256
        self.adj = AdjustAllLayer([512, 1024, 2048], [256, 256, 256])
        self.head = CARHead(in_channels= 256)
        self.Down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
        init_weights(self.adj, init_type='normal', mean=0, init_gain=0.02)
        init_weights(self.head, init_type='normal', mean=0, init_gain=0.02)
        init_weights(self.Down, init_type='normal', mean=0, init_gain=0.02)
    def template(self, z):
        self.zf = self.adj.forward(self.backbone.forward(z))
    def track(self, x):
        xf = self.adj.forward(self.backbone.forward(x))
        features = xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            feature_new = xcorr_depthwise(xf[i + 1], self.zf[i + 1])
            features = torch.cat([features, feature_new], 1)
        features = self.Down(features)
        logits, mask_reg, centerness = self.head.forward(features)

        return logits, mask_reg, centerness
    def forward(self, template, search):
        template = self.adj.forward(self.backbone.forward(template))
        search = self.adj.forward(self.backbone.forward(search))
        features = xcorr_depthwise(search[0], template[0])
        for i in range(len(template) - 1):
            feature_new = xcorr_depthwise(search[i + 1], template[i + 1])
            features = torch.cat([features, feature_new], 1)

        features = self.Down(features)
        logits, mask_reg, centerness = self.head.forward(features)

        return logits, mask_reg, centerness

 # model = Mymodel()import torch.nn as nn
# import torch
# from ResNet import resnet50
# from adjust_layer import AdjustAllLayer
# from corr import xcorr_depthwise
# from Head import CARHead
# from torch.nn import init
# from loss_fun import My_loss
# def init_weights(net, init_type='normal', mean = 0.0, init_gain=0.02):
#     """Initialize network weights.
#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
#     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
#     work better for some applications. Feel free to try yourself.
#     """
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, mean, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, mean)
#         elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, mean)
#
#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>
#
# class Mymodel(nn.Module):
#     def __init__(self):
#         super(Mymodel, self).__init__()
#         use_layers = {'used_layers': [2, 3, 4]}
#         self.backbone = resnet50(**use_layers)
#
#         pretrained_resnet50 = torch.load('pretrained_resnet50.pt')
#         assert len(self.backbone.state_dict()) == len(pretrained_resnet50)
#         pretrained_Weights = []
#         for key in pretrained_resnet50:
#             pretrained_Weights.append(pretrained_resnet50[key])
#         for i, key in enumerate(self.backbone.state_dict()):
#             self.backbone.state_dict()[key] = pretrained_Weights[i]
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#         # Adjust all layers to 256
#         self.adj = AdjustAllLayer([512, 1024, 2048], [256, 256, 256])
#         self.head = CARHead(in_channels= 256)
#         self.Down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
#         init_weights(self.adj, init_type='normal', mean=0, init_gain=0.02)
#         init_weights(self.head, init_type='normal', mean=0, init_gain=0.02)
#         init_weights(self.Down, init_type='normal', mean=0, init_gain=0.02)
#     def template(self, z):
#         self.zf = self.adj.forward(self.backbone.forward(z))
#     def track(self, x):
#         xf = self.adj.forward(self.backbone.forward(x))
#         features = xcorr_depthwise(xf[0], self.zf[0])
#         for i in range(len(xf) - 1):
#             feature_new = xcorr_depthwise(xf[i + 1], self.zf[i + 1])
#             features = torch.cat([features, feature_new], 1)
#         features = self.Down(features)
#         logits, mask_reg, centerness = self.head.forward(features)
#
#         return logits, mask_reg, centerness
#     def forward(self, template, search):
#         template = self.adj.forward(self.backbone.forward(template))
#         search = self.adj.forward(self.backbone.forward(search))
#         features = xcorr_depthwise(search[0], template[0])
#         for i in range(len(template) - 1):
#             feature_new = xcorr_depthwise(search[i + 1], template[i + 1])
#             features = torch.cat([features, feature_new], 1)
#
#         features = self.Down(features)
#         logits, mask_reg, centerness = self.head.forward(features)
#
#         return logits, mask_reg, centerness
#
#  # model = Mymodel()
#  # template = torch.randn(5,3,127,127)
#  # search = torch.randn(5,3,255,255)
#  # GT_cls = torch.zeros(5,25,25, dtype= torch.long)
#  # GT_cls[2,:,:] = 1
#  # GT_mask = torch.ones(5,25,25,36)
#  # cls, mask, centerness = model.forward(template, search)
# #
# #
# # print('cls shape: {} mask shape: {} centerness shape: {}'.format(cls.shape, mask.shape, centerness.shape))
# # criterion = My_loss()
# # cls_loss, reg_loss, centerness_loss = criterion.forward(cls, mask, centerness, GT_cls, GT_mask)
# # print('cls loss: {} mask loss: {} centerness loss: {}'.format(cls_loss.item(), reg_loss.item(), centerness_loss.item()))
 # template = torch.randn(5,3,127,127)
 # search = torch.randn(5,3,255,255)
 # GT_cls = torch.zeros(5,25,25, dtype= torch.long)
 # GT_cls[2,:,:] = 1
 # GT_mask = torch.ones(5,25,25,36)
 # cls, mask, centerness = model.forward(template, search)
#
#
# print('cls shape: {} mask shape: {} centerness shape: {}'.format(cls.shape, mask.shape, centerness.shape))
# criterion = My_loss()
# cls_loss, reg_loss, centerness_loss = criterion.forward(cls, mask, centerness, GT_cls, GT_mask)
# print('cls loss: {} mask loss: {} centerness loss: {}'.format(cls_loss.item(), reg_loss.item(), centerness_loss.item()))
