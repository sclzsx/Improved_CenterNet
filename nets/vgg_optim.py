import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .backbone import *

BN_MOMENTUM = 0.1

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PoseResNet(nn.Module):
    def __init__(self, head_conv, num_classes):
        super(PoseResNet, self).__init__()
        self.inplanes = 128
        self.deconv_with_bias = False
        vgg = VGG_MobileLittle_v3()

        self.base = nn.ModuleList(vgg)

        self.up0 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2, 0)

        self.conv0 = nn.Conv2d(128, 128, 1)
        self.conv1 = nn.Conv2d(128, 128, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Conv2d(32, 128, 1)

        if head_conv > 0:
            # heatmap layers
            self.hmap = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=5, groups=head_conv, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, num_classes, kernel_size=1))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=5, groups=head_conv, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 2, kernel_size=1))
            self.w_h_ = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=5, groups=head_conv, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 2, kernel_size=1))
        else:
            # heatmap layers
            self.hmap = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)
            # regression layers
            self.regs = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
            self.w_h_ = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                                             out_channels=planes,
                                             kernel_size=kernel,
                                             stride=2,
                                             padding=padding,
                                             output_padding=output_padding,
                                             bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        for i in range(14):
            x = self.base[i](x)
            if i == 2:
                x1 = x
            if i == 5:
                x2 = x
            if i == 8:
                x3 = x
            if i == 13:
                x4 = x
        # print(x4.shape)
        # print(x3.shape)
        x3 = self.conv1(x3) + self.up1(x4)
        x2 = self.conv2(x2) + self.up2(x3)
        x = self.conv3(x1) + self.up3(x2)
        # print(x.shape)

        # x = self.deconv_layers(x)
        out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
        return out

    def init_weights(self, num_layers):
        for m in self.hmap.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -2.19)
        for m in self.regs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.w_h_.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        # url = model_urls['resnet{}'.format(num_layers)]
        # pretrained_state_dict = model_zoo.load_url(url)
        # print('=> loading pretrained model {}'.format(url))
        # self.load_state_dict(pretrained_state_dict, strict=False)


def get_pose_net(num_layers, head_conv=64, num_classes=80):
    model = PoseResNet(head_conv=head_conv, num_classes=num_classes)
    model.init_weights(num_layers)
    return model


if __name__ == '__main__':
    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        # pass


    net = get_pose_net(num_layers=18, head_conv=0, num_classes=4)
    x = torch.randn(2, 3, 512, 512)

    # for name, module in net.named_children():
    #  x = module(x)
    #  print(name, x.shape)

    from ptflops import get_model_complexity_info

    img_dim = 512
    flops, params = get_model_complexity_info(net, (img_dim, img_dim), as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook)

    # y = net(torch.randn(2, 3, 384, 384))
    y = net(torch.randn(2, 3, 512, 512))

    # print(y.size())
