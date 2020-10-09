import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from nets.modules import BasicConv
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class CEM(nn.Module):
    def __init__(self, inplanes,out_planes):
        super(CEM, self).__init__()
        planes = out_planes
        self.branch1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,dilation=1, padding=1, bias=False),
            #nn.Conv2d(planes, planes, kernel_size=3, stride=1,dilation=3, padding=3, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, planes, 4, 2, 1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            #nn.Conv2d(planes, planes, kernel_size=3, stride=1,dilation=2, padding=2, bias=False),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, planes, 4, 2, 1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(inplanes, planes)
        self.fc = nn.Conv2d(
            inplanes, planes, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        #self.relu = nn.ReLU(inplace=True)

        self.branch3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,dilation=3, padding=3, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.ConvTranspose2d(planes, planes, 4, 2, 1)
        )

    def forward(self, x):
        #x1, x2, x3 = x.split(x.size(1) // 3, 1)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.avg_pool(x)
        #x3 = x3.view(x3.shape[0], x3.shape[1])
        x3 = self.fc(x3)
        x3 = self.bn(x3)
        #x3 = x3.view(x3.shape[0], x3.shape[1], 1, 1)

        x4 = self.branch3(x)

        x2 = x2 * x3
        x1 = x1 + x2
        #x1 = self.relu(x1)
        x = torch.cat([x1, x4], dim=1)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 5, stride, 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 5, stride, 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, head_conv,num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = 160
        self.inplanes = output_channel
        self.dcn_layer1 = CEM(self.inplanes,32)
        self.inplanes = 64
        self.dcn_layer2 = CEM(self.inplanes,32)
        self.inplanes = 64
        self.dcn_layer3 = CEM(self.inplanes,32)


        if head_conv > 0:
            self.hmap = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=5, groups=head_conv, padding=2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, num_classes, kernel_size=1))
            self.hmap[-1].bias.data.fill_(-2.19)
            self.regs = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=5, groups=head_conv,padding=2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, 2, kernel_size=1))
            self.w_h_ = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=5,groups=head_conv, padding=2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, 2, kernel_size=1))
        else:
            self.hmap = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
            self.regs = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
            self.w_h_ = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.dcn_layer1(x)
        #print(x.shape)
        x = self.dcn_layer2(x)
        x = self.dcn_layer3(x)
        out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
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
            #elif isinstance(m, nn.Linear):
            #    m.weight.data.normal_(0, 0.01)
            #    m.bias.data.zero_()

def get_pose_net(num_layers=18, head_conv=64, num_classes=20):

  model = MobileNetV2(head_conv=head_conv, num_classes=num_classes,width_mult=0.5)
  return model