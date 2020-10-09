
import torch
import torch.nn as nn
import math

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

base_channel = int(64 * 0.25)
def VGG_RFB():
    layers = []
    layers += [BasicConv(3, base_channel, 1, 1, 0)]
    layers += [BasicConv(base_channel, base_channel, kernel_size=3,stride=2, padding=1)] #150 * 150

    layers += [BasicConv(base_channel, base_channel * 2, kernel_size=3,stride=1, padding=1)]
    layers += [BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1)] #75 * 75

    layers += [BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1)] #38 * 38

    layers += [BasicConv(base_channel * 4, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicRFB(base_channel * 8, base_channel * 8, stride = 1, scale=1.0)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1)] # 19 * 19

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8,128,kernel_size=1,stride=1,padding=0)]
    layers += [BasicConv(128,128,kernel_size=3,stride=2, padding=1)] # 10*10

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1)]

    return layers

BN_MOMENTUM = 0.1
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class vgg_hrnet(nn.Module):
    def __init__(self, in_channels):
        super(vgg_hrnet, self).__init__()
        self.inplace = in_channels
        base_channel = self.inplace
        self.conv1 = BasicConv(3, base_channel, 1, 1, 0)
        self.conv2 = BasicConv(base_channel, base_channel, kernel_size=3,stride=2, padding=1) #150*150
        self.conv3 = BasicConv(base_channel, base_channel * 2, kernel_size=3,stride=1, padding=1)
        self.conv4 = BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1) #75*75
        self.conv5 = BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1)
        self.conv6 = BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)
        self.conv7 = BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1) #38*38
        self.conv8 = BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1) #1
        self.conv9 = BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)
        self.conv10 = BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)
        self.conv11 = BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)
        self.conv12 = BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)
        self.branch1 = BasicConv(base_channel * 4, base_channel * 8, stride=2, kernel_size=3, padding=1) #19*19
        self.branch2 = BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1) #1
        self.branch3 = BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1) 
        self.branch4 = BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1) 
        self.branch5 = BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1) 
        self.branch6 = BasicConv(base_channel * 8,base_channel * 16,kernel_size=3,stride=2, padding=1)  #10*10
        self.branch7 = BasicConv(base_channel * 16,base_channel * 16,kernel_size=3,stride=1, padding=1)
        self.branch8 = BasicConv(base_channel * 16,base_channel * 16,kernel_size=3,stride=1, padding=1)

        self.downconv1 = BasicConv(base_channel * 4, base_channel * 8, stride=2, kernel_size=3, padding=1)
        self.downconv2 = BasicConv(base_channel * 4, base_channel * 8, stride=2, kernel_size=3, padding=1)

        self.downconv3 = BasicConv(base_channel * 8, base_channel * 16, stride=2, kernel_size=3, padding=1)

        self.up0 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2, 0)

        self.convUP1 = nn.Conv2d(32, 128, 1)
        self.convUP2 = nn.Conv2d(64, 128, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        up1 = self.convUP1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x) 
        x = self.conv8(x)
        x = self.conv9(x)
        x1 = self.downconv1(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x1_1 = self.downconv2(x)
        

        
        x2 = self.branch1(x)
        x2 = self.branch2(x2)
        x3 = x2+x1
        x3 = self.branch3(x3)
        x3 = self.branch4(x3)
        up0= self.up0(x3)
        s0 = up0 + x 
        x_out0 = self.conv12(s0)

        x4 = self.downconv3(x3)
        x5 = x1_1 + x3
        x_out1 = self.branch5(x5)
        x6 = self.branch6(x2)
        x6 = self.branch7(x6)
        x_out2 = self.branch8(x4+x6)
        x_s1 = x_out1 + self.up1(x_out2)
        x_s2 = self.convUP2(x_out0) + self.up2(x_s1)
        x_out =    up1   + self.up3(x_s2)

        return x_out



class PoseVggNet(nn.Module):
    def __init__(self, head_conv,num_classes):
        super(PoseVggNet, self).__init__()
        self.inplanes = 128
        self.deconv_with_bias = False
        #vgg = VGG_RFB()
        self.base = vgg_hrnet(16)
    
        if head_conv > 0:
      # heatmap layers
            self.hmap = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, num_classes, kernel_size=1))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, 2, kernel_size=1))
            self.w_h_ = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1),
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


            # 3
            # [128, 128]
            # [4, 4, 4]

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):

            kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes,
            #             kernel_size=(3,3), stride=1,
            #             padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv2d(self.inplanes, planes,
                        kernel_size=3, stride=1,
                        padding=3, dilation=3, bias=False)
                # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                        in_channels=planes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x)
        #print(x.shape)
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

def get_pose_net(num_layers, head_conv=64, num_classes=80):
  model = PoseVggNet(head_conv=head_conv, num_classes=num_classes)
  model.init_weights(num_layers)
  return model
