import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class CBAModule(nn.Module):
    def __init__(self, in_channels, out_channels=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBAModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Up Sample Module
class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2,  bias=False, mode="UCBA"):
        super(UpModule, self).__init__()
        self.mode = mode

        if self.mode == "UCBA":
            # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
            self.conv = CBAModule(in_channels, out_channels, 3, padding=1, bias=bias)
        elif self.mode == "DeconvBN":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.mode == "DeCBA":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.conv = CBAModule(out_channels, out_channels, 3, padding=1, bias=bias)
        else:
            raise RuntimeError("Unsupport mode: {}".format(model))
    
    def forward(self, x):
        if self.mode == "UCBA":
            return self.conv(self.up(x))
        elif self.mode == "DeconvBN":
            return F.relu(self.bn(self.dconv(x)))
        elif self.mode == "DeCBA":
            return self.conv(self.dconv(x))


class PoseResNet(nn.Module):
  def __init__(self, block, layers, head_conv, num_classes):
    super(PoseResNet, self).__init__()
    self.inplanes = 32
    self.deconv_with_bias = False
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 32, layers[0])
    self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

    # used for deconv layers
    self.deconv_layers = self._make_deconv_layer(3, [64, 64, 128], [4, 4, 4])
    # self.final_layer = []

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

    # self.final_layer = nn.ModuleList(self.final_layer)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                           kernel_size=1, stride=stride, bias=False),
                                 nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

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
      '''layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                                       out_channels=planes,
                                       kernel_size=kernel,
                                       stride=2,
                                       padding=padding,
                                       output_padding=output_padding,
                                       bias=self.deconv_with_bias))
      layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
      layers.append(nn.ReLU(inplace=True))'''

      layers.append(UpModule(self.inplanes, planes, kernel_size=2, stride=2, mode='UCBA'))
      self.inplanes = planes

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    #print(x.shape)
    x = self.deconv_layers(x)
    #print("res:{}".format(x.shape))
    out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
    return out

  def init_weights(self, num_layers):
    for m in self.deconv_layers.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.001)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
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
    #url = model_urls['resnet{}'.format(num_layers)]
    #pretrained_state_dict = model_zoo.load_url(url)
    #print('=> loading pretrained model {}'.format(url))
    #self.load_state_dict(pretrained_state_dict, strict=False)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def resnet_18():
  model = PoseResNet(BasicBlock, [2, 2, 2, 2], head_conv=64, num_classes=80)
  model.init_weights(18)
  return model

def get_pose_net(num_layers, head_conv=64, num_classes=80):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, head_conv=head_conv, num_classes=num_classes)
  model.init_weights(num_layers)
  return model


if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    # pass


  net = get_pose_net(num_layers=18, head_conv=0, num_classes=4)
  x = torch.randn(2, 3, 512, 512)

  #for name, module in net.named_children():
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
  
  #y = net(torch.randn(2, 3, 384, 384))
  y = net(torch.randn(2, 3, 512, 512))


  # print(y.size())
