import torch
import torch.nn as nn
from nets.modules import *

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = int(in_channels / 2)
        self.double_conv = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=3, padding=1),
            BasicConv(inter_channels, out_channels, kernel_size=3, padding=1, stride=2)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, in_channels_2, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = BasicConv(in_channels_2, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            BasicConv(out_channels, out_channels // 2, kernel_size=3, padding=1),
            BasicConv(out_channels // 2, out_channels, kernel_size=1, padding=0)
        )


    def forward(self, x1, x2):
        x1 = self.up(x1)

        x2 = self.conv1(x2)

        #x = torch.cat([x2, x1],dim=1)
        x = x1 + x2

        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels):
        super(Unet, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )    #128

        self.down1 = Down(32, 64)   # 64
        self.down2 = Down(64, 128)  # 32
        self.down3 = Down(128, 128)  # 16

        self.up1 = Up(128, 128, 128) # 32
        self.up2 = Up(128, 64, 128) # 64
        self.up3 = Up(128, 32, 128)  # 128


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return x



class CenterNet(nn.Module):
  def __init__(self, head_conv, num_classes):
    super(CenterNet, self).__init__()
    self.in_channels = 3
    self.Unet = Unet(self.in_channels)

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


  def forward(self, x):
    x = self.Unet(x)

    out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
    return out

  def init_weights(self):
    for m in self.Unet.modules():
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




def get_pose_net(num_layers, head_conv=64, num_classes=80):

  model = CenterNet(head_conv=head_conv, num_classes=num_classes)
  model.init_weights()
  return model


if __name__ == '__main__':
    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        # pass


    net = get_pose_net(num_layers=18, head_conv=0, num_classes=4)
    x = torch.randn(2, 3, 512, 512)


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
