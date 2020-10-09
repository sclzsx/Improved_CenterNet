import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

#from nets.resnet_optim import get_pose_net
#from nets.MobileNetv2 import CenterNet
#from nets.MobileNetv2G import MobileNetV2
#from nets.vgg_optim import get_pose_net
#from nets.resnet_optimG import get_pose_net
#from nets.centerFace_MobileNetV2 import get_mobile_net
#from nets.CenterFace_MV2 import get_pose_net
from nets.MobileNetSSH import DBFace
#from nets.ResNet_FPN import get_pose_net
#from nets.MobileNetv2 import get_pose_net
# from nets.vgg_optim import get_pose_net
from nets.hrnet_vggv2 import get_pose_net




if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    # pass


  net = get_pose_net(num_layers=18, head_conv=64, num_classes=2)
  #net = MobileNetV2(head_conv=64, num_classes=4,width_mult=0.5)
  #net = DBFace(has_landmark=False, wide=64, has_ext=False, upmode="UCBA")

  print(net)
  y = net(torch.randn(2, 3, 320, 320))

  #sys.exit()
  x = torch.randn(10, 3, 320, 320)
  #for name, module in net.named_children():
  #  x = module(x)
  #  print(name, x.shape)

  from ptflops import get_model_complexity_info

  img_dim = 512
  flops, params = get_model_complexity_info(net, (320, 320), as_strings=True, print_per_layer_stat=True)
  print('Flops: ' + flops)
  print('Params: ' + params)
    

  
  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      m.register_forward_hook(hook)
  
  #y = net(torch.randn(2, 3, 384, 384))
 