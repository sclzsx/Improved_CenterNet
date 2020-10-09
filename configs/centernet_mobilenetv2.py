from nets.MobileNetv2 import get_pose_net

model = dict(
    model=get_pose_net,
    input_size=512,
    rgb_means=(104, 117, 123),
    p=0.6
)
