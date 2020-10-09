from nets.Unet import get_pose_net

model = dict(
    name='centernet_unet',
    model=get_pose_net,
    input_size=512,
    resume_model=False,
    resume_folder=None,
    resume_epoch=0,
    rgb_means=(104, 117, 123),
    p=0.6
)
