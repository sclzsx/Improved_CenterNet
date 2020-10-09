# from nets.resnet_optim import get_pose_net
# from nets.vgg_optim import get_pose_net
# from nets.hrnet_vggv2 import get_pose_net
from nets.vgg_optim import get_pose_net

model = dict(
    name='hrnet_vggv2',
    model=get_pose_net,
    input_size=320,
    resume_model=False,
    resume_folder="./weights/0.25vgg16_2020_03_17_14_18_21_epoch91.pth",
    resume_epoch=0,
    rgb_means=(104, 117, 123),
    p=0.6
)

dataset = dict(
    dataset='VOC',  ## VOC,  COCO
    # VOC_CLASSES = ( '__background__', 'car', 'person','zebra_crossing'),
    # VOC_CLASSES = ( '__background__', 'car', 'person'),
    VOC_CLASSES=('__background__', 'person'),
    VOC=dict(
        VOCroot="/mnt/workspace/ypgao/detectData/",
        train_sets=[('TAXIFACEDETECT', 'trainval'),
                    ('GDCY', 'trainval'),
                    ('BigFace', 'trainval')],
        eval_sets=[('TAXIFACEDETECT', 'test')],
    )
)

train_cfg = dict(
    cuda=True,
    weight_decay=0.0005,
    gamma=0.1,
    momentum=0.9,
    warmup=5,
    batch_size=64,
    lr=[0.004, 0.002, 0.0004, 0.00004, 0.000004],
    end_lr=4e-3,
    step_lr=dict(
        COCO=[90, 110, 130, 150, 160],
        VOC=[100, 150, 200, 250, 300],  # unsolve
    ),
    print_epochs=10,
    num_workers=16,
    max_epoch=300,
)

anchor = dict(
    feature_maps=[38, 19, 10, 5, 3, 1],
    steps=[8, 16, 32, 64, 100, 300],
    min_sizes=[30, 60, 111, 162, 213, 264],
    max_sizes=[60, 111, 162, 213, 264, 315],
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    variance=[0.1, 0.2],
    clip=True,
)
