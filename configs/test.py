from configs.CC import Config


global cfg
cfg = Config.fromfile("m2det320_vgg.py")


print(cfg.model.input_size)
print(cfg.model.m2det_config)



#cfg = (VOC_300, VOC_512)[args.size == '512']

if args.version == 'SSD_VGG_FPN_RFB':
    from models.SSD_VGG_Optim_FPN_RFB import build_net
elif args.version == 'ResNet18_FPN':
    from models.SSD_ResNet_FPN import build_net
elif args.version == 'EfficientDet':
    from models.EfficientDet import build_net
elif args.version == 'DetNet':
    from models.SSD_DetNet import build_net
    cfg = DetNet_300
elif args.version == 'VGG_ShareHead':
    from models.SSD_VGG_ShareHead import build_net
elif args.version == 'M2Det':
    from models.SSD_M2Det import build_net
    cfg = M2Det_320
else:
    print('Unkown version!')