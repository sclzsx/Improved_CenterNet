import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import torch.utils.data
from dataloader.pascal import PascalVOC
from utils.utils import _tranpose_and_gather_feature, init_net
from utils.losses import _neg_loss, _reg_loss
from configs.CC import Config

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

parser = argparse.ArgumentParser(description='centernet')
parser.add_argument('--save_root', type=str, default='./results')
parser.add_argument('--data_dir', type=str, default='/workspace/DATA/zhatu/zhatu0814/voc')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
parser.add_argument('--num_layers', type=str, default=18)
parser.add_argument('--split_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=140)
parser.add_argument('--test_topk', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--pretrain_file', default=None, type=str)
parser.add_argument('--config_filename', default='vgg_optim_320', type=str)

cfg = parser.parse_args()
cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]
CFG = Config.fromfile('./configs/' + cfg.config_filename + '.py')

if not cfg.pretrain_file:
    strtime = time.strftime("%Y-%m-%d-%H", time.localtime())
    save_folder = cfg.config_name + "_" + strtime
    os.makedirs(save_folder, exist_ok=True)
    resume_epoch = 1
else:
    save_folder = os.path.dirname(cfg.pretrain_file)
    resume_epoch = CFG.model.resume_epoch


def train_a_epoch(model, train_loader, epoch, optimizer):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        for k in batch:
            if k != 'img_id':
                batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

        outputs = model(batch['image'])
        hmap, regs, w_h_ = zip(*outputs)
        regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
        w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]
        hmap_loss = _neg_loss(hmap, batch['hmap'])
        reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
        w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
        loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))
              + 'hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' % (hmap_loss.item(), reg_loss.item(), w_h_loss.item()))
    return


def main():
    torch.backends.cudnn.benchmark = True
    cfg.device = torch.device('cuda')
    train_dataset = PascalVOC(cfg.data_dir, split_ratio=cfg.split_ratio, img_size=cfg.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16)
    model = CFG.model.model(num_layers=cfg.num_layers, num_classes=train_dataset.num_classes).to(cfg.device)
    init_net(model, cfg.pretrain_file)
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)
    for epoch in range(resume_epoch, cfg.num_epochs + 1):
        train_a_epoch(model, train_loader, epoch, optimizer)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_folder + '/checkpoint_%d' % epoch + '.t7')
        lr_scheduler.step(epoch)


if __name__ == '__main__':
    main()
