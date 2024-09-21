import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from res50based import STENet
from datasets import BTS_data
from torch.optim import lr_scheduler
from train_stage import train_stage
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="STE-Net")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--preprocess", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--lr", type=int, default=1e-3)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--ch", type=list, default=[1, 32])
parser.add_argument("--depth", type=int, default=7)
parser.add_argument("--group", type=int, default=8)
parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--inch', type=int, default=6)
parser.add_argument("--save_folder", type=str)
parser.add_argument('--crop_size', type=list, default=(224, 224))
parser.add_argument('--load_weight', type=bool, default=False)
parser.add_argument('--data_folder', type=str)
parser.add_argument('--resume_path', type=str)
parser.add_argument('--pretrained', type=bool, default=False)
opt = parser.parse_args()

def main():
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)
    train_set = BTS_data(opt, "train")
    val_set = BTS_data(opt, "val")
    model = STENet(opt.crop_size[0], opt)

    device_ids = [0]
    model = nn.DataParallel(model, device_ids)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99), eps=1e-8)
    current_lr = optimizer.param_groups[0]["lr"]
    if opt.pretrained:
        #if os.path.isfile(sets.resume_path):
        print("=> loading checkpoint '{}'".format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        model_dict = model.state_dict()
        checkpoint = {key: value for key, value in checkpoint.items() if (('fc' not in key and ('conv1' not in key)))}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict, strict=False)
    train_loader = DataLoader(train_set, opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, 1, shuffle=False)
    model = model.cuda()

    scheduler = lr_scheduler.PolynomialLR(optimizer, opt.epoch, power=0.9)
    model = train_stage(train_loader, val_loader, model, opt, optimizer, current_lr, scheduler)#

if __name__ == '__main__':
    main()