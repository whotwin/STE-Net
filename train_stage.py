from tqdm import tqdm
import os.path as pt
import numpy as np
import torch
from mass import mass
import torch.nn.functional as F
from utils import Dice_Loss
from attention import compute_c, compute_NS, compute_tot_avg, dempster, compute_wdn, compute_widn
from eval_metric import Dice_WT, Dice_TC, Dice_ET

def DS_Fusion(in_1, in_2, in_3, pred_1, pred_2, pred_3, label, opt, window):
    n_classes = opt.num_classes
    m = mass(window, opt.num_classes)

    m1 = m.forward(in_1, pred_1)

    m2 = m.forward(in_2, pred_2)

    m3 = m.forward(in_3, pred_3)

    c11 = compute_c(m1, m1, n_classes)
    c12 = compute_c(m1, m2, n_classes)
    c13 = compute_c(m1, m3, n_classes)
    c23 = compute_c(m2, m3, n_classes)
    c22 = compute_c(m2, m2, n_classes)
    c33 = compute_c(m3, m3, n_classes)
    n1, n2, n3, k12, k13, k23 = compute_NS(c11, c12, c13, c22, c23, c33)
    wdn1, wdn2, wdn3 = compute_wdn(n1, n2, n3)
    widn1, widn2, widn3 = compute_widn(k12, k13, k23)
    m1, m2, m3 = compute_tot_avg(wdn1, wdn2, wdn3, widn1, widn2, widn3, m1, m2, m3)
    a = torch.max(wdn1.view(1, -1), dim=1)[0]
    stage_1 = dempster(m1, m2, n_classes)
    stage_2 = dempster(stage_1, m3, n_classes)
    out = stage_2
    out = stage_2[:, :-1, ...] + (stage_2[:, -1, ...].unsqueeze(1)) / n_classes
    return out


def train_stage(train_loader, val_loader, model, opt, optimizer, current_lr, scheduler):
    best_dice = 0
    for epoch in range(opt.epoch):
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        total_loss = []
        model.train()
        
        loop = tqdm(enumerate(train_loader), total = len(train_loader))
        for i, (data, label) in loop:

            data, label = data.cuda(), label.cuda()
            
            label = label.float()

            data_1 = data[:, :6, ...]#   T2, Flair; T1c T1
            data_2 = data[:, 6:, ...]

            output_1, output_2, output = model(data_1, data_2)
            pred_1, pred_2, pred = output_1, output_2, output
            

            pred_1 = F.softmax(pred_1, dim=1)
            pred_2 = F.softmax(pred_2, dim=1)
            pred = F.softmax(pred, dim=1)

            loss = Dice_Loss(output, label) + 0.5 * Dice_Loss(output_1, label) + 0.2 * Dice_Loss(output_2, label)
            total_loss.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss = np.mean(total_loss)
            loop.set_description(f'train_stage, Epoch[{epoch} / {opt.epoch}]')
            loop.set_postfix(loss = loss)

        total_loss = []
        total_wt = []
        total_tc = []
        total_et = []
        loop = tqdm(enumerate(val_loader), total = len(val_loader))
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        with torch.no_grad():
            for i, (data, label) in loop:
                model.eval()
                data, label = data.cuda(), label.cuda()
                label = label.float()

                data_1 = data[:, :6, ...]#   T2, Flair; T1c T1
                data_2 = data[:, 6:, ...]

                output_1, output_2, output = model(data_1, data_2)

                pp = torch.softmax(output, dim=1)
                pp = torch.argmax(pp, dim=1).unsqueeze(1)
                pp_ed = (pp == 0).int();    pp_ncr = (pp == 2).int(); pp_et = (pp == 3).int()
                pp_bg = ((pp!=0)*(pp!=2)*(pp!=3)).int(); pp_3 = torch.cat([pp_ed, pp_bg, pp_ncr, pp_et], dim=1).float()

                pp = torch.softmax(output_2, dim=1)
                pp = torch.argmax(pp, dim=1).unsqueeze(1)
                pp_ed = (pp == 0).int();    pp_ncr = (pp == 2).int(); pp_et = (pp == 3).int()
                pp_bg = ((pp!=0)*(pp!=2)*(pp!=3)).int(); pp_2 = torch.cat([pp_ed, pp_bg, pp_ncr, pp_et], dim=1).float()
                
                pp = torch.softmax(output_1, dim=1)
                pp = torch.argmax(pp, dim=1).unsqueeze(1)
                pp_ed = (pp == 0).int();    pp_ncr = (pp == 2).int(); pp_et = (pp == 3).int()
                pp_bg = ((pp!=0)*(pp!=2)*(pp!=3)).int(); pp_1 = torch.cat([pp_ed, pp_bg, pp_ncr, pp_et], dim=1).float()

                refined_out = DS_Fusion(output_1, output_2, output, pp_1, pp_2, pp_3, pp, opt, 3)
                pred_1, pred_2, pred = output_1, output_2, output
                pred_1 = F.softmax(pred_1, dim=1)
                pred_2 = F.softmax(pred_2, dim=1)
                pred = F.softmax(pred, dim=1)
                loss = Dice_Loss(output, label) + 0.5 * Dice_Loss(output_1, label) + 0.2 * Dice_Loss(output_2, label)

                wt = Dice_WT(refined_out, label)
                tc = Dice_TC(refined_out, label)
                et = Dice_ET(refined_out, label)

                total_wt.append(wt.cpu().detach().numpy())
                total_tc.append(tc.cpu().detach().numpy())
                total_et.append(et.cpu().detach().numpy())
                
                dice_seg = np.mean(np.array([np.mean(total_wt), np.mean(total_tc), np.mean(total_et)]))

                total_loss.append(loss.item())
                loss = np.mean(total_loss)
                loop.set_description(f'validation_stage, Epoch[{epoch} / {opt.epoch}]')
                loop.set_postfix(loss = loss, dice_seg=dice_seg)
            print("[epoch %d][%d/%d] loss: %.4f dice_val: %.4f" % (epoch+1, i+1, len(val_loader), loss, dice_seg))
            if best_dice < dice_seg:
                best_dice = max(dice_seg, best_dice)
                torch.save(model.state_dict(), pt.join(opt.save_folder, "best_weight.pth"))
            
    return model