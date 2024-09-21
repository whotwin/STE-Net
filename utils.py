import torch
import torch.nn.functional as F
def sum_tensor(inp, axes, keepdim=False):
    for ax in sorted(axes, reverse=True):
        inp = inp.sum(int(ax))
    return inp

def Dice_2d(img, target, smooth=1):
    b, c, h, w = img.size()
    tb, tc, th, tw = target.size()
    if h != th or w != tw:
        img = F.interpolate(img, (th, tw), None, mode="bilinear", align_corners=True)
    temp_img = img.view(b, c, -1)#.permute(0, 2, 1)
    temp_target = target.view(b, c, -1)
    #pred = temp_img
    pred = torch.softmax(temp_img, dim=1)
    #pred = torch.sigmoid(temp_img)
    #pred = (predd > 0.5).int().float().requires_grad_()
    #pred = torch.nn.functional.sigmoid(temp_img).permute(0, 2, 1)

    tp = pred * temp_target
    axe = [0] + list(range(2, len(pred.shape)))
    tp = sum_tensor(tp, axe)
    volumes = sum_tensor(temp_target**2, axe)
    m = sum_tensor(pred**2, axe)
    dc = ((2. * tp) + smooth) / (m + volumes + smooth)
    #return dc.mean()
    return dc
    
def Dice_Loss(img, target):
    target = target.float()
    dice_coeff = Dice_2d(img, target)# * torch.tensor([1.2, 0.9, 0.9]).cuda()
    #dice_coeff = torch.cat([dice_coeff_f[:1], dice_coeff_b])
    return (1 - dice_coeff).mean()
