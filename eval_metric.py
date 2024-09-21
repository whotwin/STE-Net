import torch
import torch.nn.functional as F
def Dice_WT(input, target):
    b, c, h, w = input.size()
    bt, ct, ht, wt = target.size()
    if h != ht or w != wt:
        input = F.interpolate(input, (ht, wt), None, 'bilinear', align_corners=True)
    temp_in = input.view(b, c, -1)
    temp_ta = target.view(b, c, -1)
    pred = torch.softmax(temp_in, dim=1)
    argmax = torch.argmax(pred, dim=1)
    label = 1 - temp_ta[:, 1, :]
    pred = (argmax != 1).float()
    union = torch.sum(pred**2) + torch.sum(label**2) + 1
    dice = torch.sum(pred * label)

    return (((2*dice) + 1)/union)

def Dice_TC(img, label):
    b, c, w, h = img.size()
    img = img.view(b, c, -1)
    label = label.view(b, c, -1)
    pred = torch.softmax(img, dim=1)
    output = torch.argmax(pred, dim=1)
    label = label[:, 3, ...] + label[:, 2, :]
    output = (output==3).float() + (output == 2).float()
    intersect = torch.sum(output * label)
    dice = ((2 * intersect) + 1) / (torch.sum(output**2) + torch.sum(label**2) + 1)
    return dice

def Dice_ET(img, label):
    b, c, w, h = img.size()
    img = img.view(b, c, -1)
    label = label.view(b, c, -1)
    pred = torch.softmax(img, dim=1)
    output = torch.argmax(pred, dim=1)
    label = label[:, 3, ...]
    output = (output==3).int().float()
    intersect = torch.sum(output * label)
    dice = ((2 * intersect) + 1) / (torch.sum(output**2) + torch.sum(label**2) + 1)
    return dice