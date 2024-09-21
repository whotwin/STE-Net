import torch
import torch.nn as nn
import torch.nn.functional as F

class mass(nn.Module):
    def __init__(self, mask_range, num_classes):
        super(mass, self).__init__()
        self.mask_range = mask_range
        self.n_cls = num_classes

    def forward(self, x, label):
        mask = self.generate_mask()
        agg, s = self.aggregation(mask, label)
        out = self.reweight(x, s, agg)
        return out
    
    def generate_mask(self):
        mask = torch.ones((self.n_cls, 1, self.mask_range, self.mask_range)).cuda()
        center_x, center_y = (self.mask_range - 1) // 2, (self.mask_range - 1) // 2
        mask_cross = torch.zeros_like(mask)
        mask_cross[:, :, center_x] = 1
        mask_cross[:, :, :, center_y] = 1
        for i in range(self.mask_range):
            for j in range(self.mask_range):
                mask[:, :, i, j] = (self.mask_range - 1)**2 + 1 - (abs(i - (self.mask_range - 1) // 2) + abs(j - (self.mask_range - 1) // 2))**2
        mask = mask * mask_cross
        return mask
    
    def aggregation(self, mask, x):
        out = F.conv2d(x, mask, padding=(self.mask_range - 1)//2, groups=self.n_cls)
        S = torch.sum(mask, dim=[-2, -1])
        return out, S.unsqueeze(0).unsqueeze(-1)
    
    def penalty(self, x):
        b, c, h, w = x.size()
        max = (self.n_cls - 1) * abs((1 / self.n_cls)) + abs(1 - 1 / self.n_cls)
        real = torch.zeros((b, h, w)).cuda()
        for i in range(self.n_cls):
            real = real + torch.abs(x[:, i] - (1 / self.n_cls))
        return (real / max)

    def reweight(self, x, s, agg):
        x = torch.softmax(x, dim=1)
        #r = agg / s#b, c, h, w
        _, index = torch.max(x, dim=1)
        refined = x.clone()
        #p = self.penalty(refined)
        #refined = refined * p
        for i in range(self.n_cls):
            refined[:, i] = torch.where(index == i, agg[:, i]*x[:, i]/s[:, i], x[:, i])#(agg[:, i]+s[:, i])*x[:, i]/s[:, i])
            #refined[:, i] = refined[:, i] * p
        
        temp_sum = torch.sum(refined, dim=1)
        all = (1 - temp_sum).unsqueeze(1)
        fine_mass = torch.cat([refined, all], dim=1)
        return fine_mass