import torch
import itertools
import torch.nn as nn

class CFM(nn.Module):
    def __init__(self, is_first, inch):
        super(CFM, self).__init__()
        super_param = 4

        self.super = super_param
        self.slice_occ = inch // super_param
        #self.conv = nn.Conv2d(2*inch, 2*inch, 3, 1, 1)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(2*inch, 2*super_param, 3, 1, 1),
            nn.BatchNorm2d(2*super_param)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(inch, inch, 3, 1, 1),
            nn.BatchNorm2d(inch),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*inch, inch, 3, 1, 1),
            nn.BatchNorm2d(inch),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*inch, inch, 3, 1, 1),
            nn.BatchNorm2d(inch),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.first = is_first
    def forward(self, x1, x2, x):
        b, c, h, w = x1.size()
        c1 = torch.cat([x1, x2], dim=1)
        c1 = self.conv3x3(c1)
        c1 = self.pool(c1)
        
        weight = self.sigmoid(c1)
        
        weight_1 = torch.ones((b, c, 1, 1)).cuda()
        for i in range(self.super):
            weight_1[:, i*self.slice_occ:(i+1)*self.slice_occ] = weight[:, i].unsqueeze(1)
        weight_2 = torch.zeros_like(x2)
        for i in range(self.super):
            weight_2[:, i*self.slice_occ:(i+1)*self.slice_occ] = weight[:, i+self.super].unsqueeze(1)
        x1 = x1 * weight_1
        x2 = x2 * weight_2
        f = x1 + x2
        f = self.conv2(f)
        f = torch.cat([x1, x2, f], dim=1)
        f = self.conv3(f)
        if not self.first:
            f = self.conv4(torch.cat([f, x], dim=1))
        return f
    
def compute_c(m1, m2, num_classes):#num_classes共有四个类、含背景
    b, c, h, w = m1.size()
    comb = []
    for i in range(num_classes):
        ls = itertools.combinations(range(num_classes), i+1)#输入的通道数维所有的观测对象类别{A}, {B}, {A, B}
        if i == 0:
            #comb.append(list(map(lambda x:x, ls)))
            comb = list(ls)
        else:
            comb = comb + list(ls)
    #comb = list(comb[:num_classes]) + list(comb[-1])
    mass = torch.zeros((b, h, w)).cuda()
    for m in range(num_classes+1):#刨去c个全部观测的
        if m == num_classes:
            m = -1
        for n in range(num_classes+1):
            if n >= num_classes:
                n = -1
            if len(set(comb[m]) & set(comb[n])) == 0:#两个类别的交集为0
                continue
            else:
                k = len(set(comb[m]) & set(comb[n])) / len(set(comb[m]) | set(comb[n]))
                mass = mass + m1[:, m] * m2[:, n] * k
            if n == -1:
                break
        if m == -1:
            break
    return mass.unsqueeze(1)
    #return comb

def comput_k(c11, c22, c12):
    return 1 - c12 / torch.sqrt(c11 * c22)

def compute_NS(c11, c12, c13, c22, c23, c33):
    k12 = comput_k(c11, c22, c12)
    k13 = comput_k(c11, c33, c13)
    k21 = k12
    k23 = comput_k(c22, c33, c23)
    k32 = k23
    k31 = k13
    return 2 - k12 - k13, 2 - k21 - k23, 2 - k31 - k32, k12, k13, k23

def compute_wdn(ns1, ns2, ns3):
    w_sum = ns1 + ns2 + ns3
    wdn1 = ns1 / w_sum
    wdn2 = ns2 / w_sum
    wdn3 = ns3 / w_sum
    return wdn1, wdn2, wdn3

def compute_widn(k12, k13, k23):
    miu1 = (torch.abs(1 - k12) + torch.abs(1 - k13)) / 2.
    miu2 = (torch.abs(1 - k12) + torch.abs(1 - k23)) / 2.
    miu3 = (torch.abs(1 - k13) + torch.abs(1 - k23)) / 2.
    w_sum = miu1 + miu2 + miu3
    widn1 = miu1 / w_sum
    widn2 = miu2 / w_sum
    widn3 = miu3 / w_sum
    return widn1, widn2, widn3

def compute_tot_avg(wdn1, wdn2, wdn3, widn1, widn2, widn3, m1, m2, m3):
    mult_1 = wdn1*widn1
    mult_2 = wdn2*widn2
    mult_3 = wdn3*widn3
    w_sum = mult_1 + mult_2 + mult_3
    tot_1 = mult_1 / w_sum
    tot_2 = mult_2 / w_sum
    tot_3 = mult_3 / w_sum
    avg = tot_1*m1 + tot_2*m2 + tot_3*m3
    m1 = (m1 + avg) / 2.
    m2 = (m2 + avg) / 2.
    m3 = (m3 + avg) / 2.
    return m1, m2, m3

def dempster(m1, m2, num_classes):
    b, c, h, w = m1.size()
    m = torch.zeros_like(m1)
    comb = []
    for i in range(num_classes):
        ls = itertools.combinations(range(num_classes), i+1)#输入的通道数维所有的观测对象类别{A}, {B}, {A, B}
        if i == 0:
            #comb.append(list(map(lambda x:x, ls)))
            comb = list(ls)
        else:
            comb = comb + list(ls)
    #comb = list(comb[:num_classes]) + list(comb[-1])
    k = torch.zeros((b, h, w)).cuda()
    for i, set_a in enumerate(comb):
        if i >= num_classes:
            set_a = comb[-1]
            i = -1
        for j, set_b in enumerate(comb):
            if j >= num_classes:
                set_b = comb[-1]
                j = -1
            if len(set(set_a) & set(set_b)) == 0:
                k = k + m1[:, i]*m2[:, j]
            if j == -1:
                break
        if i == -1:
            break

    #k = k - 1e-8
    k = torch.where(k < 0.9999, k, 0.9)
    for l, set_c in enumerate(comb):
        if l >= num_classes:
            set_c = comb[-1]
            l = -1
        for i, set_a in enumerate(comb):
            if i >= num_classes:
                set_a = comb[-1]
                i = -1
            for j, set_b in enumerate(comb):
                if j >= num_classes:
                    set_b = comb[-1]
                    j = -1
                if set(set_a) & set(set_b) == set(set_c):
                    m[:, l] = m[:, l] + m1[:, i] * m2[:, j]
                if j == -1:
                    break
            if i == -1:
                break
        if l == -1:
            break
        #m[:, l] = m[:, l] / (1 - k)
    m = m / (1 - k.unsqueeze(1))
    return m