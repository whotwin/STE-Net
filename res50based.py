import torch
import numpy as np
import torch.nn as nn
from basic_module import *
from attention import CFM



def convolution(in_planes, out_planes, kernel_size=3, stride=1):
    padding = kernel_size // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=1, groups=4)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = convolution(inplanes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convolution(planes, planes, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = convolution(planes, planes, kernel_size=7)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class STE_Single(nn.Module):
    def __init__(self, block, layers, image_size, last_channel=4):
        super(STE_Single, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(12, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.swin_layers = nn.ModuleList()
        embed_dim = 64
        self.num_layers = [0, 1, 2, 3]#4
        self.image_size = image_size
        depths = [2, 2, 2, 2]
        num_heads = [2, 4, 8, 16]
        window_size = self.image_size // 16
        self.mlp_ratio = 4.0
        drop_path_rate = 0.4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [self.image_size//4, self.image_size//4]

        patch_size = [2, 4, 8, 16]
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=6, embed_dim=embed_dim)
        self.patch_embed_2 = PatchEmbed(img_size=image_size // 2, patch_size=patch_size, in_chans=embed_dim, embed_dim=2*embed_dim)
        self.patch_embed_3 = PatchEmbed(img_size=image_size // 4, patch_size=patch_size[:3], in_chans=2*embed_dim, embed_dim=4*embed_dim)
        self.patch_embed_4 = PatchEmbed(img_size=image_size // 8, patch_size=patch_size[:2], in_chans=4*embed_dim, embed_dim=8*embed_dim)
        self.MS1 = PatchMerging(embed_dim)

        for i_layer in self.num_layers:#range(self.num_layers):
            swin_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                               qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swin_layers.append(swin_layer)
            
        channels = [embed_dim,embed_dim*2,embed_dim*4,embed_dim*8]
        self.decode4 = Decoder(channels[3],channels[2])
        self.decode3 = Decoder(channels[2],channels[1])
        self.decode2 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                     nn.Conv2d(channels[0], channels[0], 3, 1, 1, groups=8),
                                     nn.BatchNorm2d(channels[0]),
                                     nn.ReLU(),
                                     nn.Conv2d(channels[0], channels[0], 3, 1, 1)
                                     )
        self.final = nn.Conv2d(channels[0], last_channel, kernel_size=1,bias=False)
        self.conv = nn.Conv2d(2*embed_dim, embed_dim, 1)
        self.layer_1 = self._make_layer(block, 64, layers[0])
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)
    def _make_layer(self, block, planes, blocks, stride=1): # 创建layer层，（block_num-1）表示此层中Residual Block的个数 （7）
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        encoder = []
        ms1 = self.patch_embed(x)
        ms1 = self.MS1(ms1)
        ms1 = self.conv(ms1)#.flatten(2).transpose(1, 2)
        xx = self.layer_1(ms1).flatten(2).transpose(1, 2)

        x = self.swin_layers[0](ms1.flatten(2).transpose(1, 2)) + xx
        B, L, C = x.shape

        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1,2)#b, 128, 64, 64
        
        
        encoder.append(x)
        xx = self.layer_2(x).flatten(2).transpose(1, 2)
        x = self.patch_embed_2(x)

        x = self.swin_layers[1](x) + xx
        
        B, L, C = x.shape
        
        #ms3 = self.MS3(x)
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1,2)

        encoder.append(x)
        xx = self.layer_3(x).flatten(2).transpose(1, 2)
        x = self.patch_embed_3(x)

        x = self.swin_layers[2](x) + xx
        B, L, C = x.shape

        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1,2)


        encoder.append(x)
        xx = self.layer_4(x).flatten(2).transpose(1, 2)
        x = self.patch_embed_4(x)

        x = self.swin_layers[3](x) + xx
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)


        encoder.append(x)
        d4 = self.decode4(encoder[3], encoder[2])


        d3 = self.decode3(d4, encoder[1])


        d2 = self.decode2(d3, encoder[0])

        out = self.decode0(d2)   
        out = self.final(out) 

        return out, encoder
    
class STENet(STE_Single):
    def __init__(self, image_size, opt, last_channel=4, block=BasicBlock):
        super(STE_Single, self).__init__()
        embed_dim = 64
        layers = [3, 4, 6, 3]
        self.branch_1 = STE_Single(BasicBlock, layers, image_size)#ResNet(BasicBlock, [3, 4, 6, 3], 4, opt)
        self.branch_2 = STE_Single(BasicBlock, layers, image_size)

        self.num_layers = [0, 1, 2, 3]#4
        patch_size = [2, 4, 8, 16] 
        depths = [2, 2, 2, 2]
        num_heads = [2, 4, 8, 16]
        window_size = image_size // 16
        self.mlp_ratio = 4.0
        drop_path_rate = 0.4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [image_size//4, image_size//4]
        self.patch_embed_2 = PatchEmbed(img_size=image_size // 2, patch_size=patch_size, in_chans=embed_dim, embed_dim=2*embed_dim, stride=2)
        self.patch_embed_3 = PatchEmbed(img_size=image_size // 4, patch_size=patch_size[:3], in_chans=2*embed_dim, embed_dim=4*embed_dim, stride=2)
        self.patch_embed_4 = PatchEmbed(img_size=image_size // 8, patch_size=patch_size[:2], in_chans=4*embed_dim, embed_dim=8*embed_dim, stride=2)
        self.swin_layers = nn.ModuleList()
        for i_layer in self.num_layers:#range(self.num_layers):
            swin_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                               qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swin_layers.append(swin_layer)
        self.cfm_1 = CFM(True, embed_dim)
        self.cfm_2 = CFM(False, 2*embed_dim)
        self.cfm_3 = CFM(False, 4*embed_dim)
        self.cfm_4 = CFM(False, 8*embed_dim)

        channels = [embed_dim,embed_dim*2,embed_dim*4,embed_dim*8]
        self.decode4 = Decoder(channels[3],channels[2])

        self.decode3 = Decoder(channels[2],channels[1])

        self.decode2 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                     nn.Conv2d(channels[0], channels[0], 3, 1, 1, groups=8),
                                     nn.BatchNorm2d(channels[0]),
                                     nn.ReLU(),
                                     nn.Conv2d(channels[0], channels[0], 3, 1, 1))
        self.final = nn.Conv2d(channels[0], last_channel, kernel_size=1,bias=False)

        self.mass_conv_1 = nn.Conv2d(last_channel, last_channel, 3, 1, 1)
        self.mass_conv_2 = nn.Conv2d(last_channel, last_channel, 3, 1, 1)
        self.mass_conv_3 = nn.Conv2d(last_channel, last_channel, 3, 1, 1)
        self.inplanes = 64
        self.layers_1 = self._make_layer(block, 64, layers[0])
        self.layers_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layers_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layers_4 = self._make_layer(block, 512, layers[3], stride=2)


    def forward(self, x1, x2, monte_carlo=False):
        x1 = x1.float()
        x2 = x2.float()
        out_1, encoder_1 = self.branch_1(x1)
        pre_1 = out_1

        out_2, encoder_2 = self.branch_2(x2)
        pre_2 = out_2
        encoder = []

        cfm_1 = self.cfm_1(encoder_1[0], encoder_2[0], 0)
        xx = self.layers_2(cfm_1).flatten(2).transpose(1, 2)
        f1 = self.patch_embed_2(cfm_1)
        #cfm_1 = cfm_1.flatten(2).transpose(1, 2)
        f1 = self.swin_layers[1](f1) + xx
        B, L, C = f1.shape
        #f1 = self.MS3(f1)
        f1 = f1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1,2)

        encoder.append(f1)
        cfm_2 = self.cfm_2(encoder_1[1], encoder_2[1], f1)
        xx = self.layers_3(cfm_2).flatten(2).transpose(1, 2)
        f2 = self.patch_embed_3(cfm_2)

        f2 = self.swin_layers[2](f2) + xx
        B, L, C = f2.shape

        f2 = f2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1,2)
        
        encoder.append(f2)
        cfm_3 = self.cfm_3(encoder_1[2], encoder_2[2], f2)
        xx = self.layers_4(cfm_3).flatten(2).transpose(1, 2)
        f3 = self.patch_embed_4(cfm_3)

        f3 = self.swin_layers[3](f3) + xx
        B, L, C = f3.shape

        f3 = f3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1,2)
        
        encoder.append(f3)
        cfm_4 = self.cfm_4(encoder_1[3], encoder_2[3], f3)

        de_0 = self.decode4(cfm_4, cfm_3)#self.attn_1(mcm_1))
        decoder = []
        decoder.append(de_0)

        de_1 = self.decode3(de_0, f1)

        de_2 = self.decode2(de_1, cfm_1)

        out = self.decode0(de_2)
        out = self.final(out)

        return pre_1, pre_2, out