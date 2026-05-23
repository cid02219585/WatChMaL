'''
Author: Shuoyu Chen shuoyuchen.physics@gmail.com
Date: 2025-06-23 13:09:23
LastEditors: Shuoyu Chen shuoyuchen.physics@gmail.com
LastEditTime: 2025-07-11 09:26:49
FilePath: /schen/workspace/WatChMaL/watchmal/model/SwinTransformer.py
Description: 
'''
'''
Here is a Swin Transformer model.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SwinRegressor(nn.Module):
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=(192, 192),
        in_chans=2,
        num_output_channels=3,
    ):
        super().__init__()

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_output_channels,
            img_size=img_size,
        )
        self.output_dim = num_output_channels

    def forward(self, x):
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False) # upsampling the 18 x16 double covered WCTE image -- needs to be multiple of 28
        out = self.vit(x)
        return out
