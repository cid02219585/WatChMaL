import torch.nn as nn
import torch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_pad_mode='zeros', norm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride, conv_pad_mode)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding_mode=conv_pad_mode)
        self.bn2 = norm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_pad_mode='zeros', norm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm(planes)
        self.conv2 = conv3x3(planes, planes, stride, padding_mode=conv_pad_mode)
        self.bn2 = norm(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_encoder(nn.Module):

    def __init__(self, block, layers, num_input_channels, zero_init_residual=False,
                 conv_pad_mode='zeros', group_norm=False, n_groups=32):
        if group_norm:
            class GroupNorm(nn.GroupNorm):
                def __init__(self, num_channels):
                    super().__init__(n_groups, num_channels)
            self.norm = GroupNorm
        else:
            self.norm = nn.BatchNorm2d

        super(ResNet_encoder, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, conv_pad_mode=conv_pad_mode)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, conv_pad_mode=conv_pad_mode)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, conv_pad_mode='zeros'):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.norm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, conv_pad_mode, self.norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_pad_mode=conv_pad_mode, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        return x

def resnet50_encoder(**kwargs):
    return ResNet_encoder(Bottleneck, [3, 4, 6, 3], **kwargs)


class Decoder(nn.Module):
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self, h_feat_dec=2048, num_output_channels=7, num_slots=2):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_feat_dec, h_feat_dec),
                nn.ReLU(),
                nn.Linear(h_feat_dec, num_output_channels),
            )
            for _ in range(num_slots)
        ])
 
        # self.multihead_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim, n_heads) for _ in range(num_slots)]) - only useful if you have per node embeddings

    def forward(self, enc_all_outputs):
        out = torch.stack([head(enc_all_outputs) for head in self.heads], dim=1)
        return out

class CrossAttnDecoder(nn.Module):
    def __init__(self, h_feat_dec, num_output_channels=7, num_slots=2, num_heads=4):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(num_slots, h_feat_dec) * 0.02)
        self.cross_attn = nn.MultiheadAttention(h_feat_dec, num_heads, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(h_feat_dec, h_feat_dec), nn.ReLU(), nn.Linear(h_feat_dec, num_output_channels))
            for _ in range(num_slots)
        ])

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2) 

        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1) 
        attended, _ = self.cross_attn(q, tokens, tokens)   

        return torch.stack([h(attended[:, i, :]) for i, h in enumerate(self.heads)], dim=1)
    
# class CrossAttnDecoder(nn.Module):
#     def __init__(self, h_feat_dec, num_output_channels=7, num_slots=2, num_heads=4):
#         super().__init__()
#         self.num_slots = num_slots
#         self.slot_queries = nn.Parameter(torch.randn(num_slots, h_feat_dec) * 0.02)
#         self.cross_attn = nn.MultiheadAttention(h_feat_dec, num_heads, batch_first=True)
#         # SHARED head -- one set of weights, applied to every slot
#         self.head = nn.Sequential(
#             nn.Linear(h_feat_dec, h_feat_dec),
#             nn.ReLU(),
#             nn.Linear(h_feat_dec, num_output_channels),
#         )

#     def forward(self, feat_map):
#         B, C, H, W = feat_map.shape
#         tokens = feat_map.flatten(2).transpose(1, 2)  # (B, H*W, C)

#         q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_slots, C)
#         attended, _ = self.cross_attn(q, tokens, tokens)      # (B, num_slots, C)

#         # apply the SAME head to every slot by folding slots into the batch dim
#         B, S, Hc = attended.shape
#         flat = attended.reshape(B * S, Hc)
#         out = self.head(flat)                # (B*S, num_output_channels)
#         return out.reshape(B, S, -1)          # (B, num_slots, num_output_channels)
    
class EncoderDecoder(nn.Module):
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        enc_all_outputs = self.encoder(data)
        output = self.decoder(enc_all_outputs)

        return output

# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, data):
#         feat_map = self.encoder(data)
#         output = self.decoder(feat_map)
#         return output