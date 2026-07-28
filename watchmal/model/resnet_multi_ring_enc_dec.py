import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import TransformerConv, HeteroConv, GATConv, GATv2Conv, global_add_pool, global_mean_pool

from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch
from typing import Callable
import copy
import numpy as np
import math



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





## implemented  from pytorch
# 
# so i can edit details


# def _get_seq_len(src: torch.Tensor, batch_first: bool) -> int | None:
#     if src.is_nested:
#         return None
#     else:
#         src_size = src.size()
#         if len(src_size) == 2:
#             # unbatched: S, E
#             return src_size[0]
#         else:
#             # batched: B, S, E if batch_first else S, B, E
#             seq_len_pos = 1 if batch_first else 0
#             return src_size[seq_len_pos]


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# def _detect_is_causal_mask(
#     mask: torch.Tensor | None,
#     is_causal: bool | None = None,
#     size: int | None = None,
# ) -> bool:

#     # Prevent type refinement
#     make_causal = is_causal is True

#     if is_causal is None and mask is not None:
#         sz = size if size is not None else mask.size(-2)
#         causal_comparison = _generate_square_subsequent_mask(
#             sz, device=mask.device, dtype=mask.dtype
#         )

#         # Do not use `torch.equal` so we handle batched masks by
#         # broadcasting the comparison.
#         if mask.size() == causal_comparison.size():
#             make_causal = bool((mask == causal_comparison).all())
#         else:
#             make_causal = False

#     return make_causal


class TransformerDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        # memory_mask: torch.Tensor | None = None,
        # tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        # tgt_is_causal: bool | None = None,
        # memory_is_causal: bool = False,
        return_intermediate=False,
    ) -> torch.Tensor:
        
        output = tgt
        intermediate = []

        # seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        # tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                # tgt_mask=tgt_mask,
                # memory_mask=memory_mask,
                # tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                query_pos=query_pos,
                # tgt_is_causal=tgt_is_causal,
                # memory_is_causal=memory_is_causal,
            )
            if return_intermediate:
                intermediate.append(self.norm(output))
                
        if return_intermediate:
            return torch.stack(intermediate) 

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(nn.Module):

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # # Legacy string support for activation function.
        # if isinstance(activation, str):
        #     self.activation = _get_activation_fn(activation)
        # else:
        #     self.activation = activation

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = activation 

    @staticmethod
    def _with_pos_embed(x, pos):
        return x if pos is None else x + pos

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = tgt
        if self.norm_first:
            if query_pos is None:
                # x = x + self._sa_block(
                #     self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
                # )
                # x = x + self._mha_block(
                #     self.norm2(x),
                #     memory,
                #     memory_mask,
                #     memory_key_padding_mask,
                #     memory_is_causal,
                # )

                print('error')
                # print('error')
                # x = x + self._mha_block(
                #     self.norm2(x),
                #     memory,
                #     memory_key_padding_mask,
                # )
                # x = x + self._ff_block(self.norm3(x))
            else:
                x = x + self._sa_block(
                    self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal, query_pos=query_pos
                )
                x = x + self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                    query_pos=query_pos,
                )
                # x = x + self._mha_block(
                #     self.norm2(x),
                #     memory,
                #     memory_key_padding_mask,
                #     query_pos=query_pos,
                # )

                x = x + self._ff_block(self.norm3(x))
        else:
            if query_pos is None:
                x = self.norm1(
                    x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                )
                x = self.norm2(
                    x
                    + self._mha_block(
                        x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                    )
                )
                x = self.norm3(x + self._ff_block(x))
            else:
                x = tgt
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, query_pos=query_pos))
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, query_pos=query_pos))
                x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_pos is None:
            q = k = x
        else:
            q = k = self._with_pos_embed(x, query_pos)
        x = self.self_attn(
            # x,
            # x,
            # x,
            q, k, x, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_pos is not None:
            x = self._with_pos_embed(x, query_pos)

        x = self.multihead_attn(
            x,
            # self._with_pos_embed(x, query_pos),
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

### Competitive slot attention
    # def _mha_block(
    #     self,
    #     x: torch.Tensor,                  # [B, S, D]
    #     mem: torch.Tensor,                # [B, N, D]
    #     attn_mask: torch.Tensor | None,
    #     key_padding_mask: torch.Tensor | None,
    #     is_causal: bool = False,
    #     query_pos: torch.Tensor | None = None,
    #     temperature: float = 1.0,
    # ) -> torch.Tensor:
    #     if query_pos is not None:
    #         x = self._with_pos_embed(x, query_pos)

    #     batch_size, num_slots, embed_dim = x.shape
    #     num_nodes = mem.size(1)
    #     num_heads = self.multihead_attn.num_heads
    #     head_dim = embed_dim // num_heads

    #     q_weight, k_weight, v_weight = (
    #         self.multihead_attn.in_proj_weight.chunk(3)
    #     )

    #     if self.multihead_attn.in_proj_bias is not None:
    #         q_bias, k_bias, v_bias = (
    #             self.multihead_attn.in_proj_bias.chunk(3)
    #         )
    #     else:
    #         q_bias = k_bias = v_bias = None

    #     q = F.linear(x, q_weight, q_bias)
    #     k = F.linear(mem, k_weight, k_bias)
    #     v = F.linear(mem, v_weight, v_bias)

    #     q = q.view(
    #         batch_size, num_slots, num_heads, head_dim
    #     ).transpose(1, 2)

    #     k = k.view(
    #         batch_size, num_nodes, num_heads, head_dim
    #     ).transpose(1, 2)

    #     v = v.view(
    #         batch_size, num_nodes, num_heads, head_dim
    #     ).transpose(1, 2)

    #     # [B, H, S, N]
    #     logits = torch.matmul(
    #         q,
    #         k.transpose(-2, -1),
    #     )

    #     logits = logits / math.sqrt(head_dim)
    #     logits = logits / temperature

    #     if key_padding_mask is not None:
    #         valid_nodes = ~key_padding_mask[:, None, None, :]

    #         masked_logits = logits.masked_fill(
    #             ~valid_nodes,
    #             -1e9,
    #         )
    #     else:
    #         valid_nodes = None
    #         masked_logits = logits

    #     standard_attn = torch.softmax(
    #         masked_logits,
    #         dim=-1,
    #     )

    #     assignments = torch.softmax(
    #         masked_logits,
    #         dim=-2,
    #     )

    #     if valid_nodes is not None:
    #         standard_attn = standard_attn * valid_nodes
    #         assignments = assignments * valid_nodes

    #         standard_attn = standard_attn / (
    #             standard_attn.sum(dim=-1, keepdim=True) + 1e-8
    #         )

    #     competitive_attn = assignments / (
    #         assignments.sum(dim=-1, keepdim=True) + 1e-8
    #     )

    #     alpha = 0.1

    #     attn = (
    #         (1.0 - alpha) * standard_attn
    #         + alpha * competitive_attn
    #     )

    #     attn = F.dropout(
    #         attn,
    #         p=self.multihead_attn.dropout,
    #         training=self.training,
    #     )

    #     output = torch.matmul(attn, v)

    #     output = (
    #         output.transpose(1, 2)
    #         .contiguous()
    #         .view(batch_size, num_slots, embed_dim)
    #     )

    #     output = self.multihead_attn.out_proj(output)

    #     return self.dropout2(output)

#     # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class CrossAttnDecoder(nn.Module):
    def __init__(
        self,
        h_feat_dec,
        num_output_channels=3,
        num_slots=2,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        reinject = True,
        aux_loss=True,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.aux_loss=aux_loss
        self.reinject = reinject
        # self.slot_queries = nn.Parameter(
        #     torch.randn(num_slots, h_feat_dec) * 0.02
        # )

        self.slot_queries = nn.Parameter(torch.randn(num_slots, h_feat_dec)) 

        decoder_layer = TransformerDecoderLayer(
            d_model=h_feat_dec,
            nhead=num_heads,
            dim_feedforward=4 * h_feat_dec,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(h_feat_dec),
        )

        # Shared head across slots
        self.head = nn.Sequential(
            nn.LayerNorm(h_feat_dec),
            nn.Linear(h_feat_dec, h_feat_dec),
            nn.GELU(),
            nn.Linear(h_feat_dec, num_output_channels),
        )

# sep heads
        # self.heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.LayerNorm(h_feat_dec),
        #         nn.Linear(h_feat_dec, h_feat_dec),
        #         nn.GELU(),
        #         nn.Linear(h_feat_dec, num_output_channels),
        #     )
        #     for _ in range(num_slots)
        # ])



## dir loss
        self.output_features = nn.Sequential(
            nn.LayerNorm(h_feat_dec),
            nn.Linear(h_feat_dec, h_feat_dec),
            nn.GELU(),
        )

        self.position_head = nn.Linear(
            h_feat_dec,
            num_output_channels,
        )

        self.direction_head = nn.Linear(
            h_feat_dec,
            3,
        )
# ## sep heads with dir 
#         # self.position_heads = nn.ModuleList([nn.Linear(
#         #     h_feat_dec,
#         #     num_output_channels,
#         # ) for _ in range(num_slots)]) 

#         # self.direction_heads = nn.ModuleList([nn.Linear(
#         #     h_feat_dec,
#         #     3,
#         # ) for _ in range(num_slots)])




    def forward(self, x_dense, mask=None):
        batch_size = x_dense.size(0)
        queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)

        if self.reinject:
            tgt, query_pos = torch.zeros_like(queries), queries
        else:
            tgt, query_pos = queries, None

        decoded = self.decoder(
            tgt=tgt,
            memory=x_dense,
            memory_key_padding_mask=mask,
            query_pos=query_pos,
            return_intermediate=self.aux_loss and self.training,
        )
    #     return self.head(decoded)

    #     # return torch.stack(
    #     #     [
    #     #         self.heads[i](decoded[..., i, :])
    #     #         for i in range(self.num_slots)
    #     #     ],
    #     #     dim=-2,
    #     # )

# direction loss run

        features = self.output_features(decoded)

        positions = self.position_head(features)

        directions = F.normalize(
            self.direction_head(features),
            p=2,
            dim=-1,
            eps=1e-8,
        )

        return torch.cat(
            [positions, directions],
            dim=-1,
        )

        # features = self.output_features(decoded)

        # positions = torch.stack(
        #     [
        #         self.position_heads[i](features[..., i, :])
        #         for i in range(self.num_slots)
        #     ],
        #     dim=-2,
        # )

        # dirs = torch.stack(
        #     [
        #         self.direction_heads[i](features[..., i, :])
        #         for i in range(self.num_slots)
        #     ],
        #     dim=-2,
        # )

        # directions = F.normalize(
        #     dirs,
        #     p=2,
        #     dim=-1,
        #     eps=1e-8,
        # )

        # return torch.cat(
        #     [positions, directions],
        #     dim=-1,
        # )



class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, enc_channels=2048, h_feat_dec=128):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_proj = nn.Conv2d(enc_channels, h_feat_dec, kernel_size=1)

    def forward(self, data):
        feat = self.encoder(data)              
        feat = self.input_proj(feat)       
        B, C, H, W = feat.shape
        x_dense = feat.flatten(2).transpose(1, 2)
        return self.decoder(x_dense)