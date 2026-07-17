import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import TransformerConv, HeteroConv, GATConv, GATv2Conv, global_add_pool, global_mean_pool

from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch
from typing import Callable
import copy


class NonHierGAT_encoder(nn.Module):
    def __init__(self,
        pmt_in, 
        mpmt_in, 
        virtual_in, 
        h_feat, 
        dropout,
        num_heads=4,
        num_layers=3, 
        aggr='sum',
    ):
        super().__init__()
        self.dropout = dropout

        self.pmt_encoder = nn.Linear(pmt_in, h_feat)
        self.mpmt_encoder = nn.Linear(mpmt_in, h_feat)
        self.virtual_encoder = nn.Linear(virtual_in, h_feat)

        self.convs = torch.nn.ModuleList([])

        for _ in range(num_layers):
            conv = HeteroConv({
                ('pmt', 'belongs_to', 'mpmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
                ('mpmt', 'neighbours', 'mpmt'): GATConv(h_feat, h_feat, heads=num_heads, concat=False),
                ('mpmt', 'contains', 'pmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
                ('pmt', 'neighbours', 'pmt'): GATConv(h_feat, h_feat, heads=num_heads, concat=False),
                ('mpmt', 'reports_to', 'virtual_node'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
                ('virtual_node', 'attends_to', 'mpmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
            }, aggr=aggr)
            
            self.convs.append(conv)

            self.norms = nn.ModuleList([
                nn.ModuleDict({
                    "pmt": nn.LayerNorm(h_feat),
                    "mpmt": nn.LayerNorm(h_feat),
                    "virtual_node": nn.LayerNorm(h_feat),
                })
                for _ in range(num_layers)
            ])

    def forward(self, data):
        x_dict = {
            'pmt':  self.pmt_encoder(data['pmt'].x),
            'mpmt': self.mpmt_encoder(data['mpmt'].x),
            'virtual_node': self.virtual_encoder(data['virtual_node'].x),
        }
        edge_index_dict = {
            ('pmt',  'belongs_to', 'mpmt'): data['pmt',  'belongs_to', 'mpmt'].edge_index,
            ('mpmt', 'contains',   'pmt'):  data['mpmt', 'contains',   'pmt'].edge_index,
            ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
            ('pmt', 'neighbours', 'pmt'): data['pmt', 'neighbours', 'pmt'].edge_index,
            ('mpmt', 'reports_to', 'virtual_node'): data['mpmt', 'reports_to', 'virtual_node'].edge_index,
            ('virtual_node', 'attends_to', 'mpmt'): data['virtual_node', 'attends_to', 'mpmt'].edge_index,
        }

        # for conv in self.convs:
        #     x_dict_new = conv(x_dict, edge_index_dict)
        #     x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training) + x_dict[key]
        #             for key, x in x_dict_new.items()}

        for conv, norms in zip(self.convs, self.norms):
            x_dict_new = conv(x_dict, edge_index_dict)

            x_dict = {
                key: norms[key](
                    x_dict[key]
                    + F.dropout(
                        F.gelu(x),
                        p=self.dropout,
                        training=self.training,
                    )
                )
                for key, x in x_dict_new.items()
            }

                # return global_add_pool(x_dict['mpmt'], data['mpmt'].batch)
        return x_dict['mpmt'], data['mpmt'].batch

class HierTrans_encoder(nn.Module):
    def __init__(self, pmt_in, mpmt_in, virtual_in, h_feat,
                 num_pmt_layers=1, num_mpmt_layers=3, num_heads=4, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        self.pmt_encoder  = nn.Linear(pmt_in,  h_feat)
        self.mpmt_encoder = nn.Linear(mpmt_in, h_feat)
        self.virtual_encoder = nn.Linear(virtual_in, h_feat)

        self.pmt_layers = nn.ModuleList([
            TransformerConv(h_feat, h_feat, heads=num_heads, concat=False)
            for _ in range(num_pmt_layers)
        ])

        self.mpmt_layers = nn.ModuleList([
            TransformerConv(h_feat, h_feat, heads=num_heads, concat=False)
            for _ in range(num_mpmt_layers)
        ])

        self.mpmt_norm    = nn.LayerNorm(h_feat)
        self.virtual_norm = nn.LayerNorm(h_feat)
        
        self.pmt_norms = nn.ModuleList([
            nn.LayerNorm(h_feat)
            for _ in range(num_pmt_layers)
        ])

        self.mpmt_norms = nn.ModuleList([
            nn.LayerNorm(h_feat)
            for _ in range(num_mpmt_layers)
        ])
    def forward(self, data):
        x_p = self.pmt_encoder(data['pmt'].x)
        x_v = self.virtual_encoder(data['virtual_node'].x)

        pmt_edges  = data['pmt',  'neighbours', 'pmt'].edge_index
        belongs_to = data['pmt',  'belongs_to', 'mpmt'].edge_index
        mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index

        # for conv in self.pmt_layers:
        #     x_p = F.dropout(F.relu(conv(x_p, pmt_edges)),p=self.dropout, training=self.training) + x_p

        for conv, norm in zip(self.pmt_layers, self.pmt_norms):
            update = conv(x_p, pmt_edges)
            x_p = norm(
                x_p
                + F.dropout(
                    F.gelu(update),
                    p=self.dropout,
                    training=self.training,
                )
            )

        x_m = scatter_mean(
            x_p,
            belongs_to[1],
            dim=0,
            dim_size=data['mpmt'].x.size(0),
            )

        x_m = self.mpmt_norm(x_m)
        x_v_broadcast = self.virtual_norm(x_v[data['mpmt'].batch]) 
        x_m = x_m + x_v_broadcast 

        for conv, norm in zip(self.mpmt_layers, self.mpmt_norms):
            update = conv(x_m, mpmt_edges)
            x_m = norm(
                x_m
                + F.dropout(
                    F.gelu(update),
                    p=self.dropout,
                    training=self.training,
                )
            )

        # for conv in self.mpmt_layers:
        #     x_m = F.dropout(F.relu(conv(x_m, mpmt_edges)),p=self.dropout, training=self.training) + x_m

        # out = global_add_pool(x_m, data['mpmt'].batch)
        # return out

        return x_m, data['mpmt'].batch

class Decoder(nn.Module):
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self, h_feat_dec=128, num_output_channels=7, num_slots=2):
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

# class CrossAttnDecoder(nn.Module):
#     def __init__(
#         self,
#         h_feat_dec,
#         num_output_channels=3,
#         num_slots=2,
#         num_heads=4,
#         num_layers=3,
#         dropout=0.1,
#     ):
#         super().__init__()

#         self.num_slots = num_slots

#         self.slot_queries = nn.Parameter(
#             torch.randn(num_slots, h_feat_dec) * 0.02
#         )

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=h_feat_dec,
#             nhead=num_heads,
#             dim_feedforward=4 * h_feat_dec,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )

#         self.decoder = nn.TransformerDecoder(
#             decoder_layer,
#             num_layers=num_layers,
#             norm=nn.LayerNorm(h_feat_dec),
#         )

#         # Shared head across slots
#         self.head = nn.Sequential(
#             nn.LayerNorm(h_feat_dec),
#             nn.Linear(h_feat_dec, h_feat_dec),
#             nn.GELU(),
#             nn.Linear(h_feat_dec, num_output_channels),
#         )
#     def forward(self, x_m, batch):
#         x_dense, mask = to_dense_batch(x_m, batch)

#         batch_size = x_dense.size(0)

#         queries = self.slot_queries.unsqueeze(0).expand(
#             batch_size, -1, -1
#         )

#         decoded = self.decoder(
#             tgt=queries,
#             memory=x_dense,
#             memory_key_padding_mask=~mask,
#         )

#         return self.head(decoded)


# class CrossAttnDecoder(nn.Module):
#     def __init__(
#         self,
#         h_feat_dec,
#         num_output_channels=3,
#         num_slots=2,
#         num_heads=4,
#         num_layers=3,
#         dropout=0.1,
#     ):
#         super().__init__()

#         self.num_slots = num_slots

#         self.slot_queries = nn.Parameter(
#             torch.randn(num_slots, h_feat_dec) * 0.02
#         )

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=h_feat_dec,
#             nhead=num_heads,
#             dim_feedforward=4 * h_feat_dec,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )

#         self.decoder = nn.TransformerDecoder(
#             decoder_layer,
#             num_layers=num_layers,
#             norm=nn.LayerNorm(h_feat_dec),
#         )

#         self.heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.LayerNorm(h_feat_dec),
#                 nn.Linear(h_feat_dec, h_feat_dec),
#                 nn.GELU(),
#                 nn.Linear(h_feat_dec, num_output_channels),
#             )
#             for _ in range(num_slots)
#         ])

#     def forward(self, x_m, batch):
#         x_dense, mask = to_dense_batch(x_m, batch)

#         queries = self.slot_queries.unsqueeze(0).expand(
#             x_dense.size(0), -1, -1
#         )

#         decoded = self.decoder(
#             tgt=queries,
#             memory=x_dense,
#             memory_key_padding_mask=~mask,
#         )

#         return torch.stack(
#             [
#                 self.heads[i](decoded[:, i])
#                 for i in range(self.num_slots)
#             ],
#             dim=1,
#         )
### writing it out so i can make chagnes


















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
        # tgt_is_causal: bool | None = None,
        # memory_is_causal: bool = False,
    ) -> torch.Tensor:
        
        output = tgt

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
                # tgt_is_causal=tgt_is_causal,
                # memory_is_causal=memory_is_causal,
            )

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
    ) -> torch.Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
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

        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
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
    ) -> torch.Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
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
    ):
        super().__init__()

        self.num_slots = num_slots

        self.slot_queries = nn.Parameter(
            torch.randn(num_slots, h_feat_dec) * 0.02
        )

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
    def forward(self, x_m, batch):
        x_dense, mask = to_dense_batch(x_m, batch)

        batch_size = x_dense.size(0)

        queries = self.slot_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        decoded = self.decoder(
            tgt=queries,
            memory=x_dense,
            memory_key_padding_mask=~mask,
        )

        return self.head(decoded)



    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        enc_all_outputs, batch = self.encoder(data)
        output = self.decoder(enc_all_outputs, batch)

        return output