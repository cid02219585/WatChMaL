### NonHierGAT and HierTrans encoders with a decoder built for combined ring count and position prediction for variable single-ring and two-ring events
# Architecture transferred from the fixed two ring case, only made changes to allow for ring count prediction (labelled DETRtransformer to signify this, but does not use the code of the published DETR transformer, uses the standard transformer decoder from pytorch with adaptations)
# submission scripts are in /vols/hyperk/users/sc4422/first_run/scripts/gnn/var_multi

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, HeteroConv, GATConv
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean


class NonHierGATEncoder(nn.Module):
    def __init__(
        self,
        pmt_in,
        mpmt_in,
        virtual_in,
        h_feat,
        dropout=0.0,
        num_heads=4,
        num_layers=3,
        aggr="sum",
    ):
        super().__init__()

        self.dropout = dropout

        self.pmt_encoder = nn.Linear(pmt_in, h_feat)
        self.mpmt_encoder = nn.Linear(mpmt_in, h_feat)
        self.virtual_encoder = nn.Linear(virtual_in, h_feat)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        ("pmt", "belongs_to", "mpmt"):
                            GATConv(
                                (h_feat, h_feat),
                                h_feat,
                                heads=num_heads,
                                concat=False,
                                add_self_loops=False,
                            ),

                        ("mpmt", "neighbours", "mpmt"):
                            GATConv(
                                h_feat,
                                h_feat,
                                heads=num_heads,
                                concat=False,
                            ),

                        ("mpmt", "contains", "pmt"):
                            GATConv(
                                (h_feat, h_feat),
                                h_feat,
                                heads=num_heads,
                                concat=False,
                                add_self_loops=False,
                            ),

                        ("pmt", "neighbours", "pmt"):
                            GATConv(
                                h_feat,
                                h_feat,
                                heads=num_heads,
                                concat=False,
                            ),

                        ("mpmt", "reports_to", "virtual_node"):
                            GATConv(
                                (h_feat, h_feat),
                                h_feat,
                                heads=num_heads,
                                concat=False,
                                add_self_loops=False,
                            ),

                        ("virtual_node", "attends_to", "mpmt"):
                            GATConv(
                                (h_feat, h_feat),
                                h_feat,
                                heads=num_heads,
                                concat=False,
                                add_self_loops=False,
                            ),
                    },
                    aggr=aggr,
                )
            )

            self.norms.append(
                nn.ModuleDict(
                    {
                        "pmt": nn.LayerNorm(h_feat),
                        "mpmt": nn.LayerNorm(h_feat),
                        "virtual_node": nn.LayerNorm(h_feat),
                    }
                )
            )

    def forward(self, data):
        x_dict = {
            "pmt": self.pmt_encoder(data["pmt"].x),
            "mpmt": self.mpmt_encoder(data["mpmt"].x),
            "virtual_node": self.virtual_encoder(data["virtual_node"].x),
        }

        edge_index_dict = {
            ("pmt", "belongs_to", "mpmt"):
                data["pmt", "belongs_to", "mpmt"].edge_index,

            ("mpmt", "contains", "pmt"):
                data["mpmt", "contains", "pmt"].edge_index,

            ("mpmt", "neighbours", "mpmt"):
                data["mpmt", "neighbours", "mpmt"].edge_index,

            ("pmt", "neighbours", "pmt"):
                data["pmt", "neighbours", "pmt"].edge_index,

            ("mpmt", "reports_to", "virtual_node"):
                data["mpmt", "reports_to", "virtual_node"].edge_index,

            ("virtual_node", "attends_to", "mpmt"):
                data["virtual_node", "attends_to", "mpmt"].edge_index,
        }

        for conv, norms in zip(self.convs, self.norms):
            x_new = conv(x_dict, edge_index_dict)

            x_dict = {
                key: norms[key](
                    x_dict[key]
                    + F.dropout(
                        F.gelu(update),
                        p=self.dropout,
                        training=self.training,
                    )
                )
                for key, update in x_new.items()
            }

        return x_dict["mpmt"], data["mpmt"].batch


class HierTransEncoder(nn.Module):
    def __init__(
        self,
        pmt_in,
        mpmt_in,
        virtual_in,
        h_feat,
        num_pmt_layers=1,
        num_mpmt_layers=3,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()

        self.dropout = dropout

        self.pmt_encoder = nn.Linear(pmt_in, h_feat)
        self.mpmt_encoder = nn.Linear(mpmt_in, h_feat)
        self.virtual_encoder = nn.Linear(virtual_in, h_feat)

        self.pmt_layers = nn.ModuleList([
            TransformerConv(
                h_feat,
                h_feat,
                heads=num_heads,
                concat=False,
            )
            for _ in range(num_pmt_layers)
        ])

        self.mpmt_layers = nn.ModuleList([
            TransformerConv(
                h_feat,
                h_feat,
                heads=num_heads,
                concat=False,
            )
            for _ in range(num_mpmt_layers)
        ])

        self.pmt_norms = nn.ModuleList([
            nn.LayerNorm(h_feat)
            for _ in range(num_pmt_layers)
        ])

        self.mpmt_norms = nn.ModuleList([
            nn.LayerNorm(h_feat)
            for _ in range(num_mpmt_layers)
        ])

        self.mpmt_norm = nn.LayerNorm(h_feat)
        self.virtual_norm = nn.LayerNorm(h_feat)

    def forward(self, data):
        x_p = self.pmt_encoder(data["pmt"].x)

        pmt_edges = data[
            "pmt",
            "neighbours",
            "pmt",
        ].edge_index

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

        belongs_to = data[
            "pmt",
            "belongs_to",
            "mpmt",
        ].edge_index

        x_m = scatter_mean(
            x_p,
            belongs_to[1],
            dim=0,
            dim_size=data["mpmt"].x.size(0),
        )

        x_m = self.mpmt_norm(x_m)

        x_v = self.virtual_encoder(
            data["virtual_node"].x
        )

        x_v_broadcast = self.virtual_norm(
            x_v[data["mpmt"].batch]
        )

        x_m = x_m + x_v_broadcast

        mpmt_edges = data[
            "mpmt",
            "neighbours",
            "mpmt",
        ].edge_index

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

        return x_m, data["mpmt"].batch

def _get_clones(module, n):
    return nn.ModuleList([
        copy.deepcopy(module)
        for _ in range(n)
    ])


class DETRTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__()

        self.layers = _get_clones(
            decoder_layer,
            num_layers,
        )

        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
        query_pos=None,
        return_intermediate=False,
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                query_pos=query_pos,
            )

            if return_intermediate:
                if self.norm is not None:
                    intermediate.append(
                        self.norm(output)
                    )
                else:
                    intermediate.append(output)

        if return_intermediate:
            return torch.stack(
                intermediate,
                dim=0,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class DETRTransformerDecoderLayer(nn.Module): # introducing slot comp as a flag
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        slot_competition=False,
        competition_alpha=0.1,
        competition_temperature=1.0,
    ):
        super().__init__()

        self.slot_competition = slot_competition
        self.competition_alpha = competition_alpha
        self.competition_temperature = competition_temperature

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward,
        )

        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(
                f"Unsupported activation: {activation}"
            )

    @staticmethod
    def _with_pos_embed(x, pos):
        if pos is None:
            return x

        return x + pos

    def _competitive_cross_attention(
        self,
        query,
        memory,
        key_padding_mask=None,
    ):
        batch_size, num_slots, embed_dim = query.shape
        num_nodes = memory.size(1)
        num_heads = self.cross_attn.num_heads
        head_dim = embed_dim // num_heads

        q_weight, k_weight, v_weight = (
            self.cross_attn.in_proj_weight.chunk(3)
        )

        if self.cross_attn.in_proj_bias is not None:
            q_bias, k_bias, v_bias = (
                self.cross_attn.in_proj_bias.chunk(3)
            )
        else:
            q_bias = k_bias = v_bias = None

        q = F.linear(query, q_weight, q_bias)
        k = F.linear(memory, k_weight, k_bias)
        v = F.linear(memory, v_weight, v_bias)

        q = q.view(
            batch_size,
            num_slots,
            num_heads,
            head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            num_nodes,
            num_heads,
            head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            num_nodes,
            num_heads,
            head_dim,
        ).transpose(1, 2)

        # [B, H, S, N]
        logits = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        logits = logits / math.sqrt(head_dim)
        logits = logits / self.competition_temperature

        if key_padding_mask is not None:
            valid_nodes = ~key_padding_mask[:, None, None, :]

            masked_logits = logits.masked_fill(
                ~valid_nodes,
                -1e9,
            )
        else:
            valid_nodes = None
            masked_logits = logits

        standard_attn = torch.softmax(
            masked_logits,
            dim=-1,
        )

        assignments = torch.softmax(
            masked_logits,
            dim=-2,
        )

        if valid_nodes is not None:
            standard_attn = standard_attn * valid_nodes
            assignments = assignments * valid_nodes

            standard_attn = standard_attn / (
                standard_attn.sum(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-8
            )

        competitive_attn = assignments / (
            assignments.sum(
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )

        attn = (
            (1.0 - self.competition_alpha)
            * standard_attn
            + self.competition_alpha
            * competitive_attn
        )

        attn = F.dropout(
            attn,
            p=self.cross_attn.dropout,
            training=self.training,
        )

        output = torch.matmul(
            attn,
            v,
        )

        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                num_slots,
                embed_dim,
            )
        )

        return self.cross_attn.out_proj(
            output
        )

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
        query_pos=None,
    ):
        x = tgt

        x_norm = self.norm1(x)

        q = k = self._with_pos_embed(
            x_norm,
            query_pos,
        )

        self_attn_out = self.self_attn(
            q,
            k,
            x_norm,
            need_weights=False,
        )[0]

        x = (
            x
            + self.dropout1(
                self_attn_out
            )
        )

        x_norm = self.norm2(x)

        q = self._with_pos_embed(
            x_norm,
            query_pos,
        )

        if self.slot_competition:
            cross_attn_out = self._competitive_cross_attention(
                q,
                memory,
                key_padding_mask=memory_key_padding_mask,
            )
        else:
            cross_attn_out = self.cross_attn(
                q,
                memory,
                memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]

        x = (
            x
            + self.dropout2(
                cross_attn_out
            )
        )

        x_norm = self.norm3(x)

        ff = self.linear2(
            self.ff_dropout(
                self.activation(
                    self.linear1(
                        x_norm
                    )
                )
            )
        )

        x = (
            x
            + self.dropout3(ff)
        )

        return x

class DETRCrossAttnDecoder(nn.Module): # with all tested ablations e.g. aux loss, sep heads as flags

    def __init__(
        self,
        h_feat_dec,
        num_slots=2,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        reinject=True,
        aux_loss=True,
        dim_feedforward=None,
        direction_loss=True,
        separate_heads=False,
        slot_competition=False,
        competition_alpha=0.1,
        competition_temperature=1.0,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.reinject = reinject
        self.aux_loss = aux_loss
        self.direction_loss = direction_loss
        self.separate_heads = separate_heads
        self.slot_competition = slot_competition

        if dim_feedforward is None:
            dim_feedforward = (
                4 * h_feat_dec
            )

        self.slot_queries = nn.Parameter(
            torch.randn(
                num_slots,
                h_feat_dec,
            )
        )

        decoder_layer = DETRTransformerDecoderLayer(
            d_model=h_feat_dec,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            slot_competition=slot_competition,
            competition_alpha=competition_alpha,
            competition_temperature=competition_temperature,
        )

        self.decoder = DETRTransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(h_feat_dec),
        )

        self.output_features = nn.Sequential(
            nn.LayerNorm(h_feat_dec),
            nn.Linear(
                h_feat_dec,
                h_feat_dec,
            ),
            nn.GELU(),
        )

        if self.separate_heads:
            self.position_heads = nn.ModuleList([
                nn.Linear(
                    h_feat_dec,
                    3,
                )
                for _ in range(num_slots)
            ])

            self.ring_heads = nn.ModuleList([
                nn.Linear(
                    h_feat_dec,
                    1,
                )
                for _ in range(num_slots)
            ])

            if self.direction_loss:
                self.direction_heads = nn.ModuleList([
                    nn.Linear(
                        h_feat_dec,
                        3,
                    )
                    for _ in range(num_slots)
                ])
        else:
            self.position_head = nn.Linear(
                h_feat_dec,
                3,
            )

            self.ring_head = nn.Linear(
                h_feat_dec,
                1,
            )

            if self.direction_loss:
                self.direction_head = nn.Linear(
                    h_feat_dec,
                    3,
                )

    def _apply_slot_heads(
        self,
        features,
    ):
        if self.separate_heads:
            positions = torch.stack(
                [
                    self.position_heads[i](
                        features[..., i, :]
                    )
                    for i in range(
                        self.num_slots
                    )
                ],
                dim=-2,
            )

            ring_logits = torch.stack(
                [
                    self.ring_heads[i](
                        features[..., i, :]
                    ).squeeze(-1)
                    for i in range(
                        self.num_slots
                    )
                ],
                dim=-1,
            )

            directions = None

            if self.direction_loss:
                directions = torch.stack(
                    [
                        self.direction_heads[i](
                            features[..., i, :]
                        )
                        for i in range(
                            self.num_slots
                        )
                    ],
                    dim=-2,
                )
        else:
            positions = self.position_head(
                features
            )

            ring_logits = (
                self.ring_head(
                    features
                )
                .squeeze(-1)
            )

            directions = None

            if self.direction_loss:
                directions = self.direction_head(
                    features
                )

        if directions is not None:
            directions = F.normalize(
                directions,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        return (
            positions,
            ring_logits,
            directions,
        )

    def forward(
        self,
        x_m,
        batch,
    ):
        x_dense, mask = to_dense_batch(
            x_m,
            batch,
        )

        batch_size = (
            x_dense.size(0)
        )

        queries = (
            self.slot_queries
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        if self.reinject:
            tgt = torch.zeros_like(
                queries
            )
            query_pos = queries
        else:
            tgt = queries
            query_pos = None

        return_intermediate = (
            self.aux_loss
            and self.training
        )

        decoded = self.decoder(
            tgt=tgt,
            memory=x_dense,
            memory_key_padding_mask=~mask,
            query_pos=query_pos,
            return_intermediate=return_intermediate,
        )

        features = self.output_features(
            decoded
        )

        (
            positions,
            ring_logits,
            directions,
        ) = self._apply_slot_heads(
            features
        )

        output = {
            "pred_positions":
                positions,

            "pred_logits":
                ring_logits,
        }

        if directions is not None:
            output[
                "pred_directions"
            ] = directions

        return output

class EncoderDecoder(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        data,
    ):
        encoder_output, batch = (
            self.encoder(data)
        )

        return self.decoder(
            encoder_output,
            batch,
        )
