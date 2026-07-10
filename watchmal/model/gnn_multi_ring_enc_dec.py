import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import TransformerConv, HeteroConv, GATConv, GATv2Conv, global_add_pool, global_mean_pool

from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch


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

        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training) + x_dict[key]
                    for key, x in x_dict_new.items()}

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

    def forward(self, data):
        x_p = self.pmt_encoder(data['pmt'].x)
        x_v = self.virtual_encoder(data['virtual_node'].x)

        pmt_edges  = data['pmt',  'neighbours', 'pmt'].edge_index
        belongs_to = data['pmt',  'belongs_to', 'mpmt'].edge_index
        mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index

        for conv in self.pmt_layers:
            x_p = F.dropout(F.relu(conv(x_p, pmt_edges)),p=self.dropout, training=self.training) + x_p

        x_m = scatter_mean(
            x_p,
            belongs_to[1],
            dim=0,
            dim_size=data['mpmt'].x.size(0),
            )

        x_m = self.mpmt_norm(x_m)
        x_v_broadcast = self.virtual_norm(x_v[data['mpmt'].batch]) 
        x_m = x_m + x_v_broadcast 

        for conv in self.mpmt_layers:
            x_m = F.dropout(F.relu(conv(x_m, mpmt_edges)),p=self.dropout, training=self.training) + x_m

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

class CrossAttnDecoder(nn.Module):
    def __init__(self, h_feat_dec, num_output_channels=7, num_slots=2, num_heads=4):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(num_slots, h_feat_dec) * 0.02)
        self.cross_attn = nn.MultiheadAttention(h_feat_dec, num_heads, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(h_feat_dec, h_feat_dec), nn.ReLU(), nn.Linear(h_feat_dec, num_output_channels))
            for _ in range(num_slots)
        ])

    def forward(self, x_m, batch):
        x_dense, mask = to_dense_batch(x_m, batch)  
        B = x_dense.size(0)
        key_padding_mask = ~mask

        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.cross_attn(q, x_dense, x_dense, key_padding_mask=key_padding_mask)

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

#     def forward(self, x_m, batch):
#         x_dense, mask = to_dense_batch(x_m, batch)
#         B = x_dense.size(0)
#         key_padding_mask = ~mask

#         q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
#         attended, _ = self.cross_attn(q, x_dense, x_dense, key_padding_mask=key_padding_mask)
#         # attended: (B, num_slots, h_feat_dec)

#         # apply the SAME head to every slot by folding slots into the batch dim
#         B, S, H = attended.shape
#         flat = attended.reshape(B * S, H)
#         out = self.head(flat)                     # (B*S, num_output_channels)
#         return out.reshape(B, S, -1)               # (B, num_slots, num_output_channels)

class CrossAttnDecoder(nn.Module):
    def __init__(self, h_feat_dec, num_output_channels=7, num_slots=2, num_heads=4):
        super().__init__()
        self.num_slots = num_slots
        self.slot_queries = nn.Parameter(torch.randn(num_slots, h_feat_dec) * 0.02)
        self.cross_attn = nn.MultiheadAttention(h_feat_dec, num_heads, batch_first=True)
        # SHARED head -- one set of weights, applied to every slot
        self.head = nn.Sequential(
            nn.Linear(h_feat_dec, h_feat_dec),
            nn.ReLU(),
            nn.Linear(h_feat_dec, num_output_channels),
        )

    def forward(self, x_m, batch):
        x_dense, mask = to_dense_batch(x_m, batch)
        B = x_dense.size(0)
        key_padding_mask = ~mask

        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.cross_attn(q, x_dense, x_dense, key_padding_mask=key_padding_mask)
        # attended: (B, num_slots, h_feat_dec)

        # apply the SAME head to every slot by folding slots into the batch dim
        B, S, H = attended.shape
        flat = attended.reshape(B * S, H)
        out = self.head(flat)                     # (B*S, num_output_channels)
        return out.reshape(B, S, -1)               # (B, num_slots, num_output_channels)

class EncoderDecoder(nn.Module):
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        enc_all_outputs,batch = self.encoder(data)
        output = self.decoder(enc_all_outputs,batch)

        return output
    
# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, data):
#         enc_all_outputs, batch = self.encoder(data)
#         output = self.decoder(enc_all_outputs, batch)

#         return output