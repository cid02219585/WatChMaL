### Heterogeneous graph based model attempts 

# 1. Systematic study of het gnns (hierarchy and attention type) - produced the successful final models (NonHierGAT and HierTrans)
# submission scripts are in /vols/hyperk/users/sc4422/first_run/scripts/gnn/exploration

import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import TransformerConv, HeteroConv, GATConv, GATv2Conv, global_add_pool, global_mean_pool

from torch_scatter import scatter_mean

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            # nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.net(x)

class NonHierGAT(nn.Module):
    def __init__(self,
        pmt_in, 
        mpmt_in, 
        virtual_in, 
        h_feat, 
        num_output_channels, 
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

        # self.pmt_encoder = NodeEncoder(pmt_in, h_feat)
        # self.mpmt_encoder = NodeEncoder(mpmt_in, h_feat)

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

        self.out_layer = nn.Sequential( # single layer can only produce a linear mapping to output 
            nn.Linear(h_feat, h_feat),
            nn.ReLU(),
            nn.Linear(h_feat, num_output_channels),
        )

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
            # x_dict = conv(x_dict, edge_index_dict)
            # x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training)
            #           for key, x in x_dict.items()}
            x_dict_new = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training) + x_dict[key]
                    for key, x in x_dict_new.items()}

        out = global_add_pool(x_dict['mpmt'], data['mpmt'].batch)
        # out = x_dict['virtual_node']

        return self.out_layer(out)


class HierGAT(nn.Module):
    def __init__(self, pmt_in, mpmt_in, h_feat, num_output_channels,
                 num_pmt_layers=1, num_mpmt_layers=3, num_heads=4, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        # self.pmt_encoder  = nn.Linear(pmt_in,  h_feat)
        # self.mpmt_encoder = nn.Linear(mpmt_in, h_feat)

        self.pmt_encoder  = NodeEncoder(pmt_in,  h_feat)
        self.mpmt_encoder = NodeEncoder(mpmt_in, h_feat)

        self.pmt_layers = nn.ModuleList([
            GATConv(h_feat, h_feat, heads=num_heads, concat=False)
            for _ in range(num_pmt_layers)
        ])

        # self.pool_conv = GATConv((h_feat, h_feat), h_feat, heads=num_heads,
        #                          concat=False, add_self_loops=False)

        
        # self.pool_proj = nn.Linear(2*h_feat, h_feat)

        self.mpmt_layers = nn.ModuleList([
            GATConv(h_feat, h_feat, heads=num_heads, concat=False)
            for _ in range(num_mpmt_layers)
        ])

        self.out_layer = nn.Sequential(
            nn.Linear(h_feat, h_feat),
            nn.ReLU(),
            nn.Linear(h_feat, num_output_channels),
        )

    def forward(self, data):
        x_p = self.pmt_encoder(data['pmt'].x)
        # x_m = self.mpmt_encoder(data['mpmt'].x)

        pmt_edges  = data['pmt',  'neighbours', 'pmt'].edge_index
        belongs_to = data['pmt',  'belongs_to', 'mpmt'].edge_index
        mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index

        # stage 1: local intra-mPMT
        for conv in self.pmt_layers:
            x_p = F.dropout(F.relu(conv(x_p, pmt_edges)),p=self.dropout, training=self.training)

        # # stage 2: attention-pooled hand-off (mpmt features are the queries)
        # x_m = F.dropout(F.relu(self.pool_conv(
        #     (x_p, x_m), belongs_to,
        #     size=(x_p.size(0), x_m.size(0)),
        # )),p=self.dropout, training=self.training)

        x_m = scatter_mean(
            x_p,
            belongs_to[1],
            dim=0,
            dim_size=data['mpmt'].x.size(0),
            )
        
        # x_m = F.relu(self.pool_proj(torch.cat([x_m, x_m_frompmt], dim=-1)))

        # stage 3: global inter-mPMT
        for conv in self.mpmt_layers:
            x_m = F.dropout(F.relu(conv(x_m, mpmt_edges)),p=self.dropout, training=self.training)

        out = global_add_pool(x_m, data['mpmt'].batch)
        return self.out_layer(out)


class NonHierTrans(nn.Module):
    def __init__(self,
        pmt_in, 
        mpmt_in, 
        h_feat, 
        num_output_channels, 
        dropout,
        num_heads=4,
        num_layers=3, 
        aggr='sum',
    ):
        super().__init__()
        self.dropout = dropout

        # self.pmt_encoder = nn.Linear(pmt_in, h_feat)
        # self.mpmt_encoder = nn.Linear(mpmt_in, h_feat)

        self.pmt_encoder = NodeEncoder(pmt_in, h_feat)
        self.mpmt_encoder = NodeEncoder(mpmt_in, h_feat)

        self.convs = torch.nn.ModuleList([])

        for _ in range(num_layers):
            conv = HeteroConv({
                ('pmt', 'belongs_to', 'mpmt'): TransformerConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False),
                ('mpmt', 'neighbours', 'mpmt'): TransformerConv(h_feat, h_feat, heads=num_heads, concat=False),
                ('mpmt', 'contains', 'pmt'): TransformerConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False),
                ('pmt', 'neighbours', 'pmt'): TransformerConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False),
            }, aggr=aggr)
            
            self.convs.append(conv)

        self.out_layer = nn.Sequential( # single layer can only produce a linear mapping to output 
            nn.Linear(h_feat, h_feat),
            nn.ReLU(),
            nn.Linear(h_feat, num_output_channels),
        )

    def forward(self, data):
        x_dict = {
            'pmt':  self.pmt_encoder(data['pmt'].x),
            'mpmt': self.mpmt_encoder(data['mpmt'].x),
        }
        edge_index_dict = {
            ('pmt',  'belongs_to', 'mpmt'): data['pmt',  'belongs_to', 'mpmt'].edge_index,
            ('mpmt', 'contains',   'pmt'):  data['mpmt', 'contains',   'pmt'].edge_index,
            ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
            ('pmt', 'neighbours', 'pmt'): data['pmt', 'neighbours', 'pmt'].edge_index,
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training)
                      for key, x in x_dict.items()}

        out = global_add_pool(x_dict['mpmt'], data['mpmt'].batch)

        return self.out_layer(out)


class HierTrans(nn.Module):
    def __init__(self, pmt_in, mpmt_in, virtual_in, h_feat, num_output_channels,
                 num_pmt_layers=1, num_mpmt_layers=3, num_heads=4, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        self.pmt_encoder  = nn.Linear(pmt_in,  h_feat)
        self.mpmt_encoder = nn.Linear(mpmt_in, h_feat)
        self.virtual_encoder = nn.Linear(virtual_in, h_feat)

        # self.pmt_encoder  = NodeEncoder(pmt_in,  h_feat)
        # self.mpmt_encoder = NodeEncoder(mpmt_in, h_feat)
        # self.pmt_encoder_norm = nn.LayerNorm(h_feat)

        # print('Node enc')

        self.pmt_layers = nn.ModuleList([
            TransformerConv(h_feat, h_feat, heads=num_heads, concat=False)
            for _ in range(num_pmt_layers)
        ])

        # self.pool_conv = TransformerConv((h_feat, h_feat), h_feat, heads=num_heads,
        #                          concat=False)
        # self.pool_proj = nn.Linear(2*h_feat, h_feat)

        self.mpmt_layers = nn.ModuleList([
            TransformerConv(h_feat, h_feat, heads=num_heads, concat=False)
            for _ in range(num_mpmt_layers)
        ])

        self.mpmt_norm    = nn.LayerNorm(h_feat)
        self.virtual_norm = nn.LayerNorm(h_feat)

        # self.reports_to_conv = TransformerConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False)
        # self.attends_to_conv = TransformerConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False)

        self.out_layer = nn.Sequential(
            nn.Linear(h_feat, h_feat),
            nn.ReLU(),
            nn.Linear(h_feat, num_output_channels),
        )

    def forward(self, data):
        x_p = self.pmt_encoder(data['pmt'].x)
        x_v = self.virtual_encoder(data['virtual_node'].x)
        # x_m = self.mpmt_encoder(data['mpmt'].x)

        # x_p = self.pmt_encoder_norm(self.pmt_encoder(data['pmt'].x))

        pmt_edges  = data['pmt',  'neighbours', 'pmt'].edge_index
        belongs_to = data['pmt',  'belongs_to', 'mpmt'].edge_index
        mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index
        # reports_to = data['mpmt', 'reports_to', 'virtual_node'].edge_index
        # attends_to = data['virtual_node', 'attends_to', 'mpmt'].edge_index

        for conv in self.pmt_layers:
            x_p = F.dropout(F.relu(conv(x_p, pmt_edges)),p=self.dropout, training=self.training) + x_p

        # x_m = F.dropout(F.relu(self.pool_conv(
        #     (x_p, x_m), belongs_to,
        # )),p=self.dropout, training=self.training)

        x_m = scatter_mean(
            x_p,
            belongs_to[1],
            dim=0,
            dim_size=data['mpmt'].x.size(0),
            )

        # x_m = F.relu(self.pool_proj(torch.cat([x_m, x_m_frompmt], dim=-1)))

        # Broadcast to each mPMT using batch index
        # x_m = x_m + x_v[data['mpmt'].batch]

        x_m = self.mpmt_norm(x_m)          # normalise scattered features
        x_v_broadcast = self.virtual_norm(x_v[data['mpmt'].batch])  # normalise virtual
        x_m = x_m + x_v_broadcast 

        # x_m = x_m + self.attends_to_conv((x_v, x_m), attends_to)

        for conv in self.mpmt_layers:
            x_m = F.dropout(F.relu(conv(x_m, mpmt_edges)),p=self.dropout, training=self.training) + x_m

        # x_v = self.reports_to_conv((x_m, x_v), reports_to)
        # x_m = x_m + self.attends_to_conv((x_v, x_m), attends_to)

        out = global_add_pool(x_m, data['mpmt'].batch)

        return self.out_layer(out)




# 2. Initial unsuccessful attempts at heterogeneous GNNs 
# submission scripts in /vols/hyperk/users/sc4422/first_run/scripts/gnn
# model path in config may need changing since the models were moved here from gnn.py

# class HeteroConvol(nn.Module):
#     def __init__(self,
#         pmt_in, 
#         mpmt_in, 
#         h_feat, 
#         num_output_channels, 
#         dropout,
#         num_heads=4,
#         num_layers=3, 
#         aggr='sum',
#     ):
#         super().__init__()
#         self.dropout = dropout

#         self.convs = torch.nn.ModuleList([HeteroConv({
#                 ('pmt', 'belongs_to', 'mpmt'): GATConv((pmt_in, mpmt_in), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#                 ('mpmt', 'neighbours', 'mpmt'): GATConv(mpmt_in, h_feat, heads=num_heads, concat=False),
#                 ('mpmt', 'contains', 'pmt'): GATConv((mpmt_in, pmt_in), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#                 # ('pmt', 'neighbours', 'pmt'): GATConv(pmt_in, h_feat, heads=num_heads, concat=False),
#             }, aggr=aggr)])

#         for _ in range(num_layers-1):
#             conv = HeteroConv({
#                 ('pmt', 'belongs_to', 'mpmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#                 ('mpmt', 'neighbours', 'mpmt'): GATConv(h_feat, h_feat, heads=num_heads, concat=False),
#                 ('mpmt', 'contains', 'pmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#                 # ('pmt', 'neighbours', 'pmt'): GATConv(h_feat, h_feat, heads=num_heads, concat=False),
#             }, aggr=aggr)
#             self.convs.append(conv)

#         self.out_layer = nn.Linear(h_feat, num_output_channels)

#     def forward(self, data):
#         x_dict = {
#             'pmt':  data['pmt'].x,
#             'mpmt': data['mpmt'].x,
#         }
#         edge_index_dict = {
#             ('pmt',  'belongs_to', 'mpmt'): data['pmt',  'belongs_to', 'mpmt'].edge_index,
#             ('mpmt', 'contains',   'pmt'):  data['mpmt', 'contains',   'pmt'].edge_index,
#             ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
#             # ('pmt',  'neighbours', 'pmt'):  data['pmt',  'neighbours', 'pmt'].edge_index,
#         }

#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training)
#                       for key, x in x_dict.items()}

#         out = global_add_pool(x_dict['mpmt'], data['mpmt'].batch)

#         return self.out_layer(out)


class LocalPMTGAT(nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        assert hidden_channels % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.q = nn.Linear(hidden_channels, hidden_channels)
        self.k = nn.Linear(hidden_channels, hidden_channels)
        self.v = nn.Linear(hidden_channels, hidden_channels)

        self.update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index):
        row, col = edge_index

        Q = self.q(x).view(-1, self.num_heads, self.head_dim)
        K = self.k(x).view(-1, self.num_heads, self.head_dim)
        V = self.v(x).view(-1, self.num_heads, self.head_dim)

        scores = (Q[row] * K[col]).sum(-1) / self.head_dim ** 0.5
        attn = softmax(scores, index=row)

        msg = attn.unsqueeze(-1) * V[col]
        msg = msg.view(-1, x.size(-1))

        agg = torch.zeros_like(x)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        return self.update(torch.cat([x, agg], dim=-1)) + x


class HeteroConvol(nn.Module):
    def __init__(self,
        pmt_in, 
        mpmt_in, 
        h_feat, 
        num_output_channels, 
        dropout,
        num_heads=4,
        num_layers=3, 
        num_local=1,
        aggr='sum',
    ):
        super().__init__()
        self.dropout = dropout

        self.pmt_enc  = nn.Linear(pmt_in,  h_feat)
        self.mpmt_enc = nn.Linear(mpmt_in, h_feat)

        self.local_layers = nn.ModuleList([
            LocalPMTGAT(h_feat, num_heads=num_heads)
            for _ in range(num_local)
        ])

        self.convs = torch.nn.ModuleList([
            HeteroConv({
                ('pmt', 'belongs_to', 'mpmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
                ('mpmt', 'neighbours', 'mpmt'): GATConv(h_feat, h_feat, heads=num_heads, concat=False),
                ('mpmt', 'contains', 'pmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
            }, aggr=aggr)
            for _ in range(num_layers)
        ])

        self.out_layer = nn.Linear(h_feat, num_output_channels)

    def forward(self, data):
        x_p = self.pmt_enc(data['pmt'].x)
        x_m = self.mpmt_enc(data['mpmt'].x)

        pmt_edges = data['pmt', 'neighbours', 'pmt'].edge_index

        for layer in self.local_layers:
            x_p = layer(x_p, pmt_edges)

        x_dict = {
            'pmt':  x_p,
            'mpmt': x_m,
        }
        edge_index_dict = {
            ('pmt',  'belongs_to', 'mpmt'): data['pmt',  'belongs_to', 'mpmt'].edge_index,
            ('mpmt', 'contains',   'pmt'):  data['mpmt', 'contains',   'pmt'].edge_index,
            ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training)
                      for key, x in x_dict.items()}

        out = global_add_pool(x_dict['mpmt'], data['mpmt'].batch)
        return self.out_layer(out)

# class HeteroConvol(nn.Module):
#     def __init__(self,
#         pmt_in, 
#         mpmt_in, 
#         h_feat, 
#         num_output_channels, 
#         dropout,
#         num_heads=4,
#         num_layers=3, 
#         aggr='sum',
#     ):
#         super().__init__()
#         self.dropout = dropout

#         self.convs = torch.nn.ModuleList([HeteroConv({
#                 ('pmt', 'belongs_to', 'mpmt'): GATConv((h_feat, mpmt_in), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#                 ('mpmt', 'neighbours', 'mpmt'): GATConv(mpmt_in, h_feat, heads=num_heads, concat=False),
#                 ('mpmt', 'contains', 'pmt'): GATConv((mpmt_in, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#             }, aggr=aggr)])

#         for _ in range(num_layers-1):
#             conv = HeteroConv({
#                 ('pmt', 'belongs_to', 'mpmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#                 ('mpmt', 'neighbours', 'mpmt'): GATConv(h_feat, h_feat, heads=num_heads, concat=False),
#                 ('mpmt', 'contains', 'pmt'): GATConv((h_feat, h_feat), h_feat, heads=num_heads, concat=False, add_self_loops=False),
#             }, aggr=aggr)
#             self.convs.append(conv)

#         self.out_layer = nn.Linear(h_feat, num_output_channels)

#     def forward(self, data):
#         x_dict = {
#             'pmt':  data['pmt'].x,
#             'mpmt': data['mpmt'].x,
#         }
#         edge_index_dict = {
#             ('pmt',  'belongs_to', 'mpmt'): data['pmt',  'belongs_to', 'mpmt'].edge_index,
#             ('mpmt', 'contains',   'pmt'):  data['mpmt', 'contains',   'pmt'].edge_index,
#             ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
#         }

#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: F.dropout(F.relu(x), self.dropout, training=self.training)
#                       for key, x in x_dict.items()}

#         out = global_add_pool(x_dict['mpmt'], data['mpmt'].batch)
#         return self.out_layer(out)



### without pmt pmt
# class HierarchicalHetGAT(nn.Module):
#     def __init__(self, pmt_in, mpmt_in, h_feat, num_output_channels,
#                 num_global=3, heads=4, dropout=0.0):
#         super().__init__()
#         self.pmt_enc  = nn.Linear(pmt_in,  h_feat)
#         self.mpmt_enc = nn.Linear(mpmt_in, h_feat)

#         self.pool_conv = GATConv((h_feat, h_feat), h_feat, heads=heads,
#                                 concat=False, add_self_loops=False)

#         self.global_layers = nn.ModuleList([
#             GATConv(h_feat, h_feat, heads=heads, concat=False)
#             for _ in range(num_global)
#         ])

#         self.drop = nn.Dropout(dropout)
#         self.out_layer = nn.Linear(h_feat, num_output_channels)

#     def forward(self, data):
#         x_p = self.pmt_enc(data['pmt'].x)
#         x_m = self.mpmt_enc(data['mpmt'].x)

#         belongs_to = data['pmt', 'belongs_to', 'mpmt'].edge_index
#         mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index

#         # straight to attention pooling
#         x_m = x_m + F.relu(self.pool_conv(
#             (x_p, x_m), belongs_to,
#             size=(x_p.size(0), x_m.size(0)),
#         ))

#         for conv in self.global_layers:
#             x_m = self.drop(x_m + F.relu(conv(x_m, mpmt_edges)))

#         out = global_add_pool(x_m, data['mpmt'].batch)
#         return self.out_layer(out)

# with pmt pmt - using the custom

class HierarchicalHetGAT(nn.Module):
    def __init__(self, pmt_in, mpmt_in, h_feat, num_output_channels,
                 num_local=1, num_global=3, heads=4, dropout=0.0):
        super().__init__()
        self.pmt_enc  = nn.Linear(pmt_in,  h_feat)
        self.mpmt_enc = nn.Linear(mpmt_in, h_feat)

        self.local_layers = nn.ModuleList([
            LocalPMTGAT(h_feat, num_heads=heads)
            for _ in range(num_local)
        ])

        self.pool_conv = GATConv((h_feat, h_feat), h_feat, heads=heads,
                                 concat=False, add_self_loops=False)

        self.global_layers = nn.ModuleList([
            GATConv(h_feat, h_feat, heads=heads, concat=False)
            for _ in range(num_global)
        ])

        self.drop = nn.Dropout(dropout)
        self.out_layer = nn.Linear(h_feat, num_output_channels)

    def forward(self, data):
        x_p = self.pmt_enc(data['pmt'].x)
        x_m = self.mpmt_enc(data['mpmt'].x)

        pmt_edges  = data['pmt', 'neighbours', 'pmt'].edge_index
        belongs_to = data['pmt', 'belongs_to', 'mpmt'].edge_index
        mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index

        for layer in self.local_layers:
            x_p = layer(x_p, pmt_edges)

        x_m = x_m + F.relu(self.pool_conv(
            (x_p, x_m), belongs_to,
            size=(x_p.size(0), x_m.size(0)),
        ))

        for conv in self.global_layers:
            x_m = self.drop(x_m + F.relu(conv(x_m, mpmt_edges)))

        out = global_add_pool(x_m, data['mpmt'].batch)
        return self.out_layer(out)

# class HierarchicalHetGAT(nn.Module):
#     def __init__(self, pmt_in, mpmt_in, h_feat, num_output_channels,
#                  num_local=1, num_global=3, heads=4, dropout=0.0):
#         super().__init__()
#         self.pmt_enc  = nn.Linear(pmt_in,  h_feat)
#         self.mpmt_enc = nn.Linear(mpmt_in, h_feat)

#         self.local_layers = nn.ModuleList([
#             GATConv(h_feat, h_feat, heads=heads, concat=False)
#             for _ in range(num_local)
#         ])

#         # learned attention pooling pmt -> mpmt (replaces their scatter_mean)
#         self.pool_conv = GATConv((h_feat, h_feat), h_feat, heads=heads,
#                                  concat=False, add_self_loops=False)

#         self.global_layers = nn.ModuleList([
#             GATConv(h_feat, h_feat, heads=heads, concat=False)
#             for _ in range(num_global)
#         ])

#         self.drop = nn.Dropout(dropout)
#         self.out_layer = nn.Linear(h_feat, num_output_channels)

#     def forward(self, data):
#         x_p = self.pmt_enc(data['pmt'].x)
#         x_m = self.mpmt_enc(data['mpmt'].x)

#         pmt_edges  = data['pmt',  'neighbours', 'pmt'].edge_index
#         belongs_to = data['pmt',  'belongs_to', 'mpmt'].edge_index
#         mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index

#         # stage 1: local intra-mPMT
#         for conv in self.local_layers:
#             x_p = x_p + F.relu(conv(x_p, pmt_edges))

#         # stage 2: attention-pooled hand-off (mpmt features are the queries)
#         x_m = x_m + F.relu(self.pool_conv(
#             (x_p, x_m), belongs_to,
#             size=(x_p.size(0), x_m.size(0)),
#         ))

#         # stage 3: global inter-mPMT
#         for conv in self.global_layers:
#             x_m = self.drop(x_m + F.relu(conv(x_m, mpmt_edges)))

#         out = global_add_pool(x_m, data['mpmt'].batch)
#         return self.out_layer(out)

# class HierarchicalHetGAT(nn.Module):
#     def __init__(self, pmt_in, mpmt_in, h_feat, num_output_channels,
#                  num_local=1, num_global=3, heads=4, dropout=0.0):
#         super().__init__()
#         self.pmt_enc  = nn.Linear(pmt_in,  h_feat)
#         self.mpmt_enc = nn.Linear(mpmt_in, h_feat)

#         self.local_layers = nn.ModuleList([
#             GATConv(h_feat, h_feat, heads=heads, concat=False)
#             for _ in range(num_local)
#         ])

#         # learned attention pooling pmt -> mpmt (replaces their scatter_mean)
#         self.pool_conv = GATConv((h_feat, h_feat), h_feat, heads=heads,
#                                  concat=False, add_self_loops=False)

#         self.global_layers = nn.ModuleList([
#             GATConv(h_feat, h_feat, heads=heads, concat=False)
#             for _ in range(num_global)
#         ])

#         self.drop = nn.Dropout(dropout)
#         self.out_layer = nn.Linear(h_feat, num_output_channels)

#     def forward(self, data):
#         x_p = self.pmt_enc(data['pmt'].x)
#         x_m = self.mpmt_enc(data['mpmt'].x)

#         pmt_edges  = data['pmt',  'neighbours', 'pmt'].edge_index
#         belongs_to = data['pmt',  'belongs_to', 'mpmt'].edge_index
#         mpmt_edges = data['mpmt', 'neighbours', 'mpmt'].edge_index

#         # stage 1: local intra-mPMT
#         for conv in self.local_layers:
#             x_p = x_p + F.relu(conv(x_p, pmt_edges))

#         # stage 2: attention-pooled hand-off (mpmt features are the queries)
#         x_m = x_m + F.relu(self.pool_conv(
#             (x_p, x_m), belongs_to,
#             size=(x_p.size(0), x_m.size(0)),
#         ))

#         # stage 3: global inter-mPMT
#         for conv in self.global_layers:
#             x_m = self.drop(x_m + F.relu(conv(x_m, mpmt_edges)))

#         out = global_add_pool(x_m, data['mpmt'].batch)
#         return self.out_layer(out)
        

import sys
sys.modules['pyg_lib'] = None  # must be before any torch_geometric imports

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HGTConv

class HGT(torch.nn.Module):
    def __init__(self, pmt_in=6, mpmt_in=6, h_feat=64, heads=4, num_output_channels=7): #,pmt_emb_dim = 4):
        super().__init__()

        node_types = ['pmt', 'mpmt']
        edge_types = [
            ('pmt', 'belongs_to', 'mpmt'),
            ('mpmt', 'has_pmt', 'pmt'),
            ('mpmt', 'neighbours', 'mpmt'),
            ('pmt', 'neighbours', 'pmt'), 
        ]
        metadata = (node_types, edge_types)
        # self.pmt_id_emb = torch.nn.Embedding(19, pmt_emb_dim)

        self.conv1 = HGTConv(
            in_channels={'pmt': pmt_in, 'mpmt': mpmt_in},
            # in_channels={'pmt': pmt_in + pmt_emb_dim, 'mpmt': mpmt_in},
            out_channels=h_feat, metadata=metadata, heads=heads
        )
        self.conv2 = HGTConv(
            in_channels={'pmt': h_feat, 'mpmt': h_feat},
            out_channels=h_feat, metadata=metadata, heads=heads
        )
        self.conv3 = HGTConv(
            in_channels={'pmt': h_feat, 'mpmt': h_feat},
            out_channels=h_feat, metadata=metadata, heads=heads
        )
        # self.conv4 = HGTConv(
        #     in_channels={'pmt': h_feat, 'mpmt': h_feat},
        #     out_channels=h_feat, metadata=metadata, heads=heads
        # )

        # BatchNorm for each conv layer, separate for each node type
        self.bn1_pmt  = torch.nn.BatchNorm1d(h_feat)
        self.bn1_mpmt = torch.nn.BatchNorm1d(h_feat)
        self.bn2_pmt  = torch.nn.BatchNorm1d(h_feat)
        self.bn2_mpmt = torch.nn.BatchNorm1d(h_feat)
        self.bn3_pmt  = torch.nn.BatchNorm1d(h_feat)
        self.bn3_mpmt = torch.nn.BatchNorm1d(h_feat)
        self.bn4_mpmt = torch.nn.BatchNorm1d(h_feat)  # conv4: only mpmt feeds into pool
        self.bn4_pmt = torch.nn.BatchNorm1d(h_feat)
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(h_feat, h_feat // 2),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.Linear(h_feat // 2, num_output_channels)
        # )
        self.mlp = torch.nn.Sequential(
           torch.nn.Linear(4*h_feat, h_feat),
           torch.nn.ReLU(),
           torch.nn.Dropout(0.3),
           torch.nn.Linear(h_feat, h_feat // 2),
           torch.nn.ReLU(),
           torch.nn.Linear(h_feat // 2, num_output_channels)
        )
    def forward(self, data):
        x_dict = {
            'pmt':  data['pmt'].x,
            'mpmt': data['mpmt'].x
        }
        # pmt_id_emb = self.pmt_id_emb(data['pmt'].pmt_id.long())

        # x_dict = {
        #     'pmt':  torch.cat([data['pmt'].x, pmt_id_emb], dim=1),
        #     'mpmt': data['mpmt'].x
        # }
        edge_index_dict = {
            ('pmt',  'belongs_to', 'mpmt'): data['pmt', 'belongs_to', 'mpmt'].edge_index,
            ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
            ('mpmt', 'has_pmt',    'pmt'):  data['pmt', 'belongs_to', 'mpmt'].edge_index.flip(0),
            ('pmt',  'neighbours', 'pmt'):  data['pmt', 'neighbours', 'pmt'].edge_index, 
        }

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['pmt']  = F.relu(self.bn1_pmt(x_dict['pmt']))
        x_dict['mpmt'] = F.relu(self.bn1_mpmt(x_dict['mpmt']))

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict['pmt']  = F.relu(self.bn2_pmt(x_dict['pmt']))
        x_dict['mpmt'] = F.relu(self.bn2_mpmt(x_dict['mpmt']))

        # old = {k: v for k, v in x_dict.items()}
        # new = self.conv2(x_dict, edge_index_dict)

        # x_dict = {
        #     'pmt':  F.relu(self.bn2_pmt(new['pmt'] + old['pmt'])),
        #     'mpmt': F.relu(self.bn2_mpmt(new['mpmt'] + old['mpmt'])),
        # }

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict['pmt']  = F.relu(self.bn3_pmt(x_dict['pmt']))
        x_dict['mpmt'] = F.relu(self.bn3_mpmt(x_dict['mpmt']))

        # old = {k: v for k, v in x_dict.items()}
        # new = self.conv3(x_dict, edge_index_dict)

        # x_dict = {
        #     'pmt':  F.relu(self.bn3_pmt(new['pmt'] + old['pmt'])),
        #     'mpmt': F.relu(self.bn3_mpmt(new['mpmt'] + old['mpmt'])),
        # }

        # x_dict = self.conv4(x_dict, edge_index_dict)
        # x_dict['mpmt'] = F.relu(self.bn4_mpmt(x_dict['mpmt']))
        # x_dict['pmt']  = F.relu(self.bn4_pmt(x_dict['pmt']))

        #pmt not normalised after conv4 since it doesn't feed into the pool

        # old = {k: v for k, v in x_dict.items()}
        # new = self.conv4(x_dict, edge_index_dict)

        # x_dict = {
        #     'pmt': new['pmt'],  # not used in current pooling
        #     'mpmt': F.relu(self.bn4_mpmt(new['mpmt'] + old['mpmt'])),
        # }

        # x = pyg_nn.global_mean_pool(x_dict['mpmt'], data['mpmt'].batch)

        pmt_pool = torch.cat([
            pyg_nn.global_mean_pool(x_dict['pmt'], data['pmt'].batch),
            pyg_nn.global_add_pool(x_dict['pmt'], data['pmt'].batch),
        ], dim=1)

        mpmt_pool = torch.cat([
            pyg_nn.global_mean_pool(x_dict['mpmt'], data['mpmt'].batch),
            pyg_nn.global_add_pool(x_dict['mpmt'], data['mpmt'].batch),
        ], dim=1)

        x = torch.cat([pmt_pool, mpmt_pool], dim=1)

        # pmt_pool = pyg_nn.global_mean_pool(x_dict['pmt'], data['pmt'].batch)
        # mpmt_pool = pyg_nn.global_mean_pool(x_dict['mpmt'], data['mpmt'].batch)
        # x = torch.cat([pmt_pool,mpmt_pool],dim=1)

        return self.mlp(x)

# class HGT(torch.nn.Module):
#     def __init__(self, pmt_in=6, mpmt_in=6, h_feat=64, heads=4, num_output_channels=7):
#         super().__init__()

#         node_types = ['pmt', 'mpmt']
#         edge_types = [
#             ('pmt', 'belongs_to', 'mpmt'),
#             ('mpmt', 'has_pmt', 'pmt'),
#             ('mpmt', 'neighbours', 'mpmt'),
#         ]
#         metadata = (node_types, edge_types)

#         # self.res_proj_pmt = torch.nn.Linear(pmt_in, h_feat)
#         # self.res_proj_mpmt = torch.nn.Linear(mpmt_in, h_feat)

#         self.conv1 = HGTConv(
#             in_channels={'pmt': pmt_in, 'mpmt': mpmt_in},
#             out_channels=h_feat, metadata=metadata, heads=heads
#         )
#         self.conv2 = HGTConv(
#             in_channels={'pmt': h_feat, 'mpmt': h_feat},
#             out_channels=h_feat, metadata=metadata, heads=heads
#         )
#         self.conv3 = HGTConv(
#             in_channels={'pmt': h_feat, 'mpmt': h_feat},
#             out_channels=h_feat, metadata=metadata, heads=heads
#         )
#         self.conv4 = HGTConv(
#             in_channels={'pmt': h_feat, 'mpmt': h_feat},
#             out_channels=h_feat, metadata=metadata, heads=heads
#         )

#         # BatchNorm for each conv layer, separate for each node type
#         self.bn1_pmt  = torch.nn.BatchNorm1d(h_feat)
#         self.bn1_mpmt = torch.nn.BatchNorm1d(h_feat)
#         self.bn2_pmt  = torch.nn.BatchNorm1d(h_feat)
#         self.bn2_mpmt = torch.nn.BatchNorm1d(h_feat)
#         self.bn3_pmt  = torch.nn.BatchNorm1d(h_feat)
#         self.bn3_mpmt = torch.nn.BatchNorm1d(h_feat)
#         self.bn4_mpmt = torch.nn.BatchNorm1d(h_feat)  # conv4: only mpmt feeds into pool

#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(h_feat, h_feat // 2),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
#             torch.nn.Linear(h_feat // 2, num_output_channels)
#         )

#         # self.mlp = torch.nn.Sequential(
#         #     torch.nn.Linear(h_feat, h_feat),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Dropout(0.2),
#         #     torch.nn.Linear(h_feat, h_feat // 2),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Linear(h_feat // 2, num_output_channels)
#         # )

#     def forward(self, data):
#         x_dict = {
#             'pmt':  data['pmt'].x,
#             'mpmt': data['mpmt'].x
#         }
#         edge_index_dict = {
#             ('pmt',  'belongs_to', 'mpmt'): data['pmt', 'belongs_to', 'mpmt'].edge_index,
#             ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
#             ('mpmt', 'has_pmt',    'pmt'):  data['pmt', 'belongs_to', 'mpmt'].edge_index.flip(0),
#         }

#         x_dict = self.conv1(x_dict, edge_index_dict)
#         x_dict['pmt']  = F.relu(self.bn1_pmt(x_dict['pmt']))
#         x_dict['mpmt'] = F.relu(self.bn1_mpmt(x_dict['mpmt']))

#         # res_dict = {
#         #     'pmt': self.res_proj_pmt(x_dict['pmt']),
#         #     'mpmt': self.res_proj_mpmt(x_dict['mpmt'])
#         # }

#         # out_dict = self.conv1(x_dict, edge_index_dict)

#         # x_dict = {
#         #     'pmt':  F.relu(self.bn1_pmt(out_dict['pmt'] + res_dict['pmt'])),
#         #     'mpmt': F.relu(self.bn1_mpmt(out_dict['mpmt'] + res_dict['mpmt']))
#         # }

#         x_dict = self.conv2(x_dict, edge_index_dict)
#         x_dict['pmt']  = F.relu(self.bn2_pmt(x_dict['pmt']))
#         x_dict['mpmt'] = F.relu(self.bn2_mpmt(x_dict['mpmt']))

#         # res_dict = x_dict
#         # out_dict = self.conv2(x_dict, edge_index_dict)

#         # x_dict = {
#         #     'pmt':  F.relu(self.bn2_pmt(out_dict['pmt'] + res_dict['pmt'])),
#         #     'mpmt': F.relu(self.bn2_mpmt(out_dict['mpmt'] + res_dict['mpmt']))
#         # }

#         x_dict = self.conv3(x_dict, edge_index_dict)
#         x_dict['pmt']  = F.relu(self.bn3_pmt(x_dict['pmt']))
#         x_dict['mpmt'] = F.relu(self.bn3_mpmt(x_dict['mpmt']))

#         # res_dict = x_dict
#         # out_dict = self.conv3(x_dict, edge_index_dict)

#         # x_dict = {
#         #     'pmt':  F.relu(self.bn3_pmt(out_dict['pmt'] + res_dict['pmt'])),
#         #     'mpmt': F.relu(self.bn3_mpmt(out_dict['mpmt'] + res_dict['mpmt']))
#         # }

#         x_dict = self.conv4(x_dict, edge_index_dict)
#         x_dict['mpmt'] = F.relu(self.bn4_mpmt(x_dict['mpmt']))
#         # pmt not normalised after conv4 since it doesn't feed into the pool

#         # res_dict = x_dict
#         # out_dict = self.conv4(x_dict, edge_index_dict)

#         # x_dict = {
#         #     'pmt': res_dict['pmt'],  # not needed after this, so just keep it unchanged
#         #     'mpmt': F.relu(self.bn4_mpmt(out_dict['mpmt'] + res_dict['mpmt']))
#         # }

#         x = pyg_nn.global_add_pool(x_dict['mpmt'], data['mpmt'].batch)

#         return self.mlp(x)

class HGTMPMTOnly(torch.nn.Module):
    def __init__(self, mpmt_in=41, h_feat=64, heads=4, num_output_channels=7):
        super().__init__()

        node_types = ["mpmt"]
        edge_types = [
            ("mpmt", "neighbours", "mpmt"),
        ]
        metadata = (node_types, edge_types)

        self.conv1 = HGTConv(
            in_channels={"mpmt": mpmt_in},
            out_channels=h_feat,
            metadata=metadata,
            heads=heads,
        )
        self.conv2 = HGTConv(
            in_channels={"mpmt": h_feat},
            out_channels=h_feat,
            metadata=metadata,
            heads=heads,
        )
        self.conv3 = HGTConv(
            in_channels={"mpmt": h_feat},
            out_channels=h_feat,
            metadata=metadata,
            heads=heads,
        )
        self.conv4 = HGTConv(
            in_channels={"mpmt": h_feat},
            out_channels=h_feat,
            metadata=metadata,
            heads=heads,
        )

        self.bn1 = torch.nn.BatchNorm1d(h_feat)
        self.bn2 = torch.nn.BatchNorm1d(h_feat)
        self.bn3 = torch.nn.BatchNorm1d(h_feat)
        self.bn4 = torch.nn.BatchNorm1d(h_feat)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * h_feat, h_feat),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(h_feat, num_output_channels),
        )

    def forward(self, data):
        x_dict = {
            "mpmt": data["mpmt"].x,
        }

        edge_index_dict = {
            ("mpmt", "neighbours", "mpmt"): data["mpmt", "neighbours", "mpmt"].edge_index,
        }

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict["mpmt"] = F.relu(self.bn1(x_dict["mpmt"]))

        old = x_dict["mpmt"]
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict["mpmt"] = F.relu(self.bn2(x_dict["mpmt"] + old))

        old = x_dict["mpmt"]
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict["mpmt"] = F.relu(self.bn3(x_dict["mpmt"] + old))

        old = x_dict["mpmt"]
        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict["mpmt"] = F.relu(self.bn4(x_dict["mpmt"] + old))

        x = torch.cat([
            pyg_nn.global_mean_pool(x_dict["mpmt"], data["mpmt"].batch),
            pyg_nn.global_add_pool(x_dict["mpmt"], data["mpmt"].batch),
            pyg_nn.global_max_pool(x_dict["mpmt"], data["mpmt"].batch),
        ], dim=1)

        return self.mlp(x)

