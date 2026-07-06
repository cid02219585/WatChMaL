# can add things like layer norm in after

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


























## han
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import GATv2Conv, global_add_pool
 
 
# ---------------------------------------------------------------------------
# Metapath construction (called per event in the dataset)
# ---------------------------------------------------------------------------
 
def build_same_mpmt_edges(belongs_to_edge_index, num_mpmt):
    """Hit-sibling cliques: directed edges between all hit-PMT pairs sharing
    an mPMT.
 
    belongs_to_edge_index: (2, E) tensor, row 0 = pmt indices, row 1 = mpmt indices.
    Returns edge_index of shape (2, sum_m c_m * (c_m - 1)); may be (2, 0) if
    no mPMT has two or more hit PMTs.
    """
    pmt, mpmt = belongs_to_edge_index
    order = mpmt.argsort()
    pmt, mpmt = pmt[order], mpmt[order]
    counts = torch.bincount(mpmt, minlength=num_mpmt)
 
    src, dst = [], []
    ptr = 0
    for c in counts.tolist():
        if c > 1:
            grp = pmt[ptr:ptr + c]
            a = grp.repeat_interleave(c)
            b = grp.repeat(c)
            mask = a != b                       # drop self-loops
            src.append(a[mask])
            dst.append(b[mask])
        ptr += c
 
    if not src:                                 # sparse event: no siblings
        return torch.empty((2, 0), dtype=torch.long)
 
    edge_index = torch.stack([torch.cat(src), torch.cat(dst)])
 
    expected = int((counts * (counts - 1)).sum())
    assert edge_index.size(1) == expected, "edge count mismatch"
    return edge_index
 
 
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
 
class SemanticAttention(nn.Module):
    """Learned metapath weights: beta = softmax_phi( q . tanh(W z_phi + b) ),
    with the score averaged over nodes (graph-level beta, as in the HAN paper).
    """
 
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.last_beta = None   # stashed for logging / physics readout
 
    def forward(self, z):                    # z: (P, N, dim)
        w = self.project(z).mean(dim=1)      # (P, 1)
        beta = torch.softmax(w, dim=0)       # (P, 1)
        self.last_beta = beta.detach()
        return (beta.unsqueeze(-1) * z).sum(dim=0)   # (N, dim)
 
 
class HANModel(nn.Module):
    """HAN over hit-PMT nodes with two parallel views:
       geometric kNN neighbours and hit siblings within the same mPMT.
 
    Node-level attention: GATv2 per view (dynamic additive attention).
    Semantic-level attention: learned beta per layer mixing the views.
    Note: mpmt features are deliberately unused -- the metapath collapse
    discards intermediate node content. That is part of the HAN-vs-HeteroConv
    comparison, not an oversight.
    """
 
    NEIGHBOURS = ('pmt', 'neighbours', 'pmt')
    SAME_MPMT = ('pmt', 'same_mpmt', 'pmt')   # built per event in the dataset
 
    def __init__(
        self,
        pmt_in,
        h_feat,
        num_output_channels,
        dropout=0.0,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.dropout = dropout
        self.metapaths = [self.NEIGHBOURS, self.SAME_MPMT]
 
        self.layers = nn.ModuleList()
        self.semantics = nn.ModuleList()
        in_dim = pmt_in
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                '__'.join(mp): GATv2Conv(
                    in_dim, h_feat,
                    heads=num_heads, concat=False, add_self_loops=False,
                )
                for mp in self.metapaths
            }))
            self.semantics.append(SemanticAttention(h_feat))
            in_dim = h_feat
 
        self.out_layer = nn.Sequential(
            nn.Linear(h_feat, h_feat),
            nn.ReLU(),
            nn.LayerNorm(h_feat),
            nn.Linear(h_feat, num_output_channels),
        )
 
    def forward(self, data):
        x = data['pmt'].x
        for layer, semantic in zip(self.layers, self.semantics):
            z = torch.stack([
                layer['__'.join(mp)](x, data[mp].edge_index)
                for mp in self.metapaths
            ])                                   # (P, N_pmt, h_feat)
            x = semantic(z)                      # (N_pmt, h_feat)
            x = F.dropout(F.relu(x), self.dropout, training=self.training)
 
        out = global_add_pool(x, data['pmt'].batch)
        return self.out_layer(out)
 
    @property
    def betas(self):
        """Per-layer metapath weights, shape (num_layers, num_metapaths),
        ordered as self.metapaths: (neighbours, same_mpmt)."""
        return torch.stack([
            s.last_beta.squeeze(-1) for s in self.semantics
            if s.last_beta is not None
        ])

## this is heteroconv not HAN!!!

from torch_geometric.nn import GATConv, HeteroConv

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

### baseline

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.net(x)

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

class InterMPMTGAT(nn.Module):
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

class HierarchicalGATBaseline(nn.Module):
    def __init__(
        self,
        pmt_in_channels=8,
        hidden_channels=64,
        num_output_channels=7,
        num_local_layers=1,
        num_inter_layers=3,
        num_heads_local=1,
        num_heads_global=4,
        dropout=0.0,
    ):
        super().__init__()

        self.pmt_encoder = NodeEncoder(pmt_in_channels, hidden_channels)

        self.local_pmt_layers = nn.ModuleList([
            LocalPMTGAT(hidden_channels, num_heads_local)
            for _ in range(num_local_layers)
        ])

        self.inter_layers = nn.ModuleList([
            InterMPMTGAT(hidden_channels, num_heads_global)
            for _ in range(num_inter_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, num_output_channels),
        )

        self.drop = nn.Dropout(p=dropout)

    def forward(self, data):
        # --- unpack HeteroData
        x            = data['pmt'].x
        batch_mpmt   = data['mpmt'].batch
        pmt_edges    = data['pmt',  'neighbours', 'pmt'].edge_index
        mpmt_edges   = data['mpmt', 'neighbours', 'mpmt'].edge_index
        belongs_to   = data['pmt',  'belongs_to', 'mpmt'].edge_index
        # belongs_to[0] = pmt indices, belongs_to[1] = mpmt indices
        num_mpmts    = data['mpmt'].x.size(0)

        # --- encode PMTs
        x = self.pmt_encoder(x)

        # --- local intra-mPMT PMT-PMT message passing
        for layer in self.local_pmt_layers:
            x = layer(x, pmt_edges)

        # --- aggregate PMT representations to mPMT level
        x_mpmt = scatter_mean(
            x,
            belongs_to[1],
            dim=0,
            dim_size=num_mpmts,
        )

        # --- inter-mPMT message passing
        for layer in self.inter_layers:
            x_mpmt = layer(x_mpmt, mpmt_edges)
            x_mpmt = self.drop(x_mpmt)

        # --- pool to event
        x_evt = global_mean_pool(x_mpmt, batch_mpmt)

        return self.classifier(x_evt)



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

        # stage 1: local intra-mPMT (memory-efficient custom GAT)
        for layer in self.local_layers:
            x_p = layer(x_p, pmt_edges)

        # stage 2: attention pooling PMT -> mPMT
        x_m = x_m + F.relu(self.pool_conv(
            (x_p, x_m), belongs_to,
            size=(x_p.size(0), x_m.size(0)),
        ))

        # stage 3: global inter-mPMT
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
        
# import sys, types
# # Force pyg_lib to appear absent so HGTConv uses pure PyTorch fallback
# sys.modules['pyg_lib'] = None
# sys.modules['pyg_lib.ops'] = None
# import pyg_lib
# pyg_lib.ops.segment_matmul = None

# # torch imports
# import torch
# import torch.nn.functional as F
# import torch_geometric.nn.dense.linear as pyg_linear
# pyg_linear.pyg_lib = None

# # pyg imports
# import torch_geometric
# import torch_geometric.nn as pyg_nn
# from torch_geometric.nn import HGTConv

# import sys
# sys.modules['pyg_lib'] = None

# import torch
# import torch.nn.functional as F
# import torch_geometric
# import torch_geometric.nn as pyg_nn
# from torch_geometric.nn import HGTConv

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

#         # self.conv1 = HGTConv(in_channels={'pmt': pmt_in, 'mpmt': mpmt_in}, out_channels=h_feat, metadata=metadata, heads=heads)
#         # self.conv2 = HGTConv(in_channels={'pmt': h_feat, 'mpmt': h_feat}, out_channels=h_feat, metadata=metadata, heads=heads)
#         # self.conv3 = HGTConv(in_channels={'pmt': h_feat, 'mpmt': h_feat}, out_channels=h_feat, metadata=metadata, heads=heads)
#         # self.conv4 = HGTConv(in_channels={'pmt': h_feat, 'mpmt': h_feat}, out_channels=h_feat, metadata=metadata, heads=heads)

#         self.conv1 = HGTConv({'pmt': pmt_in, 'mpmt': mpmt_in}, 32,  metadata, heads=2)
#         self.conv2 = HGTConv({'pmt': 32,     'mpmt': 32},      64,  metadata, heads=4)
#         self.conv3 = HGTConv({'pmt': 64,     'mpmt': 64},      64,  metadata, heads=4)
#         self.conv4 = HGTConv({'pmt': 64,     'mpmt': 64},      128, metadata, heads=4)

#         # Separate BN for pmt and mpmt at each layer
#         self.bn1_pmt  = torch.nn.BatchNorm1d(h_feat)
#         self.bn1_mpmt = torch.nn.BatchNorm1d(h_feat)
#         self.bn2_pmt  = torch.nn.BatchNorm1d(h_feat)
#         self.bn2_mpmt = torch.nn.BatchNorm1d(h_feat)
#         self.bn3_pmt  = torch.nn.BatchNorm1d(h_feat)
#         self.bn3_mpmt = torch.nn.BatchNorm1d(h_feat)
#         self.bn4_pmt  = torch.nn.BatchNorm1d(h_feat)
#         self.bn4_mpmt = torch.nn.BatchNorm1d(h_feat)

#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(h_feat * 2, h_feat),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
#             torch.nn.Linear(h_feat, num_output_channels)
#         )

#     def forward(self, data):
#         x_dict = {
#             'pmt': data['pmt'].x,
#             'mpmt': data['mpmt'].x
#         }
#         edge_index_dict = {
#             ('pmt', 'belongs_to', 'mpmt'): data['pmt', 'belongs_to', 'mpmt'].edge_index,
#             ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
#             ('mpmt', 'has_pmt', 'pmt'): data['pmt', 'belongs_to', 'mpmt'].edge_index.flip(0),
#         }

#         x_dict = self.conv1(x_dict, edge_index_dict)
#         x_dict['pmt']  = F.relu(self.bn1_pmt(x_dict['pmt']))
#         x_dict['mpmt'] = F.relu(self.bn1_mpmt(x_dict['mpmt']))

#         x_dict = self.conv2(x_dict, edge_index_dict)
#         x_dict['pmt']  = F.relu(self.bn2_pmt(x_dict['pmt']))
#         x_dict['mpmt'] = F.relu(self.bn2_mpmt(x_dict['mpmt']))

#         x_dict = self.conv3(x_dict, edge_index_dict)
#         x_dict['pmt']  = F.relu(self.bn3_pmt(x_dict['pmt']))
#         x_dict['mpmt'] = F.relu(self.bn3_mpmt(x_dict['mpmt']))

#         x_dict = self.conv4(x_dict, edge_index_dict)
#         x_dict['pmt']  = F.relu(self.bn4_pmt(x_dict['pmt']))
#         x_dict['mpmt'] = F.relu(self.bn4_mpmt(x_dict['mpmt']))

#         x_mpmt = pyg_nn.global_mean_pool(x_dict['mpmt'], data['mpmt'].batch)
#         x_pmt  = pyg_nn.global_mean_pool(x_dict['pmt'],  data['pmt'].batch)
        
#         x = torch.cat([x_mpmt, x_pmt], dim=-1)

#         return self.mlp(x)
# import sys
# sys.modules['pyg_lib'] = None

# import torch
# import torch.nn.functional as F
# import torch_geometric
# import torch_geometric.nn as pyg_nn
# from torch_geometric.nn import HGTConv

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

#         self.conv1 = HGTConv(in_channels={'pmt': pmt_in, 'mpmt': mpmt_in}, out_channels=h_feat, metadata=metadata, heads=heads)
#         self.conv2 = HGTConv(in_channels={'pmt': h_feat, 'mpmt': h_feat}, out_channels=h_feat, metadata=metadata, heads=heads)
#         self.conv3 = HGTConv(in_channels={'pmt': h_feat, 'mpmt': h_feat}, out_channels=h_feat, metadata=metadata, heads=heads)
#         self.conv4 = HGTConv(in_channels={'pmt': h_feat, 'mpmt': h_feat}, out_channels=h_feat, metadata=metadata, heads=heads)

#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(h_feat * 2, h_feat),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
#             torch.nn.Linear(h_feat, num_output_channels)
#         )

#     def forward(self, data):
#         x_dict = {
#             'pmt': data['pmt'].x,
#             'mpmt': data['mpmt'].x
#         }
#         edge_index_dict = {
#             ('pmt', 'belongs_to', 'mpmt'): data['pmt', 'belongs_to', 'mpmt'].edge_index,
#             ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
#             ('mpmt', 'has_pmt', 'pmt'): data['pmt', 'belongs_to', 'mpmt'].edge_index.flip(0),
#         }

#         x_dict = self.conv1(x_dict, edge_index_dict)
#         x_dict['mpmt'] = F.relu(x_dict['mpmt'])
#         x_dict['pmt'] = F.relu(x_dict['pmt'])

#         x_dict = self.conv2(x_dict, edge_index_dict)
#         x_dict['mpmt'] = F.relu(x_dict['mpmt'])
#         x_dict['pmt'] = F.relu(x_dict['pmt'])

#         x_dict = self.conv3(x_dict, edge_index_dict)
#         x_dict['mpmt'] = F.relu(x_dict['mpmt'])
#         x_dict['pmt'] = F.relu(x_dict['pmt'])

#         x_dict = self.conv4(x_dict, edge_index_dict)
#         x_dict['mpmt'] = F.relu(x_dict['mpmt'])
#         x_dict['pmt'] = F.relu(x_dict['pmt'])

#         # x_mpmt = pyg_nn.global_mean_pool(x_dict['mpmt'], data['mpmt'].batch)
#         # x_pmt  = pyg_nn.global_mean_pool(x_dict['pmt'],  data['pmt'].batch)

#         x_mpmt = pyg_nn.global_add_pool(x_dict['mpmt'], data['mpmt'].batch)
#         x_pmt  = pyg_nn.global_add_pool(x_dict['pmt'],  data['pmt'].batch)
        
#         x = torch.cat([x_mpmt, x_pmt], dim=-1)

#         return self.mlp(x)
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

class GATv2Edge(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=8, num_output_channels=4, heads=4):
        '''
        GATv2 with edge features (delta position + distance).
        edge_dim=4 matches (dx, dy, dz, dist) computed in the dataset.
        '''
        super(GATv2Edge, self).__init__()

        self.conv1 = torch_geometric.nn.GATv2Conv(in_feat, h_feat, heads=heads, edge_dim=4)
        self.conv2 = torch_geometric.nn.GATv2Conv(h_feat * heads, h_feat * 2, heads=heads, edge_dim=4)
        self.conv3 = torch_geometric.nn.GATv2Conv(h_feat * 2 * heads, h_feat * 4, heads=heads, edge_dim=4)
        self.conv4 = torch_geometric.nn.GATv2Conv(h_feat * 4 * heads, h_feat * 8, heads=1, concat=False, edge_dim=4)

        # self.bn1 = torch.nn.BatchNorm1d(h_feat * heads)
        # self.bn2 = torch.nn.BatchNorm1d(h_feat * 2 * heads)
        # self.bn3 = torch.nn.BatchNorm1d(h_feat * 4 * heads)
        # self.bn4 = torch.nn.BatchNorm1d(h_feat * 8)

        self.lin = torch.nn.Linear(h_feat * 8, num_output_channels)
        # self.softplus = torch.nn.Softplus()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        # x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr)
        # x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index, edge_attr)
        # x = self.bn4(x)

        x = torch_geometric.nn.global_add_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # x = torch.cat([
        #     x[:, :6],
        #     self.softplus(x[:, 6:7]) + 0.511
        # ], dim=1)

        return x

import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HGTConv


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

        # mean + add + max pooling
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

class GATEdge(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=8, num_output_channels=4, heads=4):
        '''
        GAT with edge features (delta position + distance).
        edge_dim=4 matches (dx, dy, dz, dist) computed in the dataset.
        '''
        super(GATEdge, self).__init__()

        self.conv1 = torch_geometric.nn.GATConv(in_feat, h_feat, heads=heads, edge_dim=4)
        self.conv2 = torch_geometric.nn.GATConv(h_feat * heads, h_feat * 2, heads=heads, edge_dim=4)
        self.conv3 = torch_geometric.nn.GATConv(h_feat * 2 * heads, h_feat * 4, heads=heads, edge_dim=4)
        self.conv4 = torch_geometric.nn.GATConv(h_feat * 4 * heads, h_feat * 8, heads=1, concat=False, edge_dim=4)

        # self.bn1 = torch.nn.BatchNorm1d(h_feat * heads)
        # self.bn2 = torch.nn.BatchNorm1d(h_feat * 2 * heads)
        # self.bn3 = torch.nn.BatchNorm1d(h_feat * 4 * heads)
        # self.bn4 = torch.nn.BatchNorm1d(h_feat * 8)

        self.lin = torch.nn.Linear(h_feat * 8, num_output_channels)
        # self.softplus = torch.nn.Softplus()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        # x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr)
        # x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index, edge_attr)
        # x = self.bn4(x)

        x = torch_geometric.nn.global_add_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # x = torch.cat([
        #     x[:, :6],
        #     self.softplus(x[:, 6:7]) + 0.511
        # ], dim=1)
        
        return x

class GATv2(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=8, num_output_channels=4, heads=2):
        '''
        Graph Attention Network v2 (GATv2)
        Uses GATv2Conv instead of GATConv - computes attention on concatenated
        (source, target) pairs rather than independently, making it strictly
        more expressive.
        '''
        super(GATv2, self).__init__()

        self.conv1 = torch_geometric.nn.GATv2Conv(in_feat, h_feat, heads=heads)
        self.conv2 = torch_geometric.nn.GATv2Conv(h_feat * heads, h_feat * 2, heads=heads)
        self.conv3 = torch_geometric.nn.GATv2Conv(h_feat * 2 * heads, h_feat * 4, heads=heads)
        self.conv4 = torch_geometric.nn.GATv2Conv(h_feat * 4 * heads, h_feat * 8, heads=1, concat=False)

        # self.bn1 = torch.nn.BatchNorm1d(h_feat * heads)
        # self.bn2 = torch.nn.BatchNorm1d(h_feat * 2 * heads)
        # self.bn3 = torch.nn.BatchNorm1d(h_feat * 4 * heads)
        # self.bn4 = torch.nn.BatchNorm1d(h_feat * 8)

        self.lin = torch.nn.Linear(h_feat * 8, num_output_channels)
        # self.softplus = torch.nn.Softplus()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        # x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        # x = self.bn4(x)

        x = torch_geometric.nn.global_add_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # x = torch.cat([
        #     x[:, :6],
        #     self.softplus(x[:, 6:7]) + 0.511
        # ], dim=1)

        return x
        
class GAT(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=8, num_output_channels=4, heads=4):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        
        self.conv1 = torch_geometric.nn.GATConv(in_feat, h_feat, heads=heads)
        self.conv2 = torch_geometric.nn.GATConv(h_feat * heads, h_feat * 2, heads=heads)
        self.conv3 = torch_geometric.nn.GATConv(h_feat * 2 * heads, h_feat * 4, heads=heads)
        self.conv4 = torch_geometric.nn.GATConv(h_feat * 4 * heads, h_feat * 8, heads=1, concat=False)

        # Projection layers to match dims for residual addition
        self.res1 = torch.nn.Linear(in_feat, h_feat * heads)
        self.res2 = torch.nn.Linear(h_feat * heads, h_feat * 2 * heads)
        self.res3 = torch.nn.Linear(h_feat * 2 * heads, h_feat * 4 * heads)
        self.res4 = torch.nn.Linear(h_feat * 4 * heads, h_feat * 8)

        self.lin = torch.nn.Linear(h_feat * 8, num_output_channels)
        self.softplus = torch.nn.Softplus()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index) + self.res1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index) + self.res2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index) + self.res3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index) + self.res4(x)

        x = torch_geometric.nn.global_add_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
# class GAT(torch.nn.Module):
#     def __init__(self, in_feat=8, h_feat=8, num_output_channels=4, heads=4):
#         '''
#         Graph Attention Network (GAT)
#         Uses GATConv operator with multi-head attention for message passing.
#         '''
#         super(GAT, self).__init__()
#         torch.manual_seed(12345)
        
#         # heads=4 means 4 attention heads, outputs get concatenated
#         self.conv1 = torch_geometric.nn.GATConv(in_feat, h_feat, heads=heads)
#         self.conv2 = torch_geometric.nn.GATConv(h_feat * heads, h_feat * 2, heads=heads)
#         self.conv3 = torch_geometric.nn.GATConv(h_feat * 2 * heads, h_feat * 4, heads=heads)
#         self.conv4 = torch_geometric.nn.GATConv(h_feat * 4 * heads, h_feat * 8, heads=1, concat=False)  # Final layer: single head
        
#         # self.bn1 = torch.nn.BatchNorm1d(h_feat * heads)
#         # self.bn2 = torch.nn.BatchNorm1d(h_feat * 2 * heads)
#         # self.bn3 = torch.nn.BatchNorm1d(h_feat * 4 * heads)
#         # self.bn4 = torch.nn.BatchNorm1d(h_feat * 8)
        
#         self.lin = torch.nn.Linear(h_feat * 8, num_output_channels)
#         self.softplus = torch.nn.Softplus()

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
#         x = self.conv1(x, edge_index)
#         # x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.conv2(x, edge_index)
#         # x = self.bn2(x)
#         x = F.relu(x)
        
#         x = self.conv3(x, edge_index)
#         # x = self.bn3(x)
#         x = F.relu(x)
        
#         x = self.conv4(x, edge_index)
#         # x = self.bn4(x)

#         x = torch_geometric.nn.global_add_pool(x, batch)

#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#         # x = torch.cat([
#         #     x[:, :6], # !!!!! hardcoded - relies on energy being the 6th index 
#         #     self.softplus(x[:, 6:7]) + 0.511 
#         # ], dim=1)

#         return x
        
class GCN(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=8, num_output_channels=4):
        '''
        Graph Convolutional Network (GCN)
        The Graph Neural Network from the 
        “Semi-supervised Classification with Graph Convolutional Networks” paper, 
        using the GCNConv operator for message passing.
        '''
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = torch_geometric.nn.GCNConv(in_feat, h_feat)
        self.conv2 = torch_geometric.nn.GCNConv(h_feat, h_feat*2)
        self.conv3 = torch_geometric.nn.GCNConv(h_feat*2, h_feat*4)
        self.conv4 = torch_geometric.nn.GCNConv(h_feat*4, h_feat*8)
        self.lin = torch.nn.Linear(h_feat * 8, num_output_channels)
        self.bn1 = torch.nn.BatchNorm1d(h_feat)
        self.bn2 = torch.nn.BatchNorm1d(h_feat*2)
        self.bn3 = torch.nn.BatchNorm1d(h_feat*4)
        self.bn4 = torch.nn.BatchNorm1d(h_feat*8)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        # x = x.tanh()
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = x.tanh()
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        # x = x.tanh()#
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)

        # 2. Readout layer
        # x = torch_geometric.nn.global_mean_pool(
        #     x, batch)  # [batch_size, hidden_channels]

        # try add pool instead of mean pool
        x = torch_geometric.nn.global_add_pool(
            x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class ResGCN(torch.nn.Module):
    def __init__(self,  in_feat=8, h_feat=128, num_classes=4, num_layers=6, dropout=0.1):
        '''
        Residual Graph Convolutional Network (ResGCN)
        The skip connection operations from the 
        “DeepGCNs: Can GCNs Go as Deep as CNNs?” 
        and “All You Need to Train Deeper GCNs” papers.
        The implemented skip connections includes the pre-activation residual connection ("res+"), 
        the residual connection ("res"), the dense connection ("dense") and no connections ("plain").
        '''
        super().__init__()

        self.node_encoder = torch.nn.Linear(in_feat, h_feat)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = torch_geometric.nn.GENConv(
                h_feat, h_feat, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = torch.nn.LayerNorm(h_feat, elementwise_affine=True)
            act = torch.nn.ReLU(inplace=True)

            layer = torch_geometric.nn.DeepGCNLayer(
                conv, norm, act, block='res+', dropout=dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.classifier = torch.nn.Linear(h_feat, num_classes)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.node_encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)

        x = torch_geometric.nn.global_add_pool(x, batch)
        return self.classifier(x)


class GINModel(torch.nn.Module):
    def __init__(self, in_feat=8, h_feat=128, num_classes=4, dropout=0.5):
        '''
        Graph Isomorphism Network (GIN)
        The Graph Neural Network from the 
        “How Powerful are Graph Neural Networks?” paper, 
        using the GINConv operator for message passing.
        '''
        super().__init__()
        self.gnn = torch_geometric.nn.GIN(
            in_feat, h_feat, 3, dropout=0.5, jk='cat')
        self.classifier = torch_geometric.nn.MLP(
            [h_feat, h_feat*2, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.gnn(x, edge_index)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.classifier(x)
        return x


class DyEdCNN(torch.nn.Module):
    def __init__(self, in_feat=8, num_classes=4, k=35, dropout_rate=0.1, aggr='max'):
        '''
        Dynamic Edge Convolutional Neural Network (DyEdCNN)
        The Graph Neural Network from the 
        “Dynamic Graph CNN for Learning on Point Clouds” paper,
        using the EdgeConv operator for message passing.
        '''
        super().__init__()

        self.conv1 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([2 * in_feat, 64, 64, 64]), k, aggr)
        self.conv2 = torch_geometric.nn.DynamicEdgeConv(
            torch_geometric.nn.MLP([2 * 64, 128]), k, aggr)
        self.lin1 = torch.nn.Linear(128 + 64, 256)

        self.mlp = torch_geometric.nn.MLP(
            [256, 128, 64, num_classes], dropout=dropout_rate, norm="batch_norm")

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = torch_geometric.nn.global_max_pool(out, batch)
        out = self.mlp(out)
        return out
