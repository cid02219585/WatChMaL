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
    def __init__(self, pmt_in=6, mpmt_in=6, h_feat=64, heads=4, num_output_channels=7):
        super().__init__()

        node_types = ['pmt', 'mpmt']
        edge_types = [
            ('pmt', 'belongs_to', 'mpmt'),
            ('mpmt', 'has_pmt', 'pmt'),
            ('mpmt', 'neighbours', 'mpmt'),
        ]
        metadata = (node_types, edge_types)

        self.conv1 = HGTConv(
            in_channels={'pmt': pmt_in, 'mpmt': mpmt_in},
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
        self.conv4 = HGTConv(
            in_channels={'pmt': h_feat, 'mpmt': h_feat},
            out_channels=h_feat, metadata=metadata, heads=heads
        )

        # BatchNorm for each conv layer, separate for each node type
        self.bn1_pmt  = torch.nn.BatchNorm1d(h_feat)
        self.bn1_mpmt = torch.nn.BatchNorm1d(h_feat)
        self.bn2_pmt  = torch.nn.BatchNorm1d(h_feat)
        self.bn2_mpmt = torch.nn.BatchNorm1d(h_feat)
        self.bn3_pmt  = torch.nn.BatchNorm1d(h_feat)
        self.bn3_mpmt = torch.nn.BatchNorm1d(h_feat)
        self.bn4_mpmt = torch.nn.BatchNorm1d(h_feat)  # conv4: only mpmt feeds into pool

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(h_feat, h_feat // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(h_feat // 2, num_output_channels)
        )

    def forward(self, data):
        x_dict = {
            'pmt':  data['pmt'].x,
            'mpmt': data['mpmt'].x
        }
        edge_index_dict = {
            ('pmt',  'belongs_to', 'mpmt'): data['pmt', 'belongs_to', 'mpmt'].edge_index,
            ('mpmt', 'neighbours', 'mpmt'): data['mpmt', 'neighbours', 'mpmt'].edge_index,
            ('mpmt', 'has_pmt',    'pmt'):  data['pmt', 'belongs_to', 'mpmt'].edge_index.flip(0),
        }

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['pmt']  = F.relu(self.bn1_pmt(x_dict['pmt']))
        x_dict['mpmt'] = F.relu(self.bn1_mpmt(x_dict['mpmt']))

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict['pmt']  = F.relu(self.bn2_pmt(x_dict['pmt']))
        x_dict['mpmt'] = F.relu(self.bn2_mpmt(x_dict['mpmt']))

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict['pmt']  = F.relu(self.bn3_pmt(x_dict['pmt']))
        x_dict['mpmt'] = F.relu(self.bn3_mpmt(x_dict['mpmt']))

        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict['mpmt'] = F.relu(self.bn4_mpmt(x_dict['mpmt']))
        # pmt not normalised after conv4 since it doesn't feed into the pool

        x = pyg_nn.global_mean_pool(x_dict['mpmt'], data['mpmt'].batch)

        return self.mlp(x)

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
