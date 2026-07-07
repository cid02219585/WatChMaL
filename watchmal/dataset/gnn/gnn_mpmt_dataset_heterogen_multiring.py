import numpy as np
import torch

from watchmal.dataset.h5_dataset_multiring import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du
import torch_geometric.data as PyGData
import h5py
from torch_cluster import knn_graph

# from watchmal.model import build_same_mpmt_edges

from torch_geometric.data import HeteroData

barrel_map_array_idxs = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18], dtype=np.int16)
pmts_per_mpmt = 19


# idk if it should be only hit mpmts or empty ones too - quite sparse no?
# i think need to make it spatial and time realted for multiring side


class GNNMultiPMTDataset(H5Dataset): # renamed for GNNs

    def __init__(self, h5file, geometry_file, k_neighbors, use_orientations=False, transforms=None, is_distributed=True, max_points=None, use_memmap=True):
        super().__init__(h5file, use_memmap)

        geo_file = np.load(geometry_file, 'r')
        geo_positions = torch.from_numpy(geo_file["position"]).float()
        geo_orientations = torch.from_numpy(geo_file["orientation"]).float()
        self.pmt_positions = geo_file["position"].T
        # self.mpmt_positions = geo_positions[18::19, :].T # 18th is the reference pmt - what you take for the pos and orientation - guessing it's the central one?
        # self.mpmt_orientations = geo_orientations[18::19, :].T
        self.mpmt_positions = geo_file["position"][18::19, :].T   # numpy
        self.mpmt_orientations = geo_file["orientation"][18::19, :].T  # numpy

        mpmt_y = np.abs(self.mpmt_positions[1, :])
        self.barrel_mpmts = np.where(mpmt_y < mpmt_y.max() - 10)[0].astype(np.int16)

        self.k_neighbors = k_neighbors
        self.max_points = max_points

        self.use_orientations = use_orientations

### max points currently doesn't do anything -- it's always hits only in this current implementation

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        hit_mpmts = self.event_hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = self.event_hit_pmts % pmts_per_mpmt
        hit_barrel = np.where(np.in1d(hit_mpmts, self.barrel_mpmts))[0]
        hit_pmt_in_modules[hit_barrel] = barrel_map_array_idxs[hit_pmt_in_modules[hit_barrel]]

        unique_mpmts, inverse_indices = np.unique(hit_mpmts, return_inverse=True)
        n_mpmts = len(unique_mpmts)

        mpmt_pos = self.mpmt_positions[:, unique_mpmts].T        
        mpmt_ori = self.mpmt_orientations[:, unique_mpmts].T
        n_hits = np.zeros(n_mpmts, dtype=np.float32)
        total_charge = np.zeros(n_mpmts, dtype=np.float32)
        mean_time = np.zeros(n_mpmts, dtype=np.float32)
        np.add.at(n_hits, inverse_indices, 1)
        np.add.at(total_charge, inverse_indices, self.event_hit_charges)
        np.add.at(mean_time, inverse_indices, self.event_hit_times)
        mean_time /= np.maximum(n_hits, 1)

        if self.use_orientations:
            mpmt_features = np.concatenate([
                mpmt_pos / 100.,
                mpmt_ori,
                n_hits[:, None] / 19.,
                total_charge[:, None],
                mean_time[:, None] / 1000.
            ], axis=1)
        else:
            mpmt_features = np.concatenate([
                mpmt_pos / 100.,
                n_hits[:, None] / 19.,
                total_charge[:, None],
                mean_time[:, None] / 1000.
            ], axis=1)

        # mpmt_features = np.concatenate([
        #     mpmt_pos / 100.,
        #     mpmt_ori,
        #     n_hits[:, None] / 19.,       # normalise by max PMTs per module
        #     total_charge[:, None],
        #     mean_time[:, None] / 1000.
        # ], axis=1)                         # (n_mpmts, 9)

        # all_pmt_positions = self.pmt_positions 
        # hit_pmt_global_pos = all_pmt_positions[:, self.event_hit_pmts].T 
        hit_pmt_global_pos = self.pmt_positions[:, self.event_hit_pmts].T
        parent_mpmt_pos = self.mpmt_positions[:, hit_mpmts].T    
        local_pos = (hit_pmt_global_pos - parent_mpmt_pos) / 100.
        global_pos = hit_pmt_global_pos / 100.

        pmt_features = np.concatenate([
            self.event_hit_charges[:, None],
            self.event_hit_times[:, None] / 1000.,
            global_pos,
            local_pos,
            # hit_pmt_in_modules[:, None] / 19.
        ], axis=1)

        # PMT -> mPMT: each hit PMT connects to its parent mPMT
        # inverse_indices maps hit PMT i -> index in unique_mpmts
        n_hits_total = len(self.event_hit_pmts)
        pmt_to_mpmt = torch.tensor(
            np.array([np.arange(n_hits_total), inverse_indices]), dtype=torch.long
        )
        mpmt_to_pmt = pmt_to_mpmt.flip(0)

        # same_mpmt = build_same_mpmt_edges(pmt_to_mpmt, n_mpmts)
        # hetero_data['pmt', 'same_mpmt', 'pmt'].edge_index = same_mpmt

        # mPMT -> mPMT: k-NN on mPMT positions
        mpmt_pos_tensor = torch.tensor(mpmt_pos / 100., dtype=torch.float32)
        mpmt_to_mpmt = knn_graph(mpmt_pos_tensor, k=min(self.k_neighbors, n_mpmts - 1)).long()

        # edge features for mPMT->mPMT edges - not used , hgt doesn't accept edge features
        row, col = mpmt_to_mpmt
        delta = mpmt_pos_tensor[row] - mpmt_pos_tensor[col]
        dist = delta.norm(dim=1, keepdim=True)
        mpmt_edge_attr = torch.cat([delta, dist], dim=1)  # (E, 4)


        # pmt_pos_tensor = torch.tensor(global_pos, dtype=torch.float32)

        # if n_hits_total > 1:
        #     pmt_to_pmt = knn_graph(
        #         pmt_pos_tensor,
        #         k=min(3, n_hits_total - 1)
        #     ).long()
        # else:
        #     pmt_to_pmt = torch.empty((2, 0), dtype=torch.long)

        # PMT -> PMT: connect hit PMTs inside the same mPMT
        pmt_edges = []

        for mpmt_local_idx in range(n_mpmts):
            # indices of hit PMT nodes belonging to this local mPMT
            hit_idxs = np.where(inverse_indices == mpmt_local_idx)[0]

            # no PMT-PMT edges possible if only one hit PMT in this mPMT
            if len(hit_idxs) <= 1:
                continue

            # local PMT positions within this mPMT
            pos = torch.tensor(local_pos[hit_idxs], dtype=torch.float32)

            # connect each hit PMT to nearest neighbours inside the same mPMT
            k_pmt = min(3, len(hit_idxs) - 1)

            local_edge_index = knn_graph(pos, k=k_pmt).long()

            # remap local indices back to global hit-PMT node indices
            global_edge_index = torch.tensor(hit_idxs, dtype=torch.long)[local_edge_index]

            pmt_edges.append(global_edge_index)

        if len(pmt_edges) > 0:
            pmt_to_pmt = torch.cat(pmt_edges, dim=1)
        else:
            pmt_to_pmt = torch.empty((2, 0), dtype=torch.long)

        hetero_data = HeteroData()

        hetero_data['pmt'].x = torch.tensor(pmt_features, dtype=torch.float32)
        hetero_data['mpmt'].x = torch.tensor(mpmt_features, dtype=torch.float32)

        hetero_data['pmt', 'neighbours', 'pmt'].edge_index = pmt_to_pmt
        hetero_data['pmt', 'belongs_to', 'mpmt'].edge_index = pmt_to_mpmt
        hetero_data['mpmt', 'contains', 'pmt'].edge_index = mpmt_to_pmt
        hetero_data['mpmt', 'neighbours', 'mpmt'].edge_index = mpmt_to_mpmt
        hetero_data['mpmt', 'neighbours', 'mpmt'].edge_attr = mpmt_edge_attr

        for key in ['positions', 'directions', 'energies', 'angles']:
            if key in data_dict:
                val = torch.tensor(data_dict[key], dtype=torch.float32)
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                hetero_data[key] = val

        hetero_data['indices'] = torch.tensor([item], dtype=torch.long)

        # virtual_node = np.array([[
        #     n_mpmts / self.mpmt_positions.shape[1], 
        #     np.sum(total_charge),
        #     mean_time.mean(),
        #     self.event_hit_times.min(),
        # ]])

        virtual_node = np.array([[
            n_mpmts / self.mpmt_positions.shape[1],
            np.sum(total_charge) / n_mpmts,  # mean charge per mPMT rather than total
            mean_time.mean() / 1000.,         # consistent with pmt time normalisation
            self.event_hit_times.min() / 1000.,
        ]])
        hetero_data['virtual_node'].x = torch.tensor(virtual_node, dtype=torch.float32)

        all_mpmt_idx = torch.arange(n_mpmts, dtype=torch.long)
        global_idx = torch.zeros(n_mpmts, dtype=torch.long)

        hetero_data['mpmt', 'reports_to', 'virtual_node'].edge_index = torch.stack([all_mpmt_idx, global_idx])
        hetero_data['virtual_node', 'attends_to', 'mpmt'].edge_index = torch.stack([global_idx, all_mpmt_idx])

        # hetero_data['positions'] = hetero_data['positions'][:, [2, 1, 0]] 
        
        # print(f"positions shape: {hetero_data['positions'].shape}, values: {hetero_data['positions']}")
        return hetero_data
