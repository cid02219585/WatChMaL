### GNN dataset with hit mPMTs only as nodes
# Code adapted from the PointNet mPMT dataset

import numpy as np
import torch

from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du
import torch_geometric.data as PyGData
import h5py
from torch_cluster import knn_graph

barrel_map_array_idxs = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18], dtype=np.int16)
pmts_per_mpmt = 19

class GNNMultiPMTDataset(H5Dataset): # renamed for GNNs

    def __init__(self, h5file, geometry_file, k_neighbors, use_orientations=False, transforms=None, is_distributed=True, max_points=None, use_memmap=True):
        super().__init__(h5file, use_memmap)

        geo_file = np.load(geometry_file, 'r')
        geo_positions = torch.from_numpy(geo_file["position"]).float()
        geo_orientations = torch.from_numpy(geo_file["orientation"]).float()
        self.mpmt_positions = geo_positions[18::19, :].T # 18th is the reference pmt
        self.mpmt_orientations = geo_orientations[18::19, :].T

        mpmt_y = np.abs(self.mpmt_positions[1, :])
        self.barrel_mpmts = np.where(mpmt_y < mpmt_y.max() - 10)[0].astype(np.int16)

        self.k_neighbors = k_neighbors
        self.max_points = max_points

    def __getitem__(self, item):

        data_dict = super().__getitem__(item)

        hit_mpmts = self.event_hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = self.event_hit_pmts % pmts_per_mpmt
        hit_barrel = np.where(np.in1d(hit_mpmts, self.barrel_mpmts))[0]
        hit_pmt_in_modules[hit_barrel] = barrel_map_array_idxs[hit_pmt_in_modules[hit_barrel]]

        if self.max_points is not None:
            unique_mpmts, inverse_indices = np.unique(hit_mpmts, return_inverse=True)
            n_points = len(unique_mpmts)

            data = np.zeros((41, n_points), dtype=np.float32)
            data[:3, :] = self.mpmt_positions[:, unique_mpmts]
            
            charge_channels = hit_pmt_in_modules + 3
            time_channels = hit_pmt_in_modules + 22
            data[charge_channels, inverse_indices] = self.event_hit_charges
            data[time_channels, inverse_indices] = self.event_hit_times
        else:
            n_points = self.mpmt_positions.shape[1]
            all_mpmts = np.arange(n_points)

            data = np.zeros((41, n_points), dtype=np.float32)
            data[:3, :] = self.mpmt_positions[:, all_mpmts]
            
            charge_channels = hit_pmt_in_modules + 3
            time_channels = hit_pmt_in_modules + 22
            data[charge_channels, hit_mpmts] = self.event_hit_charges
            data[time_channels, hit_mpmts] = self.event_hit_times

        data = data.T
        
        scale = np.ones(41, dtype=np.float32)
        scale[:3] = 100.
        scale[3:22] = 1.
        scale[22:41] = 1000.
        data /= scale

        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                if key == "edge_index":
                    data_dict[key] = torch.tensor(value, dtype=torch.long)
                elif key in ["positions", "directions", "energies", "angles", "three_momenta"]:
                    val = torch.tensor(value, dtype=torch.float32)
                    if val.dim() == 0:
                        val = val.unsqueeze(0).unsqueeze(0)
                    elif val.dim() == 1:
                        val = val.unsqueeze(0)
                    data_dict[key] = val
                else:
                    data_dict[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, (int, np.integer)):
                data_dict[key] = torch.tensor([value], dtype=torch.long)
            elif isinstance(value, (float, np.floating)):
                data_dict[key] = torch.tensor([[value]], dtype=torch.float32)

        x = torch.tensor(data, dtype=torch.float32)
        edge_index = knn_graph(x[:, 0:3], k=min(self.k_neighbors, n_points - 1)).long()

        data_dict["x"] = x
        data_dict["edge_index"] = edge_index


        row, col = edge_index
        delta_pos = x[row, :3] - x[col, :3]
        dist = delta_pos.norm(dim=1, keepdim=True)

        unit_dir = delta_pos / (dist + 1e-8)
        dist_norm = dist / dist.max()

        edge_attr = torch.cat([unit_dir, dist_norm], dim=1)  # (E, 4)

        data_dict["edge_attr"] = edge_attr

        return PyGData.Data.from_dict(data_dict)