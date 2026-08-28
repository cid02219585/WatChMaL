"""GNN mPMT dataset for mixed one-ring / two-ring reconstruction."""

import numpy as np
import torch
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData

from watchmal.dataset.h5_dataset_single_multi import H5Dataset


barrel_map_array_idxs = np.array(
    [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18],
    dtype=np.int16,
)

pmts_per_mpmt = 19


class GNNMultiPMTDataset(H5Dataset):
    """
    Graph representation for the combined single-ring/two-ring H5 dataset.

    Input graph construction is intentionally kept the same as your existing
    GNNMultiPMTDataset. The main difference is that the base H5 loader can
    retrieve an event from either source H5 and returns ``ring_mask``.
    """

    def __init__(
        self,
        single_h5file,
        multi_h5file,
        geometry_file,
        k_neighbors,
        use_orientations=False,
        transforms=None,
        is_distributed=True,
        max_points=None,
        use_memmap=True,
    ):
        super().__init__(
            single_h5file=single_h5file,
            multi_h5file=multi_h5file,
            use_memmap=use_memmap,
        )

        geo_file = np.load(geometry_file, "r")

        self.pmt_positions = geo_file["position"].T
        self.mpmt_positions = geo_file["position"][18::19, :].T
        self.mpmt_orientations = geo_file["orientation"][18::19, :].T

        mpmt_y = np.abs(self.mpmt_positions[1, :])
        self.barrel_mpmts = np.where(
            mpmt_y < mpmt_y.max() - 10
        )[0].astype(np.int16)

        self.k_neighbors = k_neighbors
        self.max_points = max_points
        self.use_orientations = use_orientations

        # Kept in the signature for compatibility with your existing configs.
        self.transforms = transforms
        self.is_distributed = is_distributed

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        # ====================================================
        # Hit PMT / mPMT bookkeeping
        # ====================================================
        hit_mpmts = self.event_hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = self.event_hit_pmts % pmts_per_mpmt

        hit_barrel = np.where(
            np.isin(hit_mpmts, self.barrel_mpmts)
        )[0]

        hit_pmt_in_modules = hit_pmt_in_modules.copy()
        hit_pmt_in_modules[hit_barrel] = barrel_map_array_idxs[
            hit_pmt_in_modules[hit_barrel]
        ]

        unique_mpmts, inverse_indices = np.unique(
            hit_mpmts,
            return_inverse=True,
        )

        n_mpmts = len(unique_mpmts)

        if n_mpmts == 0:
            raise RuntimeError(
                f"Event {item} contains no hit mPMTs and cannot form a graph."
            )

        # ====================================================
        # mPMT node features
        # ====================================================
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
            mpmt_features = np.concatenate(
                [
                    mpmt_pos / 100.0,
                    mpmt_ori,
                    n_hits[:, None] / 19.0,
                    total_charge[:, None],
                    mean_time[:, None] / 1000.0,
                ],
                axis=1,
            )
        else:
            mpmt_features = np.concatenate(
                [
                    mpmt_pos / 100.0,
                    n_hits[:, None] / 19.0,
                    total_charge[:, None],
                    mean_time[:, None] / 1000.0,
                ],
                axis=1,
            )

        # ====================================================
        # PMT node features
        # ====================================================
        hit_pmt_global_pos = self.pmt_positions[:, self.event_hit_pmts].T
        parent_mpmt_pos = self.mpmt_positions[:, hit_mpmts].T

        local_pos = (
            hit_pmt_global_pos - parent_mpmt_pos
        ) / 100.0

        global_pos = hit_pmt_global_pos / 100.0

        pmt_features = np.concatenate(
            [
                self.event_hit_charges[:, None],
                self.event_hit_times[:, None] / 1000.0,
                global_pos,
                local_pos,
            ],
            axis=1,
        )

        # ====================================================
        # PMT <-> parent mPMT edges
        # ====================================================
        n_hits_total = len(self.event_hit_pmts)

        pmt_to_mpmt = torch.tensor(
            np.array(
                [
                    np.arange(n_hits_total),
                    inverse_indices,
                ]
            ),
            dtype=torch.long,
        )

        mpmt_to_pmt = pmt_to_mpmt.flip(0)

        # ====================================================
        # mPMT -> mPMT spatial kNN edges
        # ====================================================
        mpmt_pos_tensor = torch.tensor(
            mpmt_pos / 100.0,
            dtype=torch.float32,
        )

        if n_mpmts > 1:
            mpmt_to_mpmt = knn_graph(
                mpmt_pos_tensor,
                k=min(self.k_neighbors, n_mpmts - 1),
            ).long()
        else:
            mpmt_to_mpmt = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        row, col = mpmt_to_mpmt

        if mpmt_to_mpmt.shape[1] > 0:
            delta = mpmt_pos_tensor[row] - mpmt_pos_tensor[col]
            dist = delta.norm(dim=1, keepdim=True)
            mpmt_edge_attr = torch.cat([delta, dist], dim=1)
        else:
            mpmt_edge_attr = torch.empty(
                (0, 4),
                dtype=torch.float32,
            )

        # ====================================================
        # PMT -> PMT edges inside each mPMT
        # ====================================================
        pmt_edges = []

        for mpmt_local_idx in range(n_mpmts):
            hit_idxs = np.where(
                inverse_indices == mpmt_local_idx
            )[0]

            if len(hit_idxs) <= 1:
                continue

            pos = torch.tensor(
                local_pos[hit_idxs],
                dtype=torch.float32,
            )

            k_pmt = min(3, len(hit_idxs) - 1)

            local_edge_index = knn_graph(
                pos,
                k=k_pmt,
            ).long()

            global_edge_index = torch.tensor(
                hit_idxs,
                dtype=torch.long,
            )[local_edge_index]

            pmt_edges.append(global_edge_index)

        if pmt_edges:
            pmt_to_pmt = torch.cat(
                pmt_edges,
                dim=1,
            )
        else:
            pmt_to_pmt = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        # ====================================================
        # Build heterogeneous graph
        # ====================================================
        hetero_data = HeteroData()

        hetero_data["pmt"].x = torch.tensor(
            pmt_features,
            dtype=torch.float32,
        )

        hetero_data["mpmt"].x = torch.tensor(
            mpmt_features,
            dtype=torch.float32,
        )

        hetero_data[
            "pmt", "neighbours", "pmt"
        ].edge_index = pmt_to_pmt

        hetero_data[
            "pmt", "belongs_to", "mpmt"
        ].edge_index = pmt_to_mpmt

        hetero_data[
            "mpmt", "contains", "pmt"
        ].edge_index = mpmt_to_pmt

        hetero_data[
            "mpmt", "neighbours", "mpmt"
        ].edge_index = mpmt_to_mpmt

        hetero_data[
            "mpmt", "neighbours", "mpmt"
        ].edge_attr = mpmt_edge_attr

        # ====================================================
        # Graph-level truth
        #
        # PyG concatenates graph-level tensors across a batch.
        # The leading singleton dimension therefore matters:
        #
        # positions: [1, 2, 3] per graph -> [B, 2, 3]
        # ring_mask: [1, 2] per graph     -> [B, 2]
        # ====================================================
        hetero_data["positions"] = torch.tensor(
            data_dict["positions"],
            dtype=torch.float32,
        ).unsqueeze(0)

        hetero_data["directions"] = torch.tensor(
            data_dict["directions"],
            dtype=torch.float32,
        ).unsqueeze(0)

        hetero_data["ring_mask"] = torch.tensor(
            data_dict["ring_mask"],
            dtype=torch.float32,
        ).unsqueeze(0)

        hetero_data["n_rings"] = torch.tensor(
            [data_dict["n_rings"]],
            dtype=torch.long,
        )

        hetero_data["source"] = torch.tensor(
            [data_dict["source"]],
            dtype=torch.long,
        )

        hetero_data["indices"] = torch.tensor(
            [data_dict["indices"]],
            dtype=torch.long,
        )

        hetero_data["local_indices"] = torch.tensor(
            [data_dict["local_indices"]],
            dtype=torch.long,
        )


        # ====================================================
        # Virtual node - same feature definition as your
        # existing dataset.
        # ====================================================
        virtual_node = np.array(
            [
                [
                    n_mpmts / self.mpmt_positions.shape[1],
                    np.sum(total_charge) / n_mpmts,
                    mean_time.mean() / 1000.0,
                    self.event_hit_times.min() / 1000.0,
                ]
            ],
            dtype=np.float32,
        )

        hetero_data["virtual_node"].x = torch.tensor(
            virtual_node,
            dtype=torch.float32,
        )

        all_mpmt_idx = torch.arange(
            n_mpmts,
            dtype=torch.long,
        )

        global_idx = torch.zeros(
            n_mpmts,
            dtype=torch.long,
        )

        hetero_data[
            "mpmt", "reports_to", "virtual_node"
        ].edge_index = torch.stack(
            [all_mpmt_idx, global_idx]
        )

        hetero_data[
            "virtual_node", "attends_to", "mpmt"
        ].edge_index = torch.stack(
            [global_idx, all_mpmt_idx]
        )

        return hetero_data