"""H5 loader for a combined single-ring / two-ring dataset."""

import h5py
import numpy as np
from torch.utils.data import Dataset
from watchmal.utils.math import direction_from_angles, momentum_from_energy

class H5Dataset(Dataset):
    """
    Load events from two HDF5 files through one global index space.

    Index convention
    ----------------
    0 <= item < n_single
        -> single_h5[item]

    n_single <= item < n_single + n_multi
        -> multi_h5[item - n_single]

    The class deliberately exposes ``event_hit_pmts``,
    ``event_hit_times`` and ``event_hit_charges`` in the same way as the
    existing WatChMaL H5Dataset so downstream detector representations can
    continue to use them.

    Position targets are always returned with two slots:

        single ring: positions.shape == (2, 3), ring_mask == [1, 0]
        two rings:   positions.shape == (2, 3), ring_mask == [1, 1]

    The empty position slot for a single-ring event is zero padded. The loss
    must use ``ring_mask`` so that this padded slot receives no position loss.
    """

    def __init__(
        self,
        single_h5file,
        multi_h5file,
        use_memmap=True,
    ):
        self.single_h5file = single_h5file
        self.multi_h5file = multi_h5file
        self.use_memmap = use_memmap

        with h5py.File(self.single_h5file, "r") as h5_file:
            self.n_single = h5_file["event_hits_index"].shape[0]

        with h5py.File(self.multi_h5file, "r") as h5_file:
            self.n_multi = h5_file["event_hits_index"].shape[0]

        self.dataset_length = self.n_single + self.n_multi

        self.initialized = False
        self.target_key = None

        self.single_h5 = None
        self.multi_h5 = None

        self.single_event_hits_index = None
        self.multi_event_hits_index = None

        self.single_hit_pmt = None
        self.single_hit_time = None
        self.single_hit_charge = None

        self.multi_hit_pmt = None
        self.multi_hit_time = None
        self.multi_hit_charge = None

        # Interface used by downstream WatChMaL dataset classes.
        self.event_hit_pmts = None
        self.event_hit_times = None
        self.event_hit_charges = None

    def __len__(self):
        return self.dataset_length

    def set_target(self, target_key):
        """Store the requested target key(s), matching the normal H5Dataset API."""
        self.target_key = target_key

    def _load_hits(self, h5_file, h5_path, h5_key):
        data = h5_file[h5_key]

        if self.use_memmap:
            return np.memmap(
                h5_path,
                mode="r",
                shape=data.shape,
                offset=data.id.get_offset(),
                dtype=data.dtype,
            )

        return np.array(data)

    def initialize(self):
        """Open both H5 files lazily inside the dataloader worker."""
        self.single_h5 = h5py.File(self.single_h5file, "r")
        self.multi_h5 = h5py.File(self.multi_h5file, "r")

        self.single_event_hits_index = np.append(
            self.single_h5["event_hits_index"],
            self.single_h5["hit_pmt"].shape[0],
        ).astype(np.int64)

        self.multi_event_hits_index = np.append(
            self.multi_h5["event_hits_index"],
            self.multi_h5["hit_pmt"].shape[0],
        ).astype(np.int64)

        self.single_hit_pmt = self._load_hits(
            self.single_h5,
            self.single_h5file,
            "hit_pmt",
        )
        self.single_hit_time = self._load_hits(
            self.single_h5,
            self.single_h5file,
            "hit_time",
        )
        self.single_hit_charge = self._load_hits(
            self.single_h5,
            self.single_h5file,
            "hit_charge",
        )

        self.multi_hit_pmt = self._load_hits(
            self.multi_h5,
            self.multi_h5file,
            "hit_pmt",
        )
        self.multi_hit_time = self._load_hits(
            self.multi_h5,
            self.multi_h5file,
            "hit_time",
        )
        self.multi_hit_charge = self._load_hits(
            self.multi_h5,
            self.multi_h5file,
            "hit_charge",
        )

        self.initialized = True

    def _resolve_event(self, item):
        """Return H5 objects and local index corresponding to a global index."""
        item = int(item)

        if item < 0 or item >= self.dataset_length:
            raise IndexError(
                f"Combined H5 index {item} outside [0, {self.dataset_length})."
            )

        if item < self.n_single:
            return {
                "source": 0,
                "n_rings": 1,
                "local_idx": item,
                "h5": self.single_h5,
                "event_hits_index": self.single_event_hits_index,
                "hit_pmt": self.single_hit_pmt,
                "hit_time": self.single_hit_time,
                "hit_charge": self.single_hit_charge,
            }

        local_idx = item - self.n_single
        return {
            "source": 1,
            "n_rings": 2,
            "local_idx": local_idx,
            "h5": self.multi_h5,
            "event_hits_index": self.multi_event_hits_index,
            "hit_pmt": self.multi_hit_pmt,
            "hit_time": self.multi_hit_time,
            "hit_charge": self.multi_hit_charge,
        }
    @staticmethod
    def _ordered_multi_targets(h5_file, local_idx):
        positions = np.asarray(
            h5_file["positions"][local_idx],
            dtype=np.float32,
        ).reshape(-1, 3)

        angles = np.asarray(
            h5_file["angles"][local_idx],
            dtype=np.float32,
        )

        directions = direction_from_angles(angles)
        directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        energies = np.asarray(
            h5_file["energies"][local_idx]
        ).reshape(-1)

        order = np.argsort(-energies)

        return positions[order], directions[order]

    def __getitem__(self, item):
        if not self.initialized:
            self.initialize()

        event = self._resolve_event(item)

        local_idx = event["local_idx"]
        n_rings = event["n_rings"]
        h5_file = event["h5"]

        # ----------------------------------------------------
        # Event hits
        # ----------------------------------------------------
        start = event["event_hits_index"][local_idx]
        stop = event["event_hits_index"][local_idx + 1]

        self.event_hit_pmts = event["hit_pmt"][start:stop]
        self.event_hit_times = event["hit_time"][start:stop]
        self.event_hit_charges = event["hit_charge"][start:stop]

        # ----------------------------------------------------
        # Position and direction truth: always two slots
        # ----------------------------------------------------
        if n_rings == 1:
            positions = np.asarray(
                h5_file["positions"][local_idx],
                dtype=np.float32,
            ).reshape(-1, 3)

            angles = np.asarray(
                h5_file["angles"][local_idx],
                dtype=np.float32,
            )

            directions = direction_from_angles(angles)
            directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)

        else:
            positions, directions = self._ordered_multi_targets(
                h5_file,
                local_idx,
            )


        # ----------------------------------------------------
        # Pad single-ring events to two slots
        # ----------------------------------------------------
        padded_positions = np.zeros((2, 3), dtype=np.float32)
        padded_positions[:n_rings] = positions[:n_rings]

        padded_directions = np.zeros((2, 3), dtype=np.float32)
        padded_directions[:n_rings] = directions[:n_rings]


        # ----------------------------------------------------
        # Real-ring mask
        # ----------------------------------------------------
        ring_mask = np.zeros(2, dtype=np.float32)
        ring_mask[:n_rings] = 1.0

        return {
            "positions": padded_positions,
            "directions": padded_directions,
            "ring_mask": ring_mask,
            "n_rings": np.int64(n_rings),
            "source": np.int64(event["source"]),
            "indices": np.int64(item),
            "local_indices": np.int64(local_idx),
        }