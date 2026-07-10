import torch

from watchmal.engine.reconstruction import ReconstructionEngine
from collections.abc import Mapping

# define some useful metrics for different regression targets

def three_momenta_metrics(reco, true):
    reco_mag = torch.linalg.vector_norm(reco, dim=-1)
    true_mag = torch.linalg.vector_norm(true, dim=-1)
    return {'momentum bias': torch.mean((reco_mag - true_mag) / true_mag),
            'momentum error': torch.mean(torch.abs(reco_mag - true_mag) / true_mag),
            'direction error': torch.mean(torch.arccos(torch.clamp(torch.sum(reco * true, dim=-1)
                                                                    / (reco_mag * true_mag), -1, 1)))}

# metric_functions = {
#     'positions':  # mean 3D position error
#         lambda x, y: {'position error': torch.mean(torch.linalg.vector_norm(x-y, dim=1))},
#     'directions':  # mean angle between directions
#         lambda x, y: {'direction error': torch.mean(torch.arccos(torch.clamp(torch.sum(x*y, dim=-1)
#                                                 / torch.linalg.vector_norm(x, dim=-1), -1, 1)))},
#     'angles':  # mean angle between directions
#         lambda x, y: {'direction error': torch.mean(torch.arccos(torch.cos(x[:, 0])*torch.cos(y[:, 0])
#                                                 + torch.sin(x[:, 0])*torch.sin(y[:, 0])*torch.cos(x[:, 1]-y[:, 1])))},
#     'energies':  # mean fractional error
#         lambda x, y: {'energy bias': torch.mean((x - y) / y),
#                       'energy error': torch.mean(torch.abs(x-y)/y)},
#     'three_momenta':  three_momenta_metrics,
# }

metric_functions = {
    'positions':  # mean 3D position error
        lambda x, y: {'position error': torch.mean(torch.linalg.vector_norm(x-y, dim=-1), dim=0)},
    'directions':  # mean angle between directions
        lambda x, y: {'direction error': torch.mean(torch.arccos(torch.clamp(torch.sum(x*y, dim=-1)
                                                / torch.linalg.vector_norm(x, dim=-1), -1, 1)), dim=0)},
    'angles':  # mean angle between directions
        lambda x, y: {'direction error': torch.stack([torch.mean(torch.arccos(torch.cos(x[:,i, 0])*torch.cos(y[:,i, 0]) # cld add clamp if nans
                                                + torch.sin(x[:,i, 0])*torch.sin(y[:,i, 0])*torch.cos(x[:,i, 1]-y[:,i, 1]))) for i in range(x.shape[1])])},
    'energies':  # mean fractional error
        lambda x, y: {'energy bias': torch.mean((x - y) / y, dim=0),
                      'energy error': torch.mean(torch.abs(x-y)/y, dim=0)},
    'three_momenta':  three_momenta_metrics,
}


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, target_key, model, rank, device, dump_path, target_scale_offset=0, target_scale_factor=1, clip_grad_norm=None):
        """
        Parameters
        ==========
        target_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        device : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        target_scale_offset : float or dict of float
            Offset to subtract from target values when calculating the loss, or dict of offsets for each target
        target_scale_factor : float or dict of float
            Scale factor to divide target values by when calculating the loss, or dict of scale factors for each target
        """
        # create the directory for saving the log and dump files
        super().__init__(target_key, model, rank, device, dump_path, clip_grad_norm=clip_grad_norm)
        if isinstance(self.target_key, str):
            self.target_key = [self.target_key]
        self.target_sizes = None
        if isinstance(target_scale_offset, Mapping):  # each target has its own offset
            self.offset = {t: torch.tensor(target_scale_offset.get(t, 0)).to(self.device) for t in self.target_key}
        else:  # each target has the same offset
            self.offset = {t: torch.tensor(target_scale_offset).to(self.device) for t in self.target_key}
        if isinstance(target_scale_factor, Mapping):  # each target has its own scale
            self.scale = {t: torch.tensor(target_scale_factor.get(t, 1)).to(self.device) for t in self.target_key}
        else:  # each target has the same scale
            self.scale = {t: torch.tensor(target_scale_factor).to(self.device) for t in self.target_key}
        self.target_dict = None
        self.stacked_target = None
        self.predictions = None
        self.warmup_iterations = 500

    # def process_target(self, data):
    #     """Extract the event data and target from the input data dict"""
    #     self.target_dict = {t: data[t].to(self.device) for t in self.target_key}
    #     # First time we get data, determine the target sizes
    #     if self.target_sizes is None:
    #         self.target_sizes = [v.shape[-1] if len(v.shape) > 1 else 1 for v in self.target_dict.values()]
    #     # scale and stack the targets for calculating the loss
    #     self.stacked_target = torch.column_stack([(v - self.offset[t]) / self.scale[t] for t, v in self.target_dict.items()])

    # def process_target(self, data):
    #     """Extract target(s) from either a dict batch or a PyG Batch/Data object."""
    #     if isinstance(data, Mapping):
    #         self.target_dict = {t: data[t].to(self.device) for t in self.target_key}
    #     else:
    #         self.target_dict = {t: getattr(data, t).to(self.device) for t in self.target_key}

    #     # if self.target_sizes is None:
    #     #     self.target_sizes = [v.shape[-1] if len(v.shape) > 1 else 1
    #     #                         for v in self.target_dict.values()]
    #     if self.target_sizes is None:
    #         self.target_sizes = [v[0].shape[-1] if len(v[0].shape) > 1 else 1 # taking the first dimension
    #                             for v in self.target_dict.values()]
    #     self.stacked_target = torch.cat([
    #         (v - self.offset[t]) / self.scale[t]
    #         for t, v in self.target_dict.items()
    #     ], dim = -1)

    #     # for t, v in self.target_dict.items():
    #     #     print(f"{t} shape: {v.shape}, first row: {v[0]}")

    def process_target(self, data):
        """Extract target(s) from either a dict batch or a PyG Batch/Data object."""
        if isinstance(data, Mapping):
            self.target_dict = {t: data[t].to(self.device) for t in self.target_key}
        else:
            self.target_dict = {t: getattr(data, t).to(self.device) for t in self.target_key}

        for t, v in self.target_dict.items():
            if v.dim() == 2:
                self.target_dict[t] = v.unsqueeze(-1)
        
        if self.target_sizes is None:
            self.target_sizes = [v[0].shape[-1] if len(v[0].shape) > 1 else 1  # taking the first dimension
                                for v in self.target_dict.values()]

        scaled_targets = []
        for t, v in self.target_dict.items():
            scaled = (v - self.offset[t]) / self.scale[t]
            if scaled.dim() == 2: # bc energies may only
                scaled = scaled.unsqueeze(-1)
            scaled_targets.append(scaled)
            
        self.stacked_target = torch.cat(scaled_targets, dim=-1)

    def forward_pass(self):
        """Compute predictions for a batch of data"""
        # evaluate the model on the data
        self.model_out = self.model(self.data)
        # split the output for each target
        split_model_out = torch.split(self.model_out, self.target_sizes, dim=-1) # along the last dim

        self.predictions = {"predicted_" + t: o* self.scale[t] + self.offset[t]
                        for t, o in zip(self.target_key, split_model_out)}

        # print(f"target x: {self.stacked_target[:3, 0]}")
        # print(f"target y: {self.stacked_target[:3, 1]}")
        # print(f"target z: {self.stacked_target[:3, 2]}")
        # print(f"pred x:   {self.model_out[:3, 0]}")
        # print(f"pred y:   {self.model_out[:3, 1]}")
        # print(f"pred z:   {self.model_out[:3, 2]}")

        if self.target_dict is None:
            return self.predictions
        
        return self.target_dict | self.predictions

    def compute_metrics(self):
        self.loss = self.criterion(self.model_out, self.stacked_target)
        # return loss and metrics for the predictions
        # metrics = {k: m for t, v in self.target_dict.items() if t in metric_functions
        #            for k, m in metric_functions[t](self.predictions["predicted_"+t], v).items()}
        metrics = {}

        for t, v in self.target_dict.items():
            if t not in metric_functions:
                continue
            else:
                for k, m in metric_functions[t](self.predictions["predicted_"+t], v).items():
                    for i in range(v.shape[1]):
                        metrics[k+f'slot{i}'] = m[i].reshape(())

        metrics['loss'] = self.loss

        return metrics

    def compute_metrics(self):
        B = self.model_out.shape[0]
        w_pos = 1.0 / 100.0**2 
        w_dir = 1.0
        w_energy = 1.0

        pred_dict = {
            t: self.predictions["predicted_" + t]
            for t in self.target_key
        }  # (B, 2, d_t)
        true_dict = self.target_dict  # (B, 2, d_t)

        def per_ring_loss(pred_d, true_d):
            # Position: (B, 2)
            pos_loss = ((pred_d["positions"] - true_d["positions"]) ** 2).mean(dim=-1)

            # Direction: (B, 2)
            pred_dir = pred_d["directions"]
            true_dir = true_d["directions"]
            pred_dir_n = pred_dir / (
                torch.linalg.vector_norm(pred_dir, dim=-1, keepdim=True) + 1e-8
            )
            true_dir_n = true_dir / (
                torch.linalg.vector_norm(true_dir, dim=-1, keepdim=True) + 1e-8
            )
            dir_loss = 1.0 - torch.sum(pred_dir_n * true_dir_n, dim=-1)

            # Energy: (B, 2)
            pred_e = torch.clamp(pred_d["energies"].squeeze(-1), min=1e-3)
            true_e = torch.clamp(true_d["energies"].squeeze(-1), min=1e-3)
            energy_loss = (torch.log(pred_e) - torch.log(true_e)) ** 2

            total_loss = (
                w_pos * pos_loss
                + w_dir * dir_loss
                + w_energy * energy_loss
            )
            components = {
                "pos_loss": pos_loss,
                "dir_loss": dir_loss,
                "energy_loss": energy_loss,
            }
            return total_loss, components

        straight_loss, straight_components = per_ring_loss(pred_dict, true_dict)

        true_dict_swapped = {
            t: v.flip(dims=[1])
            for t, v in true_dict.items()
        }
        swapped_loss, swapped_components = per_ring_loss(pred_dict, true_dict_swapped)

        straight_total = straight_loss.sum(dim=-1)  # (B,)
        swapped_total = swapped_loss.sum(dim=-1)    # (B,)

        # Warm-up: matching against untrained predictions is unstable (target flips
        # almost every batch when pred is close to random). Use fixed ("straight")
        # assignment for the first warmup_iterations, then switch to real matching.
        if self.iteration < self.warmup_iterations:
            use_swapped = torch.zeros(B, dtype=torch.bool, device=self.device)
        else:
            use_swapped = swapped_total < straight_total

        matched_total = torch.where(use_swapped, swapped_total, straight_total)
        self.loss = matched_total.mean()

        metrics = {}
        metrics["loss"] = self.loss
        metrics["straight_loss"] = straight_total.mean()
        metrics["swapped_loss"] = swapped_total.mean()
        metrics["swap_fraction"] = use_swapped.float().mean()

        # Component losses for scale debugging
        metrics["straight_pos_loss"] = straight_components["pos_loss"].mean()
        metrics["straight_dir_loss"] = straight_components["dir_loss"].mean()
        metrics["straight_energy_loss"] = straight_components["energy_loss"].mean()

        # Matched targets for normal metric functions
        use_swapped_m = use_swapped.detach().view(B, 1, 1)
        for t, v in self.target_dict.items():
            if t not in metric_functions:
                continue
            pred_t = self.predictions["predicted_" + t]
            v_swapped = v.flip(dims=[1])
            matched_v = torch.where(use_swapped_m, v_swapped, v)
            for i in range(matched_v.shape[1]):
                pred_i = pred_t[:, i]
                true_i = matched_v[:, i]
                for k, m in metric_functions[t](pred_i, true_i).items():
                    metrics[f"{k}_slot{i}"] = m.reshape(())

        return metrics

    def save_state(self, suffix="", name=None):
        self.state_data["target_sizes"] = self.target_sizes
        super().save_state(suffix, name)

    def restore_state(self, weight_file):
        super().restore_state(weight_file)
        if "target_sizes" in self.state_data:
            self.target_sizes = self.state_data["target_sizes"]
