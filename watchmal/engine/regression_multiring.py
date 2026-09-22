### Regression engine for two ring
# Only applicable for two ring events - simplified matching algorithm + no classification in matching cost because of fixed number of rings
# The ablations of auxiliary layer losses and auxiliary direction losses for the compute_metrics function are commented out blocks underneath comments labelling them (all models benefitted from auxiliary layer losses)
# The direction loss is a scaled cos loss (angle only, no magnitude) not huber 

import torch
import torch.nn.functional as F

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


        # split_model_out = torch.split(self.model_out, self.target_sizes, dim=-1) # along the last dim
        final_out = (
            self.model_out[-1]
            if (self.stacked_target is not None
                and self.model_out.dim() == self.stacked_target.dim() + 1)
            else self.model_out
        )
        split_model_out = torch.split(final_out, self.target_sizes, dim=-1) # along the last dim

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



# # ### huber loss
    # def compute_metrics(self):
    #     pred_scaled = self.model_out
    #     true_scaled = self.stacked_target

    #     pairwise_cost = F.huber_loss(
    #         pred_scaled.unsqueeze(2),
    #         true_scaled.unsqueeze(1),
    #         delta=self.criterion.delta,
    #         reduction="none",
    #     ).mean(dim=-1)

    #     # num_pred_slots = pred_scaled.size(1)
    #     # num_true_slots = true_scaled.size(1)

    #     # pred_pairwise = pred_scaled.unsqueeze(2).expand(
    #     #     -1,
    #     #     num_pred_slots,
    #     #     num_true_slots,
    #     #     -1,
    #     # )

    #     # true_pairwise = true_scaled.unsqueeze(1).expand(
    #     #     -1,
    #     #     num_pred_slots,
    #     #     num_true_slots,
    #     #     -1,
    #     # )

    #     # pairwise_cost = F.huber_loss(
    #     #     pred_pairwise,
    #     #     true_pairwise,
    #     #     delta=self.criterion.delta,
    #     #     reduction="none",
    #     # ).mean(dim=-1)

    #     identity_cost = (
    #         pairwise_cost[:, 0, 0]
    #         + pairwise_cost[:, 1, 1]
    #     )

    #     swap_cost = (
    #         pairwise_cost[:, 0, 1]
    #         + pairwise_cost[:, 1, 0]
    #     )

    #     use_swap = swap_cost < identity_cost

    #     self.loss = torch.where(
    #         use_swap,
    #         swap_cost,
    #         identity_cost,
    #     ).mean()

    #     pred_positions = self.predictions["predicted_positions"]
    #     true_positions = self.target_dict["positions"]

    #     matched_true = torch.where(
    #         use_swap[:, None, None],
    #         true_positions.flip(dims=[1]),
    #         true_positions,
    #     )

    #     position_error = torch.linalg.vector_norm(
    #         pred_positions - matched_true,
    #         dim=-1,
    #     )

    #     return {
    #         "loss": self.loss,
    #         "position_error_slot0": position_error[:, 0].mean(),
    #         "position_error_slot1": position_error[:, 1].mean(),
    #         "mean_position_error": position_error.mean(),
    #         "swap_fraction": use_swap.float().mean(),
    #         "predicted_slot_separation": torch.linalg.vector_norm(
    #             pred_positions[:, 0] - pred_positions[:, 1],
    #             dim=-1,
    #         ).mean(),
    #         "true_slot_separation": torch.linalg.vector_norm(
    #             true_positions[:, 0] - true_positions[:, 1],
    #             dim=-1,
    #         ).mean(),
    #     }


## auxiliary layer loss

    def compute_metrics(self):
        true_scaled = self.stacked_target

        if self.model_out.dim() == true_scaled.dim() + 1:
            per_layer_preds = list(self.model_out)        # L tensors of (B, S, C)
        else:
            per_layer_preds = [self.model_out]

        layer_losses = []
        for pred_scaled in per_layer_preds:
            pairwise_cost = F.huber_loss(
                pred_scaled.unsqueeze(2),
                true_scaled.unsqueeze(1),
                delta=self.criterion.delta,
                reduction="none",
            ).mean(dim=-1)

            identity_cost = pairwise_cost[:, 0, 0] + pairwise_cost[:, 1, 1]
            swap_cost     = pairwise_cost[:, 0, 1] + pairwise_cost[:, 1, 0]

            use_swap = swap_cost < identity_cost

            layer_losses.append(
                torch.where(use_swap, swap_cost, identity_cost).mean()
            )

        self.loss = torch.stack(layer_losses).mean()

        # final_loss = layer_losses[-1]

        # if len(layer_losses) > 1:
        #     auxiliary_loss = torch.stack(layer_losses[:-1]).mean()
        # else:
        #     auxiliary_loss = final_loss.new_zeros(())

        # aux_weight = 0.2

        # self.loss = final_loss + aux_weight * auxiliary_loss

        pred_positions = self.predictions["predicted_positions"]
        true_positions = self.target_dict["positions"]

        matched_true = torch.where(
            use_swap[:, None, None],
            true_positions.flip(dims=[1]),
            true_positions,
        )

        position_error = torch.linalg.vector_norm(
            pred_positions - matched_true, dim=-1,
        )

        return {
            "loss": self.loss,
            "position_error_slot0": position_error[:, 0].mean(),
            "position_error_slot1": position_error[:, 1].mean(),
            "mean_position_error": position_error.mean(),
            "swap_fraction": use_swap.float().mean(),
            "predicted_slot_separation": torch.linalg.vector_norm(
                pred_positions[:, 0] - pred_positions[:, 1], dim=-1,
            ).mean(),
            "true_slot_separation": torch.linalg.vector_norm(
                true_positions[:, 0] - true_positions[:, 1], dim=-1,
            ).mean(),
        }


# ## direction loss

    # def compute_metrics(self):
    #     # Truth targets
    #     true_positions_scaled = (
    #         self.target_dict["positions"] - self.offset["positions"]
    #     ) / self.scale["positions"]

    #     true_directions = F.normalize(
    #         self.target_dict["directions"],
    #         p=2,
    #         dim=-1,
    #         eps=1e-8,
    #     )

    #     # Training with aux decoder outputs:
    #     # [L, B, S, 6]
    #     # Evaluation:
    #     # [B, S, 6]
    #     if self.model_out.dim() == self.stacked_target.dim() + 1:
    #         per_layer_outputs = list(self.model_out)
    #     else:
    #         per_layer_outputs = [self.model_out]

    #     layer_losses = []

    #     final_use_swap = None
    #     final_position_loss = None
    #     final_direction_loss = None

    #     direction_weight = 0.05

    #     for layer_output in per_layer_outputs:
    #         pred_positions_scaled = layer_output[..., :3]

    #         pred_directions = F.normalize(
    #             layer_output[..., 3:6],
    #             p=2,
    #             dim=-1,
    #             eps=1e-8,
    #         )

    #         # Pairwise position cost:
    #         # [B, predicted slot, truth slot]
    #         pairwise_position_cost = F.huber_loss(
    #             pred_positions_scaled.unsqueeze(2),
    #             true_positions_scaled.unsqueeze(1),
    #             delta=self.criterion.delta,
    #             reduction="none",
    #         ).mean(dim=-1)

    #         identity_cost = (
    #             pairwise_position_cost[:, 0, 0]
    #             + pairwise_position_cost[:, 1, 1]
    #         )

    #         swap_cost = (
    #             pairwise_position_cost[:, 0, 1]
    #             + pairwise_position_cost[:, 1, 0]
    #         )

    #         use_swap = swap_cost < identity_cost

    #         position_loss = torch.where(
    #             use_swap,
    #             swap_cost,
    #             identity_cost,
    #         ).mean()

    #         # Use the position-derived assignment for directions
    #         matched_true_directions = torch.where(
    #             use_swap[:, None, None],
    #             true_directions.flip(dims=[1]),
    #             true_directions,
    #         )

    #         direction_loss = (
    #             1.0
    #             - torch.sum(
    #                 pred_directions * matched_true_directions,
    #                 dim=-1,
    #             )
    #         ).mean()

    #         layer_losses.append(
    #             position_loss
    #             + direction_weight * direction_loss
    #         )

    #         final_use_swap = use_swap
    #         final_position_loss = position_loss
    #         final_direction_loss = direction_loss

    #     self.loss = torch.stack(layer_losses).mean()

    #     # Final-layer reporting in physical units
    #     pred_positions = self.predictions["predicted_positions"]
    #     true_positions = self.target_dict["positions"]

    #     matched_true_positions = torch.where(
    #         final_use_swap[:, None, None],
    #         true_positions.flip(dims=[1]),
    #         true_positions,
    #     )

    #     position_error = torch.linalg.vector_norm(
    #         pred_positions - matched_true_positions,
    #         dim=-1,
    #     )

    #     return {
    #         "loss": self.loss,
    #         "position_loss": final_position_loss,
    #         "direction_loss": final_direction_loss,
    #         "weighted_direction_loss": (
    #             direction_weight * final_direction_loss
    #         ),
    #         "position_error_slot0": position_error[:, 0].mean(),
    #         "position_error_slot1": position_error[:, 1].mean(),
    #         "mean_position_error": position_error.mean(),
    #         "swap_fraction": final_use_swap.float().mean(),
    #         "predicted_slot_separation": torch.linalg.vector_norm(
    #             pred_positions[:, 0] - pred_positions[:, 1],
    #             dim=-1,
    #         ).mean(),
    #         "true_slot_separation": torch.linalg.vector_norm(
    #             true_positions[:, 0] - true_positions[:, 1],
    #             dim=-1,
    #         ).mean(),
    #     }

    def save_state(self, suffix="", name=None):
        self.state_data["target_sizes"] = self.target_sizes
        super().save_state(suffix, name)

    def restore_state(self, weight_file):
        super().restore_state(weight_file)
        if "target_sizes" in self.state_data:
            self.target_sizes = self.state_data["target_sizes"]
