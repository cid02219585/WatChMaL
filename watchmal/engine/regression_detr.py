import torch
import torch.nn.functional as F

from collections.abc import Mapping

from watchmal.engine.reconstruction import ReconstructionEngine


class DETRRegressionEngine(ReconstructionEngine):
    """
    DETR-style engine for variable one/two-ring position reconstruction,
    with optional direction supervision.

    Expected truth
    --------------
    positions : [B, 2, 3]
        Single-ring events are padded in slot 1.

    ring_mask : [B, 2]
        [1, 0] for a single-ring event.
        [1, 1] for a two-ring event.

    directions : [B, 2, 3], optional
        Required only when direction_loss=True.

    Expected model output
    ---------------------
    {
        "pred_positions":  [B, 2, 3] or [L, B, 2, 3],
        "pred_logits":     [B, 2]    or [L, B, 2],
        "pred_directions": [B, 2, 3] or [L, B, 2, 3],  # optional
    }

    When auxiliary decoder supervision is enabled, loss is calculated
    independently for every decoder layer and averaged over layers.

    Final-layer predictions are used for reported physics metrics.
    """

    def __init__(
        self,
        target_key,
        model,
        rank,
        device,
        dump_path,
        target_scale_offset=0,
        target_scale_factor=1,
        clip_grad_norm=None,
        classification_weight=1.0,
        ring_threshold=0.5,
        direction_loss=True,
        direction_weight=0.05,
    ):
        super().__init__(
            target_key,
            model,
            rank,
            device,
            dump_path,
            clip_grad_norm=clip_grad_norm,
        )

        self.target_key = ["positions"]

        if isinstance(target_scale_offset, Mapping):
            offset = target_scale_offset.get("positions", 0)
        else:
            offset = target_scale_offset

        if isinstance(target_scale_factor, Mapping):
            scale = target_scale_factor.get("positions", 1)
        else:
            scale = target_scale_factor

        self.offset = torch.as_tensor(offset, dtype=torch.float32, device=self.device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=self.device)

        self.classification_weight = classification_weight
        self.ring_threshold = ring_threshold
        self.direction_loss = direction_loss
        self.direction_weight = direction_weight

        self.target_dict = None
        self.true_positions = None
        self.true_positions_scaled = None
        self.true_directions = None
        self.ring_mask = None

        self.model_out = None
        self.pred_positions_scaled = None
        self.pred_positions = None
        self.pred_logits = None
        self.pred_directions = None
        self.predictions = None

        # Retained for checkpoint compatibility with RegressionEngine.
        self.target_sizes = [3]

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _remove_extra_graph_dimension(x):
        """
        PyG graph-level attributes can occasionally acquire an extra singleton
        dimension depending on how they were stored.

        Normalises:
            [B, 1, 2, 3] -> [B, 2, 3]
            [B, 1, 2]    -> [B, 2]
        """
        if x.dim() >= 3 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    @staticmethod
    def _nan_like(reference):
        return reference.new_tensor(float("nan"))

    # ========================================================
    # Targets
    # ========================================================

    def process_target(self, data):
        if isinstance(data, Mapping):
            positions = data["positions"]
            ring_mask = data["ring_mask"]
            directions = data["directions"] if self.direction_loss else None
        else:
            positions = getattr(data, "positions")
            ring_mask = getattr(data, "ring_mask")
            directions = getattr(data, "directions") if self.direction_loss else None

        positions = self._remove_extra_graph_dimension(positions)
        ring_mask = self._remove_extra_graph_dimension(ring_mask)
        if self.direction_loss:
            directions = self._remove_extra_graph_dimension(directions)

        self.true_positions = positions.to(self.device).float()
        self.ring_mask = ring_mask.to(self.device).float()

        if self.direction_loss:
            self.true_directions = F.normalize(
                directions.to(self.device).float(), p=2, dim=-1, eps=1e-8
            )
        else:
            self.true_directions = None

        if self.true_positions.dim() != 3 or self.true_positions.shape[1:] != (2, 3):
            raise ValueError(
                "Expected positions with shape [B, 2, 3], "
                f"got {tuple(self.true_positions.shape)}"
            )

        if self.ring_mask.dim() != 2 or self.ring_mask.shape[1] != 2:
            raise ValueError(
                "Expected ring_mask with shape [B, 2], "
                f"got {tuple(self.ring_mask.shape)}"
            )

        if self.direction_loss and (
            self.true_directions.dim() != 3 or self.true_directions.shape[1:] != (2, 3)
        ):
            raise ValueError(
                "Expected directions with shape [B, 2, 3], "
                f"got {tuple(self.true_directions.shape)}"
            )

        self.true_positions_scaled = (self.true_positions - self.offset) / self.scale

        self.target_dict = {
            "positions": self.true_positions,
            "ring_mask": self.ring_mask,
        }
        if self.direction_loss:
            self.target_dict["directions"] = self.true_directions

        # ReconstructionEngine / training infrastructure may expect this.
        self.stacked_target = self.true_positions_scaled

    # ========================================================
    # Forward
    # ========================================================

    def forward_pass(self):
        self.model_out = self.model(self.data)

        if not isinstance(self.model_out, Mapping):
            raise TypeError(
                "DETRRegressionEngine expects the model to return a dictionary "
                "with 'pred_positions' and 'pred_logits'."
            )

        pred_positions_scaled = self.model_out["pred_positions"]
        pred_logits = self.model_out["pred_logits"]

        pred_directions = None
        if self.direction_loss:
            if "pred_directions" not in self.model_out:
                raise KeyError(
                    "direction_loss=True but model output does not contain "
                    "'pred_directions'."
                )
            pred_directions = self.model_out["pred_directions"]

        # Auxiliary training outputs have one extra leading decoder-layer axis.
        if pred_positions_scaled.dim() == self.true_positions_scaled.dim() + 1:
            final_positions_scaled = pred_positions_scaled[-1]
            final_logits = pred_logits[-1]
            final_directions = pred_directions[-1] if self.direction_loss else None
        else:
            final_positions_scaled = pred_positions_scaled
            final_logits = pred_logits
            final_directions = pred_directions if self.direction_loss else None

        self.pred_positions_scaled = final_positions_scaled
        self.pred_logits = final_logits
        self.pred_positions = self.pred_positions_scaled * self.scale + self.offset

        if self.direction_loss:
            self.pred_directions = F.normalize(
                final_directions, p=2, dim=-1, eps=1e-8
            )
        else:
            self.pred_directions = None

        self.predictions = {
            "predicted_positions": self.pred_positions,
            "predicted_ring_logits": self.pred_logits,
            "predicted_ring_probabilities": torch.sigmoid(self.pred_logits),
        }
        if self.direction_loss:
            self.predictions["predicted_directions"] = self.pred_directions

        if self.target_dict is None:
            return self.predictions
        return self.target_dict | self.predictions

    # ========================================================
    # One decoder-layer loss
    # ========================================================

    def _layer_loss(
        self,
        pred_positions_scaled,
        pred_logits,
        pred_directions=None,
        return_matching=False,
    ):
        """
        Match predictions to truth and compute loss for one decoder layer.

        Matching
        --------
        1-ring:
            choose whether query 0 or query 1 owns truth slot 0.

        2-ring:
            choose identity vs swapped assignment.

        The matching cost combines position Huber cost and classification cost.
        Direction does not affect matching; it is supervised after the assignment.
        """
        batch_size = pred_positions_scaled.shape[0]

        # [B, predicted query, truth slot]
        pairwise_position_cost = F.huber_loss(
            pred_positions_scaled.unsqueeze(2),
            self.true_positions_scaled.unsqueeze(1),
            delta=self.criterion.delta,
            reduction="none",
        ).mean(dim=-1)

        n_rings = self.ring_mask.sum(dim=1).long()
        single_event = n_rings == 1
        double_event = n_rings == 2

        if not torch.all(single_event | double_event):
            raise ValueError("Every event must contain either 1 or 2 rings.")

        matched_ring_targets = torch.zeros_like(pred_logits)
        matched_position_mask = torch.zeros_like(self.ring_mask, dtype=torch.bool)
        matched_truth_index = torch.full(
            (batch_size, 2), -1, dtype=torch.long, device=self.device
        )

        # ----------------------------------------------------
        # Single-ring events
        # ----------------------------------------------------
        if single_event.any():
            ids = torch.where(single_event)[0]

            q0_position_cost = pairwise_position_cost[ids, 0, 0]
            q1_position_cost = pairwise_position_cost[ids, 1, 0]
            logits = pred_logits[ids]

            target_if_q0 = torch.zeros_like(logits)
            target_if_q0[:, 0] = 1.0

            target_if_q1 = torch.zeros_like(logits)
            target_if_q1[:, 1] = 1.0

            q0_class_cost = F.binary_cross_entropy_with_logits(
                logits, target_if_q0, reduction="none"
            ).mean(dim=1)
            q1_class_cost = F.binary_cross_entropy_with_logits(
                logits, target_if_q1, reduction="none"
            ).mean(dim=1)

            q0_total_cost = q0_position_cost + self.classification_weight * q0_class_cost
            q1_total_cost = q1_position_cost + self.classification_weight * q1_class_cost
            use_query1 = q1_total_cost.detach() < q0_total_cost.detach()

            q0_ids = ids[~use_query1]
            q1_ids = ids[use_query1]

            if q0_ids.numel() > 0:
                matched_ring_targets[q0_ids, 0] = 1.0
                matched_position_mask[q0_ids, 0] = True
                matched_truth_index[q0_ids, 0] = 0

            if q1_ids.numel() > 0:
                matched_ring_targets[q1_ids, 1] = 1.0
                matched_position_mask[q1_ids, 1] = True
                matched_truth_index[q1_ids, 1] = 0

        # ----------------------------------------------------
        # Two-ring events
        # ----------------------------------------------------
        if double_event.any():
            ids = torch.where(double_event)[0]

            identity_cost = (
                pairwise_position_cost[ids, 0, 0] + pairwise_position_cost[ids, 1, 1]
            )
            swap_cost = (
                pairwise_position_cost[ids, 0, 1] + pairwise_position_cost[ids, 1, 0]
            )
            use_swap = swap_cost.detach() < identity_cost.detach()

            matched_ring_targets[ids] = 1.0
            matched_position_mask[ids] = True

            normal_ids = ids[~use_swap]
            swapped_ids = ids[use_swap]

            if normal_ids.numel() > 0:
                matched_truth_index[normal_ids, 0] = 0
                matched_truth_index[normal_ids, 1] = 1

            if swapped_ids.numel() > 0:
                matched_truth_index[swapped_ids, 0] = 1
                matched_truth_index[swapped_ids, 1] = 0

        # ----------------------------------------------------
        # Gather matched truth positions
        # ----------------------------------------------------
        matched_true_positions_scaled = torch.zeros_like(pred_positions_scaled)

        for query in range(2):
            valid = matched_truth_index[:, query] >= 0
            if valid.any():
                batch_ids = torch.where(valid)[0]
                truth_ids = matched_truth_index[batch_ids, query]
                matched_true_positions_scaled[batch_ids, query] = self.true_positions_scaled[
                    batch_ids, truth_ids
                ]

        per_query_position_loss = F.huber_loss(
            pred_positions_scaled,
            matched_true_positions_scaled,
            delta=self.criterion.delta,
            reduction="none",
        ).mean(dim=-1)
        position_loss = per_query_position_loss[matched_position_mask].mean()

        # Both real and no-object queries contribute to classification loss.
        classification_loss = F.binary_cross_entropy_with_logits(
            pred_logits, matched_ring_targets
        )

        # ----------------------------------------------------
        # Direction loss
        # ----------------------------------------------------
        if self.direction_loss:
            if pred_directions is None:
                raise ValueError(
                    "direction_loss=True but pred_directions was not supplied."
                )

            pred_directions = F.normalize(pred_directions, p=2, dim=-1, eps=1e-8)
            matched_true_directions = torch.zeros_like(pred_directions)

            for query in range(2):
                valid = matched_truth_index[:, query] >= 0
                if valid.any():
                    batch_ids = torch.where(valid)[0]
                    truth_ids = matched_truth_index[batch_ids, query]
                    matched_true_directions[batch_ids, query] = self.true_directions[
                        batch_ids, truth_ids
                    ]

            per_query_direction_loss = 1.0 - torch.sum(
                pred_directions * matched_true_directions, dim=-1
            )
            direction_loss = per_query_direction_loss[matched_position_mask].mean()
        else:
            direction_loss = position_loss.new_zeros(())

        total_loss = (
            position_loss
            + self.classification_weight * classification_loss
            + self.direction_weight * direction_loss
        )

        if not return_matching:
            return total_loss, position_loss, classification_loss, direction_loss

        return {
            "loss": total_loss,
            "position_loss": position_loss,
            "classification_loss": classification_loss,
            "direction_loss": direction_loss,
            "matched_ring_targets": matched_ring_targets,
            "matched_position_mask": matched_position_mask,
            "matched_truth_index": matched_truth_index,
        }

    # ========================================================
    # Metrics / loss
    # ========================================================

    def compute_metrics(self):
        all_positions = self.model_out["pred_positions"]
        all_logits = self.model_out["pred_logits"]
        all_directions = (
            self.model_out.get("pred_directions") if self.direction_loss else None
        )

        # Normalise outputs to one list entry per supervised decoder layer.
        if all_positions.dim() == self.true_positions_scaled.dim() + 1:
            position_layers = list(all_positions)
            logit_layers = list(all_logits)
            direction_layers = (
                list(all_directions)
                if self.direction_loss
                else [None] * len(position_layers)
            )
        else:
            position_layers = [all_positions]
            logit_layers = [all_logits]
            direction_layers = [all_directions] if self.direction_loss else [None]

        layer_losses = []
        layer_position_losses = []
        layer_classification_losses = []
        layer_direction_losses = []
        final_matching = None

        for layer_idx, (pred_positions_scaled, pred_logits, pred_directions) in enumerate(
            zip(position_layers, logit_layers, direction_layers)
        ):
            is_final = layer_idx == len(position_layers) - 1
            result = self._layer_loss(
                pred_positions_scaled,
                pred_logits,
                pred_directions=pred_directions,
                return_matching=is_final,
            )

            if is_final:
                final_matching = result
                layer_losses.append(result["loss"])
                layer_position_losses.append(result["position_loss"])
                layer_classification_losses.append(result["classification_loss"])
                layer_direction_losses.append(result["direction_loss"])
            else:
                layer_loss, position_loss, classification_loss, direction_loss = result
                layer_losses.append(layer_loss)
                layer_position_losses.append(position_loss)
                layer_classification_losses.append(classification_loss)
                layer_direction_losses.append(direction_loss)

        # Equal weighting over decoder layers, matching the existing aux-loss behaviour.
        self.loss = torch.stack(layer_losses).mean()

        matched_truth_index = final_matching["matched_truth_index"]
        matched_position_mask = final_matching["matched_position_mask"]
        matched_ring_targets = final_matching["matched_ring_targets"]

        # ----------------------------------------------------
        # Final-layer matched truth in physical units
        # ----------------------------------------------------
        matched_true_positions = torch.zeros_like(self.pred_positions)
        matched_true_directions = (
            torch.zeros_like(self.pred_directions) if self.direction_loss else None
        )

        for query in range(2):
            valid = matched_truth_index[:, query] >= 0
            if valid.any():
                batch_ids = torch.where(valid)[0]
                truth_ids = matched_truth_index[batch_ids, query]

                matched_true_positions[batch_ids, query] = self.true_positions[
                    batch_ids, truth_ids
                ]

                if self.direction_loss:
                    matched_true_directions[batch_ids, query] = self.true_directions[
                        batch_ids, truth_ids
                    ]

        position_error = torch.linalg.vector_norm(
            self.pred_positions - matched_true_positions, dim=-1
        )
        real_ring_errors = position_error[matched_position_mask]

        # ----------------------------------------------------
        # Event-type masks
        # ----------------------------------------------------
        true_n_rings = self.ring_mask.sum(dim=1).long()
        single_event = true_n_rings == 1
        double_event = true_n_rings == 2

        single_mask = matched_position_mask & single_event[:, None]
        double_mask = matched_position_mask & double_event[:, None]

        single_position_error = (
            position_error[single_mask].mean()
            if single_mask.any()
            else self._nan_like(self.loss)
        )
        two_ring_position_error = (
            position_error[double_mask].mean()
            if double_mask.any()
            else self._nan_like(self.loss)
        )

        # ----------------------------------------------------
        # Direction physics metrics
        # ----------------------------------------------------
        if self.direction_loss:
            pred_dir = F.normalize(self.pred_directions, p=2, dim=-1, eps=1e-8)
            true_dir = F.normalize(matched_true_directions, p=2, dim=-1, eps=1e-8)

            cos_angle = torch.clamp(torch.sum(pred_dir * true_dir, dim=-1), -1.0, 1.0)
            direction_error = torch.acos(cos_angle)
            real_direction_errors = direction_error[matched_position_mask]

            mean_direction_error = real_direction_errors.mean()
            mean_direction_error_deg = torch.rad2deg(mean_direction_error)

            if single_mask.any():
                single_direction_error = direction_error[single_mask].mean()
                single_direction_error_deg = torch.rad2deg(single_direction_error)
            else:
                single_direction_error = self._nan_like(self.loss)
                single_direction_error_deg = single_direction_error

            if double_mask.any():
                two_ring_direction_error = direction_error[double_mask].mean()
                two_ring_direction_error_deg = torch.rad2deg(two_ring_direction_error)
            else:
                two_ring_direction_error = self._nan_like(self.loss)
                two_ring_direction_error_deg = two_ring_direction_error

        # ----------------------------------------------------
        # Ring counting/classification
        # ----------------------------------------------------
        probabilities = torch.sigmoid(self.pred_logits)
        predicted_ring_mask = probabilities >= self.ring_threshold
        predicted_n_rings = predicted_ring_mask.sum(dim=1)

        ring_count_accuracy = (predicted_n_rings == true_n_rings).float().mean()
        matched_query_accuracy = (
            predicted_ring_mask == matched_ring_targets.bool()
        ).float().mean()

        metrics = {
            "loss": self.loss,
            "position_loss": layer_position_losses[-1],
            "classification_loss": layer_classification_losses[-1],
            "aux_mean_position_loss": torch.stack(layer_position_losses).mean(),
            "aux_mean_classification_loss": torch.stack(
                layer_classification_losses
            ).mean(),
            "mean_position_error": real_ring_errors.mean(),
            "single_ring_position_error": single_position_error,
            "two_ring_position_error": two_ring_position_error,
            "ring_count_accuracy": ring_count_accuracy,
            "query_classification_accuracy": matched_query_accuracy,
            "mean_predicted_n_rings": predicted_n_rings.float().mean(),
            "mean_true_n_rings": true_n_rings.float().mean(),
            "ring_probability_query0": probabilities[:, 0].mean(),
            "ring_probability_query1": probabilities[:, 1].mean(),
            "n_decoder_loss_layers": self.loss.new_tensor(float(len(layer_losses))),
        }

        if self.direction_loss:
            metrics.update(
                {
                    "direction_loss": layer_direction_losses[-1],
                    "weighted_direction_loss": (
                        self.direction_weight * layer_direction_losses[-1]
                    ),
                    "aux_mean_direction_loss": torch.stack(
                        layer_direction_losses
                    ).mean(),
                    "mean_direction_error": mean_direction_error,
                    "mean_direction_error_deg": mean_direction_error_deg,
                    "single_ring_direction_error": single_direction_error,
                    "single_ring_direction_error_deg": single_direction_error_deg,
                    "two_ring_direction_error": two_ring_direction_error,
                    "two_ring_direction_error_deg": two_ring_direction_error_deg,
                }
            )

        return metrics

    # ========================================================
    # Checkpoint state
    # ========================================================

    def save_state(self, suffix="", name=None):
        self.state_data["target_sizes"] = self.target_sizes
        super().save_state(suffix, name)

    def restore_state(self, weight_file):
        super().restore_state(weight_file)
        if "target_sizes" in self.state_data:
            self.target_sizes = self.state_data["target_sizes"]

# import torch
# import torch.nn.functional as F

# from collections.abc import Mapping

# from watchmal.engine.reconstruction import ReconstructionEngine


# class DETRRegressionEngine(ReconstructionEngine):
#     """
#     DETR-style engine for variable one/two-ring position reconstruction.

#     Expected truth
#     --------------
#     positions : [B, 2, 3]
#         Single-ring events are padded in slot 1.

#     ring_mask : [B, 2]
#         [1, 0] for a single-ring event.
#         [1, 1] for a two-ring event.

#     Expected model output
#     ---------------------
#     {
#         "pred_positions": [B, 2, 3] or [L, B, 2, 3],
#         "pred_logits":    [B, 2]    or [L, B, 2]
#     }

#     When auxiliary decoder supervision is enabled, loss is calculated
#     independently for every decoder layer and averaged over layers.

#     Final-layer predictions are used for reported physics metrics.
#     """

#     def __init__(
#         self,
#         target_key,
#         model,
#         rank,
#         device,
#         dump_path,
#         target_scale_offset=0,
#         target_scale_factor=1,
#         clip_grad_norm=None,
#         classification_weight=1.0,
#         ring_threshold=0.5,
#     ):
#         super().__init__(
#             target_key,
#             model,
#             rank,
#             device,
#             dump_path,
#             clip_grad_norm=clip_grad_norm,
#         )

#         self.target_key = ["positions"]

#         if isinstance(
#             target_scale_offset,
#             Mapping,
#         ):
#             offset = (
#                 target_scale_offset.get(
#                     "positions",
#                     0,
#                 )
#             )
#         else:
#             offset = target_scale_offset

#         if isinstance(
#             target_scale_factor,
#             Mapping,
#         ):
#             scale = (
#                 target_scale_factor.get(
#                     "positions",
#                     1,
#                 )
#             )
#         else:
#             scale = target_scale_factor

#         self.offset = torch.as_tensor(
#             offset,
#             dtype=torch.float32,
#             device=self.device,
#         )

#         self.scale = torch.as_tensor(
#             scale,
#             dtype=torch.float32,
#             device=self.device,
#         )

#         self.classification_weight = (
#             classification_weight
#         )

#         self.ring_threshold = (
#             ring_threshold
#         )

#         self.target_dict = None

#         self.true_positions = None
#         self.true_positions_scaled = None
#         self.ring_mask = None

#         self.model_out = None

#         self.pred_positions_scaled = None
#         self.pred_positions = None
#         self.pred_logits = None

#         self.predictions = None

#         # Retained for checkpoint compatibility with RegressionEngine.
#         self.target_sizes = [3]

#     # ========================================================
#     # Helpers
#     # ========================================================

#     @staticmethod
#     def _remove_extra_graph_dimension(x):
#         """
#         PyG graph-level attributes can occasionally acquire an extra
#         singleton dimension depending on how they were stored.

#         Normalises:
#             [B, 1, 2, 3] -> [B, 2, 3]
#             [B, 1, 2]    -> [B, 2]
#         """
#         if (
#             x.dim() >= 3
#             and x.shape[1] == 1
#         ):
#             return x.squeeze(1)

#         return x

#     # ========================================================
#     # Targets
#     # ========================================================

#     def process_target(
#         self,
#         data,
#     ):
#         if isinstance(data, Mapping):
#             positions = data[
#                 "positions"
#             ]

#             ring_mask = data[
#                 "ring_mask"
#             ]

#         else:
#             positions = getattr(
#                 data,
#                 "positions",
#             )

#             ring_mask = getattr(
#                 data,
#                 "ring_mask",
#             )

#         positions = (
#             self._remove_extra_graph_dimension(
#                 positions
#             )
#         )

#         ring_mask = (
#             self._remove_extra_graph_dimension(
#                 ring_mask
#             )
#         )

#         self.true_positions = (
#             positions
#             .to(self.device)
#             .float()
#         )

#         self.ring_mask = (
#             ring_mask
#             .to(self.device)
#             .float()
#         )

#         if (
#             self.true_positions.dim() != 3
#             or self.true_positions.shape[1:] != (2, 3)
#         ):
#             raise ValueError(
#                 "Expected positions with shape [B, 2, 3], "
#                 f"got {tuple(self.true_positions.shape)}"
#             )

#         if (
#             self.ring_mask.dim() != 2
#             or self.ring_mask.shape[1] != 2
#         ):
#             raise ValueError(
#                 "Expected ring_mask with shape [B, 2], "
#                 f"got {tuple(self.ring_mask.shape)}"
#             )

#         self.true_positions_scaled = (
#             self.true_positions
#             - self.offset
#         ) / self.scale

#         self.target_dict = {
#             "positions":
#                 self.true_positions,

#             "ring_mask":
#                 self.ring_mask,
#         }

#         # ReconstructionEngine / training infrastructure may expect this.
#         self.stacked_target = (
#             self.true_positions_scaled
#         )

#     # ========================================================
#     # Forward
#     # ========================================================

#     def forward_pass(
#         self,
#     ):
#         self.model_out = self.model(
#             self.data
#         )

#         if not isinstance(
#             self.model_out,
#             Mapping,
#         ):
#             raise TypeError(
#                 "DETRRegressionEngine expects the model to return "
#                 "a dictionary with 'pred_positions' and 'pred_logits'."
#             )

#         pred_positions_scaled = (
#             self.model_out[
#                 "pred_positions"
#             ]
#         )

#         pred_logits = (
#             self.model_out[
#                 "pred_logits"
#             ]
#         )

#         # ----------------------------------------------------
#         # Auxiliary training outputs:
#         #
#         # [L, B, 2, 3] -> use final layer for normal predictions.
#         # [L, B, 2]    -> final layer logits.
#         # ----------------------------------------------------

#         if (
#             pred_positions_scaled.dim()
#             == self.true_positions_scaled.dim() + 1
#         ):
#             final_positions_scaled = (
#                 pred_positions_scaled[-1]
#             )
#             final_logits = (
#                 pred_logits[-1]
#             )
#         else:
#             final_positions_scaled = (
#                 pred_positions_scaled
#             )
#             final_logits = pred_logits

#         self.pred_positions_scaled = (
#             final_positions_scaled
#         )

#         self.pred_logits = (
#             final_logits
#         )

#         self.pred_positions = (
#             self.pred_positions_scaled
#             * self.scale
#             + self.offset
#         )

#         self.predictions = {
#             "predicted_positions":
#                 self.pred_positions,

#             "predicted_ring_logits":
#                 self.pred_logits,

#             "predicted_ring_probabilities":
#                 torch.sigmoid(
#                     self.pred_logits
#                 ),
#         }

#         if self.target_dict is None:
#             return self.predictions

#         return (
#             self.target_dict
#             | self.predictions
#         )

#     # ========================================================
#     # One decoder-layer loss
#     # ========================================================

#     def _layer_loss(
#         self,
#         pred_positions_scaled,
#         pred_logits,
#         return_matching=False,
#     ):
#         """
#         Match predictions to truth and compute loss for one decoder layer.

#         Matching
#         --------
#         1-ring:
#             choose whether query 0 or query 1 owns truth slot 0.

#         2-ring:
#             choose identity vs swapped assignment.

#         The matching cost combines:
#             position Huber cost
#             + classification_weight * BCE classification cost.

#         For a two-ring event, both classification targets are 1 for
#         either permutation, so the assignment itself depends only on
#         position cost.
#         """

#         B = (
#             pred_positions_scaled
#             .shape[0]
#         )

#         # ----------------------------------------------------
#         # Pairwise position cost
#         #
#         # [B, predicted query, truth slot]
#         # ----------------------------------------------------

#         pairwise_position_cost = (
#             F.huber_loss(
#                 pred_positions_scaled.unsqueeze(2),
#                 self.true_positions_scaled.unsqueeze(1),
#                 delta=self.criterion.delta,
#                 reduction="none",
#             )
#             .mean(dim=-1)
#         )

#         n_rings = (
#             self.ring_mask
#             .sum(dim=1)
#             .long()
#         )

#         single_event = (
#             n_rings == 1
#         )

#         double_event = (
#             n_rings == 2
#         )

#         if not torch.all(
#             single_event | double_event
#         ):
#             raise ValueError(
#                 "Every event must contain either 1 or 2 rings."
#             )

#         # ----------------------------------------------------
#         # Classification target after matching.
#         #
#         # Start with all queries = no object.
#         # ----------------------------------------------------

#         matched_ring_targets = (
#             torch.zeros_like(
#                 pred_logits
#             )
#         )

#         # Which predicted queries get a position loss?
#         matched_position_mask = (
#             torch.zeros_like(
#                 self.ring_mask,
#                 dtype=torch.bool,
#             )
#         )

#         # Stores truth-slot index assigned to each query.
#         #
#         # -1 means no real ring matched to that query.
#         matched_truth_index = torch.full(
#             (
#                 B,
#                 2,
#             ),
#             -1,
#             dtype=torch.long,
#             device=self.device,
#         )

#         # ====================================================
#         # Single-ring events
#         # ====================================================

#         if single_event.any():
#             ids = torch.where(
#                 single_event
#             )[0]

#             # Cost if query 0 owns the real ring.
#             q0_position_cost = (
#                 pairwise_position_cost[
#                     ids,
#                     0,
#                     0,
#                 ]
#             )

#             # Cost if query 1 owns the real ring.
#             q1_position_cost = (
#                 pairwise_position_cost[
#                     ids,
#                     1,
#                     0,
#                 ]
#             )

#             logits = pred_logits[
#                 ids
#             ]

#             target_if_q0 = torch.zeros_like(
#                 logits
#             )
#             target_if_q0[:, 0] = 1.0

#             target_if_q1 = torch.zeros_like(
#                 logits
#             )
#             target_if_q1[:, 1] = 1.0

#             q0_class_cost = (
#                 F.binary_cross_entropy_with_logits(
#                     logits,
#                     target_if_q0,
#                     reduction="none",
#                 )
#                 .mean(dim=1)
#             )

#             q1_class_cost = (
#                 F.binary_cross_entropy_with_logits(
#                     logits,
#                     target_if_q1,
#                     reduction="none",
#                 )
#                 .mean(dim=1)
#             )

#             q0_total_cost = (
#                 q0_position_cost
#                 + self.classification_weight
#                 * q0_class_cost
#             )

#             q1_total_cost = (
#                 q1_position_cost
#                 + self.classification_weight
#                 * q1_class_cost
#             )

#             use_query1 = (
#                 q1_total_cost.detach()
#                 < q0_total_cost.detach()
#             )

#             q0_ids = ids[
#                 ~use_query1
#             ]

#             q1_ids = ids[
#                 use_query1
#             ]

#             if q0_ids.numel() > 0:
#                 matched_ring_targets[
#                     q0_ids,
#                     0,
#                 ] = 1.0

#                 matched_position_mask[
#                     q0_ids,
#                     0,
#                 ] = True

#                 matched_truth_index[
#                     q0_ids,
#                     0,
#                 ] = 0

#             if q1_ids.numel() > 0:
#                 matched_ring_targets[
#                     q1_ids,
#                     1,
#                 ] = 1.0

#                 matched_position_mask[
#                     q1_ids,
#                     1,
#                 ] = True

#                 matched_truth_index[
#                     q1_ids,
#                     1,
#                 ] = 0

#         # ====================================================
#         # Two-ring events
#         # ====================================================

#         if double_event.any():
#             ids = torch.where(
#                 double_event
#             )[0]

#             identity_cost = (
#                 pairwise_position_cost[
#                     ids,
#                     0,
#                     0,
#                 ]
#                 +
#                 pairwise_position_cost[
#                     ids,
#                     1,
#                     1,
#                 ]
#             )

#             swap_cost = (
#                 pairwise_position_cost[
#                     ids,
#                     0,
#                     1,
#                 ]
#                 +
#                 pairwise_position_cost[
#                     ids,
#                     1,
#                     0,
#                 ]
#             )

#             use_swap = (
#                 swap_cost.detach()
#                 < identity_cost.detach()
#             )

#             matched_ring_targets[
#                 ids
#             ] = 1.0

#             matched_position_mask[
#                 ids
#             ] = True

#             normal_ids = ids[
#                 ~use_swap
#             ]

#             swapped_ids = ids[
#                 use_swap
#             ]

#             if normal_ids.numel() > 0:
#                 matched_truth_index[
#                     normal_ids,
#                     0,
#                 ] = 0

#                 matched_truth_index[
#                     normal_ids,
#                     1,
#                 ] = 1

#             if swapped_ids.numel() > 0:
#                 matched_truth_index[
#                     swapped_ids,
#                     0,
#                 ] = 1

#                 matched_truth_index[
#                     swapped_ids,
#                     1,
#                 ] = 0

#         # ====================================================
#         # Gather matched true positions
#         # ====================================================

#         matched_true_positions_scaled = (
#             torch.zeros_like(
#                 pred_positions_scaled
#             )
#         )

#         for query in range(2):
#             valid = (
#                 matched_truth_index[
#                     :,
#                     query,
#                 ] >= 0
#             )

#             if valid.any():
#                 batch_ids = torch.where(
#                     valid
#                 )[0]

#                 truth_ids = (
#                     matched_truth_index[
#                         batch_ids,
#                         query,
#                     ]
#                 )

#                 matched_true_positions_scaled[
#                     batch_ids,
#                     query,
#                 ] = (
#                     self.true_positions_scaled[
#                         batch_ids,
#                         truth_ids,
#                     ]
#                 )

#         # ====================================================
#         # Position loss
#         # ====================================================

#         per_query_position_loss = (
#             F.huber_loss(
#                 pred_positions_scaled,
#                 matched_true_positions_scaled,
#                 delta=self.criterion.delta,
#                 reduction="none",
#             )
#             .mean(dim=-1)
#         )

#         position_loss = (
#             per_query_position_loss[
#                 matched_position_mask
#             ]
#             .mean()
#         )

#         # ====================================================
#         # Classification loss
#         #
#         # Both real and no-object queries contribute.
#         # ====================================================

#         classification_loss = (
#             F.binary_cross_entropy_with_logits(
#                 pred_logits,
#                 matched_ring_targets,
#             )
#         )

#         total_loss = (
#             position_loss
#             + self.classification_weight
#             * classification_loss
#         )

#         if not return_matching:
#             return (
#                 total_loss,
#                 position_loss,
#                 classification_loss,
#             )

#         return {
#             "loss":
#                 total_loss,

#             "position_loss":
#                 position_loss,

#             "classification_loss":
#                 classification_loss,

#             "matched_ring_targets":
#                 matched_ring_targets,

#             "matched_position_mask":
#                 matched_position_mask,

#             "matched_truth_index":
#                 matched_truth_index,
#         }

#     # ========================================================
#     # Metrics / loss
#     # ========================================================

#     def compute_metrics(
#         self,
#     ):
#         all_positions = (
#             self.model_out[
#                 "pred_positions"
#             ]
#         )

#         all_logits = (
#             self.model_out[
#                 "pred_logits"
#             ]
#         )

#         # ----------------------------------------------------
#         # Normalise to a list of layers.
#         # ----------------------------------------------------

#         if (
#             all_positions.dim()
#             == self.true_positions_scaled.dim() + 1
#         ):
#             position_layers = list(
#                 all_positions
#             )

#             logit_layers = list(
#                 all_logits
#             )
#         else:
#             position_layers = [
#                 all_positions
#             ]

#             logit_layers = [
#                 all_logits
#             ]

#         layer_losses = []
#         layer_position_losses = []
#         layer_classification_losses = []

#         final_matching = None

#         for layer_idx, (
#             pred_positions_scaled,
#             pred_logits,
#         ) in enumerate(
#             zip(
#                 position_layers,
#                 logit_layers,
#             )
#         ):
#             is_final = (
#                 layer_idx
#                 == len(position_layers) - 1
#             )

#             result = self._layer_loss(
#                 pred_positions_scaled,
#                 pred_logits,
#                 return_matching=is_final,
#             )

#             if is_final:
#                 final_matching = result

#                 layer_losses.append(
#                     result["loss"]
#                 )

#                 layer_position_losses.append(
#                     result[
#                         "position_loss"
#                     ]
#                 )

#                 layer_classification_losses.append(
#                     result[
#                         "classification_loss"
#                     ]
#                 )
#             else:
#                 (
#                     layer_loss,
#                     position_loss,
#                     classification_loss,
#                 ) = result

#                 layer_losses.append(
#                     layer_loss
#                 )

#                 layer_position_losses.append(
#                     position_loss
#                 )

#                 layer_classification_losses.append(
#                     classification_loss
#                 )

#         # Equal weighting over decoder layers, matching the previous
#         # auxiliary-loss behaviour.
#         self.loss = torch.stack(
#             layer_losses
#         ).mean()

#         # ----------------------------------------------------
#         # Final-layer matching in physical units
#         # ----------------------------------------------------

#         matched_truth_index = (
#             final_matching[
#                 "matched_truth_index"
#             ]
#         )

#         matched_position_mask = (
#             final_matching[
#                 "matched_position_mask"
#             ]
#         )

#         matched_ring_targets = (
#             final_matching[
#                 "matched_ring_targets"
#             ]
#         )

#         matched_true_positions = (
#             torch.zeros_like(
#                 self.pred_positions
#             )
#         )

#         for query in range(2):
#             valid = (
#                 matched_truth_index[
#                     :,
#                     query,
#                 ] >= 0
#             )

#             if valid.any():
#                 batch_ids = torch.where(
#                     valid
#                 )[0]

#                 truth_ids = (
#                     matched_truth_index[
#                         batch_ids,
#                         query,
#                     ]
#                 )

#                 matched_true_positions[
#                     batch_ids,
#                     query,
#                 ] = (
#                     self.true_positions[
#                         batch_ids,
#                         truth_ids,
#                     ]
#                 )

#         position_error = (
#             torch.linalg.vector_norm(
#                 self.pred_positions
#                 - matched_true_positions,
#                 dim=-1,
#             )
#         )

#         real_ring_errors = (
#             position_error[
#                 matched_position_mask
#             ]
#         )

#         # ----------------------------------------------------
#         # Single / double event metrics
#         # ----------------------------------------------------

#         true_n_rings = (
#             self.ring_mask
#             .sum(dim=1)
#             .long()
#         )

#         single_event = (
#             true_n_rings == 1
#         )

#         double_event = (
#             true_n_rings == 2
#         )

#         single_mask = (
#             matched_position_mask
#             & single_event[:, None]
#         )

#         double_mask = (
#             matched_position_mask
#             & double_event[:, None]
#         )

#         if single_mask.any():
#             single_position_error = (
#                 position_error[
#                     single_mask
#                 ]
#                 .mean()
#             )
#         else:
#             single_position_error = (
#                 self.loss.new_tensor(
#                     float("nan")
#                 )
#             )

#         if double_mask.any():
#             two_ring_position_error = (
#                 position_error[
#                     double_mask
#                 ]
#                 .mean()
#             )
#         else:
#             two_ring_position_error = (
#                 self.loss.new_tensor(
#                     float("nan")
#                 )
#             )

#         # ----------------------------------------------------
#         # Ring counting
#         # ----------------------------------------------------

#         probabilities = torch.sigmoid(
#             self.pred_logits
#         )

#         predicted_ring_mask = (
#             probabilities
#             >= self.ring_threshold
#         )

#         predicted_n_rings = (
#             predicted_ring_mask
#             .sum(dim=1)
#         )

#         ring_count_accuracy = (
#             predicted_n_rings
#             == true_n_rings
#         ).float().mean()

#         matched_query_accuracy = (
#             predicted_ring_mask
#             == matched_ring_targets.bool()
#         ).float().mean()

#         metrics = {
#             "loss":
#                 self.loss,

#             # Final-layer components
#             "position_loss":
#                 layer_position_losses[-1],

#             "classification_loss":
#                 layer_classification_losses[-1],

#             # Mean auxiliary components across all layers
#             "aux_mean_position_loss":
#                 torch.stack(
#                     layer_position_losses
#                 ).mean(),

#             "aux_mean_classification_loss":
#                 torch.stack(
#                     layer_classification_losses
#                 ).mean(),

#             # Physics metrics
#             "mean_position_error":
#                 real_ring_errors.mean(),

#             "single_ring_position_error":
#                 single_position_error,

#             "two_ring_position_error":
#                 two_ring_position_error,

#             # Ring classification/counting
#             "ring_count_accuracy":
#                 ring_count_accuracy,

#             "query_classification_accuracy":
#                 matched_query_accuracy,

#             "mean_predicted_n_rings":
#                 predicted_n_rings.float().mean(),

#             "mean_true_n_rings":
#                 true_n_rings.float().mean(),

#             "ring_probability_query0":
#                 probabilities[:, 0].mean(),

#             "ring_probability_query1":
#                 probabilities[:, 1].mean(),

#             "n_decoder_loss_layers":
#                 self.loss.new_tensor(
#                     float(
#                         len(layer_losses)
#                     )
#                 ),
#         }

#         return metrics

#     # ========================================================
#     # Checkpoint state
#     # ========================================================

#     def save_state(
#         self,
#         suffix="",
#         name=None,
#     ):
#         self.state_data[
#             "target_sizes"
#         ] = self.target_sizes

#         super().save_state(
#             suffix,
#             name,
#         )

#     def restore_state(
#         self,
#         weight_file,
#     ):
#         super().restore_state(
#             weight_file
#         )

#         if (
#             "target_sizes"
#             in self.state_data
#         ):
#             self.target_sizes = (
#                 self.state_data[
#                     "target_sizes"
#                 ]
#             )