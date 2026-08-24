import numpy as np

from analysis.regression_multiring import (
    RegressionRun,
    PositionPrediction,
    DirectionPrediction,
    MomentumPrediction,
    plot_histograms,
    plot_resolution_profile,
    plot_resolution_profile_2d,
    plot_bias_profile,
    tabulate_statistics,
)
from analysis.read_variable_multi import WatChMaLOutput

def match_variable_slots(pred_positions, true_positions, ring_mask):
    """Match two DETR query slots to 1- or 2-ring truth."""
    pred_positions = np.asarray(pred_positions)
    true_positions = np.asarray(true_positions)
    ring_mask = np.asarray(ring_mask, dtype=bool)

    if pred_positions.shape[1:] != (2, 3):
        raise ValueError(f"pred_positions must be [N,2,3], got {pred_positions.shape}")
    if true_positions.shape != pred_positions.shape:
        raise ValueError(f"true_positions must match predictions, got {true_positions.shape}")
    if ring_mask.shape != pred_positions.shape[:2]:
        raise ValueError(f"ring_mask must be [N,2], got {ring_mask.shape}")

    cost = ((pred_positions[:, :, None, :] - true_positions[:, None, :, :]) ** 2).mean(axis=-1)
    n_events = len(pred_positions)
    matched_query = np.full((n_events, 2), -1, dtype=np.int64)

    n_rings = ring_mask.sum(axis=1)
    one_ring = n_rings == 1
    two_ring = n_rings == 2

    # Single ring: either prediction query may own the only real truth ring.
    if np.any(one_ring):
        matched_query[one_ring, 0] = np.argmin(cost[one_ring, :, 0], axis=1)

    # Two rings: compare identity and swap assignments.
    if np.any(two_ring):
        c = cost[two_ring]
        identity = c[:, 0, 0] + c[:, 1, 1]
        swap = c[:, 0, 1] + c[:, 1, 0] < identity
        q = np.empty((np.sum(two_ring), 2), dtype=np.int64)
        q[:, 0] = np.where(swap, 1, 0)
        q[:, 1] = np.where(swap, 0, 1)
        matched_query[two_ring] = q

    if np.any(n_rings == 0):
        raise ValueError("Found an event with zero real rings")
    if np.any(n_rings > 2):
        raise ValueError("Only 1- and 2-ring events are supported")

    matched_predictions = np.full(pred_positions.shape, np.nan, dtype=float)
    event_idx, truth_slot = np.where(ring_mask)
    query_idx = matched_query[event_idx, truth_slot]
    matched_predictions[event_idx, truth_slot] = pred_positions[event_idx, query_idx]

    return matched_query, matched_predictions


class WatChMaLRegression(RegressionRun, WatChMaLOutput):
    predictions_name = None

    def __init__(self, directory, run_label, indices=None, selection=None, **plot_args):
        RegressionRun.__init__(self, run_label=run_label, selection=selection, **plot_args)
        WatChMaLOutput.__init__(self, directory=directory, indices=indices)
        self._predictions = None

    @property
    def predictions(self):
        if self._predictions is None:
            self._predictions = self.get_outputs(
                "predicted_" + self.predictions_name,
            )
        return self._predictions


class WatChMaLPositionRegression(WatChMaLRegression, PositionPrediction):
    """Position analysis for the 1/2-ring DETR model.

    true_positions: [N,2,3], padded for single-ring events
    ring_mask:      [N,2], e.g. [1,0] or [1,1]

    With ring=None, particle-level arrays contain only real truth rings, so a
    single-ring event contributes one particle and a two-ring event contributes two.
    """

    predictions_name = "positions"

    def __init__(self, directory, run_label, true_positions, ring_mask,
                 true_directions=None, indices=None, selection=None,
                 ring_threshold=0.5, ring=None, **plot_args):
        WatChMaLRegression.__init__(
            self, directory=directory, run_label=run_label,
            indices=indices, selection=selection, **plot_args
        )

        self.true_positions_by_event = np.asarray(true_positions)
        self.ring_mask = np.asarray(ring_mask, dtype=bool)
        self.ring_threshold = ring_threshold

        if self.true_positions_by_event.shape[1:] != (2, 3):
            raise ValueError(f"true_positions must be [N,2,3], got {self.true_positions_by_event.shape}")
        if self.ring_mask.shape != self.true_positions_by_event.shape[:2]:
            raise ValueError(f"ring_mask must be [N,2], got {self.ring_mask.shape}")

        self.raw_position_predictions = self.get_outputs(
            "predicted_positions"
        )
        self.ring_probabilities = np.asarray(self.get_outputs(
            "predicted_ring_probabilities"
        ))
        
        (
            self.matched_query,
            self.matched_position_predictions,
        ) = match_variable_slots(
            self.raw_position_predictions,
            self.true_positions_by_event,
            self.ring_mask,
        )
        
        self.true_n_rings = self.ring_mask.sum(axis=1)
        
        self.particle_event_index = np.where(self.ring_mask)[0]

        self.single_ring_events = self.true_n_rings == 1
        self.two_ring_events = self.true_n_rings == 2
        
        # for overall - not by each
        self._position_prediction = (
            self.matched_position_predictions[self.ring_mask]
        )
        
        truth_for_metric = self.true_positions_by_event[self.ring_mask]
        directions_for_metric = (None if true_directions is None else np.asarray(true_directions)[self.ring_mask])
        
        PositionPrediction.__init__(self, true_positions=truth_for_metric, true_directions=directions_for_metric)
        
        ### Regression 
        
        # [N, 2, 3]
        self.position_residuals_by_event_slot = (
            self.matched_position_predictions
            - self.true_positions_by_event
        )

        # [N, 2]
        self.x_residuals_by_event_slot = (
            self.position_residuals_by_event_slot[:, :, 0]
        )

        self.y_residuals_by_event_slot = (
            self.position_residuals_by_event_slot[:, :, 1]
        )

        self.z_residuals_by_event_slot = (
            self.position_residuals_by_event_slot[:, :, 2]
        )

        self.position_3d_errors_by_event_slot = np.linalg.norm(
            self.position_residuals_by_event_slot,
            axis=-1,
        )

        self.mean_position_3d_error_per_event = np.nanmean(
            self.position_3d_errors_by_event_slot,
            axis=1,
        )
        
#         self.single_ring_mean_position_3d_error_per_event = self.mean_position_3d_error_per_event[self.single_ring_events]
#         self.two_ring_mean_position_3d_error_per_event = self.mean_position_3d_error_per_event[self.two_ring_events]

        self.single_ring_mean_position_3d_error_per_event = np.where(
            self.single_ring_events,
            self.mean_position_3d_error_per_event,
            np.nan,
        )

        self.two_ring_mean_position_3d_error_per_event = np.where(
            self.two_ring_events,
            self.mean_position_3d_error_per_event,
            np.nan,
        )
        
        ### Classification
        
        self.predicted_ring_mask = (
            self.ring_probabilities >= self.ring_threshold
        )

        self.predicted_n_rings = self.predicted_ring_mask.sum(axis=1)

        self.ring_count_correct = (
            self.predicted_n_rings == self.true_n_rings
        )
        
        self.single_ring_ring_count_correct = np.where(
            self.single_ring_events,
            self.ring_count_correct,
            np.nan
        )
        
        self.two_ring_ring_count_correct = np.where(
            self.two_ring_events,
            self.ring_count_correct,
            np.nan
        )
        
        ### Combined 
        self.mean_position_3d_error_correct_classification = np.where(
            self.ring_count_correct,
            self.mean_position_3d_error_per_event,
            np.nan,
        )
        
        self.single_ring_mean_position_3d_error_correct_classification = np.where(
            self.ring_count_correct,
            self.single_ring_mean_position_3d_error_per_event,
            np.nan,
        )
        
        self.two_ring_mean_position_3d_error_correct_classification = np.where(
            self.ring_count_correct,
            self.two_ring_mean_position_3d_error_per_event,
            np.nan,
        )


    @property
    def position_prediction(self):
        return self._position_prediction # as defined above this is already matched
    
#     def classification_summary(self):
#         return {
#             "ring_count_accuracy": self.ring_count_accuracy,
#             "query_classification_accuracy": self.query_classification_accuracy,
#             "mean_predicted_n_rings": self.predicted_n_rings.mean(),
#             "mean_true_n_rings": self.true_n_rings.mean(),
#             "mean_query0_ring_probability": self.ring_probabilities[:, 0].mean(),
#             "mean_query1_ring_probability": self.ring_probabilities[:, 1].mean(),
#             "single_ring_count_accuracy": self.ring_count_correct[self.single_ring_events].mean(),
#             "two_ring_count_accuracy": self.ring_count_correct[self.two_ring_events].mean(),
#         }
    
    
