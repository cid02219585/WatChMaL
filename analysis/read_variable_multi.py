import glob
import numpy as np
import matplotlib.pyplot as plt

from analysis.read_multiring import (
    WatChMaLOutput as _BaseWatChMaLOutput,
    FiTQunOutput,
)


class WatChMaLOutput(_BaseWatChMaLOutput):
    """WatChMaL output reader for variable ring multiplicity."""

    def __init__(self, directory, indices=None, ring=None):
        super().__init__(
            directory=directory,
            ring=None,
            indices=indices
        )

# simple 
    def get_outputs(self, name):
        outputs = np.load(
            self.directory + "/outputs/" + name + ".npy"
        )
        output_indices = np.load(
            self.directory + "/outputs/indices.npy"
        )

        if self.indices is None:
            return outputs[output_indices.argsort()]

        intersection = np.intersect1d(
            self.indices,
            output_indices,
            return_indices=True
        )

        if len(intersection[0]) != len(self.indices):
            raise ValueError(
                f"Requested {len(self.indices)} events but only "
                f"found {len(intersection[0])} in outputs"
            )

        sorted_outputs = np.empty(
            self.indices.shape + outputs.shape[1:],
            dtype=outputs.dtype
        )

        sorted_outputs[intersection[1]] = outputs[intersection[2]]

        return sorted_outputs

    def read_training_log_from_csv(self, directory):

        train_files = glob.glob(
            directory + "/outputs/log_train*.csv"
        )

        self._log_train = np.array([
            np.genfromtxt(
                f,
                delimiter=',',
                names=True,
                dtype=None
            )
            for f in train_files
        ])

        self._log_val = np.genfromtxt(
            directory + "/outputs/log_val.csv",
            delimiter=',',
            names=True,
            dtype=None
        )

        train_iteration = self._log_train['iteration'][0]
        train_epoch = self._log_train['epoch'][0]

        if np.max(train_epoch) == 0:
            it_per_epoch = np.max(train_iteration)
        else:
            it_per_epoch = (
                np.min(
                    train_iteration[train_epoch == 1]
                ) - 1
            )

        self._train_log_epoch = (
            train_iteration / it_per_epoch
        )

        self._val_log_epoch = (
            self._log_val['iteration'] / it_per_epoch
        )

        # Joint loss
        self._train_log_loss = np.mean(
            self._log_train['loss'],
            axis=0
        )
        self._val_log_loss = self._log_val['loss']

        # Regression loss
        self._train_log_position_loss = np.mean(
            self._log_train['position_loss'],
            axis=0
        )
        self._val_log_position_loss = \
            self._log_val['position_loss']

        # Classification loss
        self._train_log_classification_loss = np.mean(
            self._log_train['classification_loss'],
            axis=0
        )
        self._val_log_classification_loss = \
            self._log_val['classification_loss']

        self._val_log_best = \
            self._log_val['saved_best']

        return (
            self._train_log_epoch,
            self._train_log_loss,
            self._val_log_epoch,
            self._val_log_loss,
            self._val_log_best,
        )

    @property
    def train_log_position_loss(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_position_loss

    @property
    def val_log_position_loss(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_position_loss

    @property
    def train_log_classification_loss(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._train_log_classification_loss

    @property
    def val_log_classification_loss(self):
        if self._training_log is None:
            self._training_log = self.read_training_log()
        return self._val_log_classification_loss

    def plot_loss_components(
        self,
        fig_size=(12, 5),
        legend='best'
    ):

        fig, axes = plt.subplots(
            1, 2,
            figsize=fig_size
        )

        # Regression
        axes[0].plot(
            self.train_log_epoch,
            self.train_log_position_loss,
            lw=2,
            alpha=0.3,
            label="Train"
        )

        axes[0].plot(
            self.val_log_epoch,
            self.val_log_position_loss,
            lw=2,
            label="Validation"
        )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Position loss")
        axes[0].set_title("Regression")
        axes[0].legend(loc=legend)

        # Classification
        axes[1].plot(
            self.train_log_epoch,
            self.train_log_classification_loss,
            lw=2,
            alpha=0.3,
            label="Train"
        )

        axes[1].plot(
            self.val_log_epoch,
            self.val_log_classification_loss,
            lw=2,
            label="Validation"
        )

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Classification loss")
        axes[1].set_title("Classification")
        axes[1].legend(loc=legend)

        fig.tight_layout()

        return fig, axes