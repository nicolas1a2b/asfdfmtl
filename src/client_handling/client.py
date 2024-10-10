import os
from collections import OrderedDict
import warnings
import torch
import client_handling.hca as hca
from client_handling.seed import set_seed

"""
Class that describes a federation client. Clients are instantieated via the client manager.
"""


class Client:
    def __init__(
        self,
        c_id,
        title,
        tasktype,
        model,
        seed,
        current_and_last_checkpoint=None,
    ):
        """Initializes a federation client.

        Args:
            c_id (str): The id of a client.
            title (str): The title of a client.
            tasktype (str): The tasktype of a client.
            model (trochvision.model): A torchvision model.
            seed (int): The seed of the client.
            current_and_last_checkpoint ([checkpoint], optional): The current and last checkpoint. Defaults to None.
        """
        self.c_id = c_id
        self.task_type = tasktype
        self.title_of_experiment = title
        self.model = model
        self.num_epochs = 1
        self.seed = seed

        self.prev_backbone = None
        self.curr_backbone = None
        self.prev_neighbour_backbones = []
        self.curr_neighbour_backbones = []
        self.curr_backbone_avg_with_neighbors = None
        self.prev_backbone_avg_with_neighbors = None

        self.round_metrics = []
        self.test_metrics = None
        self.checkpoint_dir = (
            f"checkpoints/{self.title_of_experiment}/{self.task_type}/{self.c_id}"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if current_and_last_checkpoint == None:
            self.current_checkpoint = self.initialize_checkpoint()
            self.last_round_checkpoint = self.initialize_checkpoint()
        else:
            self.current_checkpoint = current_and_last_checkpoint[0]
            self.last_round_checkpoint = current_and_last_checkpoint[1]
            self.round_metrics = current_and_last_checkpoint[0]["metrics"]

    def initialize_checkpoint(self):
        """Initializes a checkpoint.

        Returns:
            {}: returns a dictionary containg checkpoint metrics, including the model_state_dict.
        """
        return {
            "c_id": self.c_id,
            "epoch": 0,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "metrics": self.round_metrics,
            "test_metrics": self.test_metrics,
        }

    def update_checkpoint(self, epoch):
        """Updates a checkpoint.

        Args:
            epoch (int): The current epoch number.
        """
        self.current_checkpoint = {
            "c_id": self.c_id,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "metrics": self.round_metrics,
            "test_metrics": self.test_metrics,
        }

    def save_checkpoint_to_disk(self, prev_current):
        """Saves a checkpoint to disk.

        Args:
            prev_current (dict): The previous or current checkpoint that shall be saved.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{prev_current}_{self.c_id}_ckpt.pth"
        )
        # Save the checkpoint to a files
        torch.save(self.current_checkpoint, checkpoint_path)

    def conduct_training(self, num_epochs, current_round):
        """Manages the training of a client.

        Args:
            num_epochs (int): The amount of epochs the model shall be trained for.
            current_round (int): The current communication round.

        Returns:
            {}: The training and validation metrics.
        """
        self.save_checkpoint_to_disk("prev")
        start_epoch = current_round * num_epochs
        metrics = self.model.train_model(num_epochs, start_epoch)
        completed_epochs = num_epochs * current_round
        self.round_metrics.extend(metrics)
        self.update_checkpoint(completed_epochs)
        self.save_checkpoint_to_disk("curr")
        self.extract_backbones()
        return metrics

    def conduct_testing(self):
        """Manages the testing of the model."""
        self.test_metrics = self.model.test_model()
        self.update_checkpoint(self.current_checkpoint["epoch"])

    def conflict_averse(self, curr_backbones_dicts, prev_backbones_dicts, hca_alpha):
        """Performs the hyper conflict averse update

        Args:
            curr_backbones_dicts [{}]: The backbones of the current communication round
            prev_backbones_dicts [{}]: The backbones of the previous communication round
            hca_alpha float: Hyperparameter controlling the impact of the hyper conflict averse aggregation.

        Returns:
            [dict]: Backbones update with the hyper conflict averse aggregation scheme.
        """
        return hca.conflict_averse(
            curr_backbones_dicts, prev_backbones_dicts, hca_alpha
        )

    def replace_backbone(self, averaged_state_dicts):
        """Replaces a backbone with an averaged backbone state dict.

        Args:
            averaged_state_dicts {}: The average backbone dictionary.

        Returns:
            model: The current model with the backbone replaced.
        """

        # here the normal averaged state dict can and has to be used
        self.curr_backbone = averaged_state_dicts

        # Remove the 'backbone.' prefix, This has to be done as I will replace the backbone itself which does not have the prefix!!!.
        cleaned_state_dict = {
            key.replace("backbone.", ""): value
            for key, value in averaged_state_dicts.items()
        }
        self.model.backbone.load_state_dict(cleaned_state_dict, strict=True)

        return self.model

    def average_backbones(
        self, curr_personal_state_dict, curr_other_clients_state_dicts
    ):
        """Averages backbones.

        Args:
            curr_personal_state_dict .{}: Personal model state dictionary.
            curr_other_clients_state_dicts [{}]: Model state dictionaries of neighbours.

        Raises:
            ValueError: An Error if State dicts do not match.

        Returns:
            {}: Averaged State dictionary.
        """

        if curr_other_clients_state_dicts == None:
            warnings.warn("State dicts do not have matching keys.", UserWarning)
            return None

        state_dicts = [curr_personal_state_dict] + curr_other_clients_state_dicts

        reference_keys = list(state_dicts[0].keys())

        # Check if all state_dicts have the same keys in the same order
        for state_dict in state_dicts:
            if list(state_dict.keys()) != reference_keys:
                raise ValueError(
                    "State dicts do not have matching keys or the keys are in a different order."
                )

        averaged_state_dict = OrderedDict()

        for key in reference_keys:
            # Start by summing all the corresponding tensors
            summed_tensor = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)

            for state_dict in state_dicts:
                summed_tensor += state_dict[key].float()

            # Perform the division
            averaged_state_dict[key] = summed_tensor / len(state_dicts)

            # If the original tensor was an integer type, convert it back to the original dtype
            if state_dicts[0][key].dtype in [torch.int32, torch.int64]:
                averaged_state_dict[key] = averaged_state_dict[key].to(
                    state_dicts[0][key].dtype
                )

        return averaged_state_dict

    def extract_backbones(self):
        """Extract the backbones from the client.

        Returns:
            [{}]: The previous and current backbone of the client.
        """
        prev_checkpoint_path = os.path.join(
            self.checkpoint_dir, f"prev_{self.c_id}_ckpt.pth"
        )
        curr_checkpoint_path = os.path.join(
            self.checkpoint_dir, f"curr_{self.c_id}_ckpt.pth"
        )

        prev_checkpoint = torch.load(prev_checkpoint_path, weights_only=False)
        curr_checkpoint = torch.load(curr_checkpoint_path, weights_only=False)
        prev_backbone = self._extract_backbone_from_checkpoint(prev_checkpoint)
        curr_backbone = self._extract_backbone_from_checkpoint(curr_checkpoint)

        self.prev_backbone = prev_backbone
        self.curr_backbone = curr_backbone

        return prev_backbone, curr_backbone

    def _extract_backbone_from_checkpoint(self, ckpt):
        """Extracts the backbone from a checkpoint

        Args:
            ckpt (checkpoint): The loaded checkpoint.

        Returns:
            {}: The model state dict of the backbone
        """
        backbone = ckpt[
            "model_state_dict"
        ].copy()  # Make a copy to avoid modifying the original checkpoint

        # Identify all keys that belong to the head
        keys_to_remove = [key for key in backbone.keys() if key.startswith("head")]

        # Remove all identified keys
        for key in keys_to_remove:
            del backbone[key]

        return backbone
