import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
import pandas as pd
import numpy as np

# Assuming dataset.py, models/resnet.py, models/vit.py are accessible from this script's location
# If src is in PYTHONPATH or running from src/, these should work.
from dataset import OxfordPetDataset
from models.resnet import ResNet50
from models.vit import ViT


class ModelEvaluator:
    def __init__(
        self, model_path, device, data_dir, batch_size=32, results_dir="../evaluation"
    ):
        self.model_path = model_path
        self.device = device
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.results_dir = results_dir

        print(f"DEBUG: Initializing ModelEvaluator for {self.model_path}")

        try:
            self.checkpoint = torch.load(self.model_path, map_location=self.device)
            print(f"DEBUG: Checkpoint loaded successfully for {self.model_path}.")
            print(f"DEBUG: self.checkpoint type: {type(self.checkpoint)}")
            if self.checkpoint is None:
                raise FileNotFoundError(
                    f"Checkpoint data is None after loading {self.model_path}"
                )
            print(
                f"DEBUG: self.checkpoint keys (first level): {list(self.checkpoint.keys()) if isinstance(self.checkpoint, dict) else 'Not a dict'}"
            )
        except FileNotFoundError as e:
            print(
                f"ERROR_DEBUG: Could not load checkpoint: {self.model_path}. Error: {e}"
            )
            raise  # Re-raise the exception to stop further execution
        except Exception as e:
            print(
                f"ERROR_DEBUG: An unexpected error occurred while loading checkpoint {self.model_path}. Error: {e}"
            )
            raise  # Re-raise for critical failure

        self.model_architecture = self.checkpoint.get("model_architecture", "unknown")
        self.is_binary_classification = self.checkpoint.get(
            "binary_classification", True
        )  # Default based on original project
        self.model_name_or_path = self.checkpoint.get("model_name_or_path")  # For ViT

        print(f"DEBUG: Model architecture from checkpoint: {self.model_architecture}")
        print(
            f"DEBUG: Is binary classification from checkpoint: {self.is_binary_classification}"
        )
        print(
            f"DEBUG: ViT model_name_or_path from checkpoint: {self.model_name_or_path}"
        )

        self.model_filename = os.path.basename(self.model_path)
        self.evaluation_subdir = os.path.join(
            self.results_dir, os.path.splitext(self.model_filename)[0]
        )
        os.makedirs(self.evaluation_subdir, exist_ok=True)
        print(f"DEBUG: Evaluation subdirectory: {self.evaluation_subdir}")

        # self.num_classes will be set by _load_model_and_set_classes
        # self.class_names will be set by _get_test_dataloader
        self.model, self.num_classes = self._load_model_and_set_classes()

        if self.model:
            print(f"DEBUG: Model loaded and num_classes set to: {self.num_classes}")
            self.test_loader, self.class_names = self._get_test_dataloader()
            print(
                f"DEBUG: Test dataloader and class_names ({self.class_names}) obtained."
            )
        else:
            print(
                f"ERROR_DEBUG: Model could not be loaded in __init__. Aborting further setup for {self.model_path}."
            )
            # Handle cases where model loading might fail gracefully if needed, or ensure _load_model_and_set_classes raises
            self.test_loader = None
            self.class_names = []

    def _load_model_and_set_classes(self):
        print(f"DEBUG: Entering _load_model_and_set_classes for {self.model_path}")
        if not self.checkpoint or not isinstance(self.checkpoint, dict):
            print(
                f"ERROR_DEBUG: Checkpoint not loaded or not a dict in _load_model_and_set_classes for {self.model_path}."
            )
            raise ValueError(
                "Checkpoint not loaded correctly before calling _load_model_and_set_classes."
            )

        state_dict = self.checkpoint.get("model_state_dict")
        if not state_dict:
            print(
                f"ERROR_DEBUG: model_state_dict not found in checkpoint for {self.model_path}"
            )
            raise ValueError(
                f"model_state_dict not found in checkpoint: {self.model_path}"
            )
        print(f"DEBUG: model_state_dict obtained. Keys count: {len(state_dict.keys())}")

        num_classes = 0
        loaded_model = None

        if self.model_architecture == "resnet":
            print(f"DEBUG: Loading ResNet model from checkpoint.")
            determined_key_name = None
            last_layer_weights = None

            if "fc.weight" in state_dict:
                determined_key_name = "fc.weight"
            elif (
                "backbone.fc.weight" in state_dict
            ):  # Should not happen with current saving
                determined_key_name = "backbone.fc.weight"
            else:  # Fallback
                for key_in_dict in state_dict.keys():
                    if key_in_dict.endswith("fc.weight") or key_in_dict.endswith(
                        "classifier.weight"
                    ):
                        determined_key_name = key_in_dict
                        break

            if determined_key_name:
                print(f"DEBUG: ResNet classifier key: '{determined_key_name}'")
                last_layer_weights = state_dict[determined_key_name]
                num_classes = last_layer_weights.shape[0]
                # Use self.is_binary_classification which is read from the checkpoint
                # freeze_backbone=False as we are loading a saved model for evaluation, not training setup
                print(
                    f"DEBUG: Initializing ResNet50 with binary_classification={self.is_binary_classification} (derived from checkpoint)"
                )
                loaded_model = ResNet50(
                    binary_classification=self.is_binary_classification,
                    freeze_backbone=False,
                )
                print(
                    f"DEBUG: ResNet initialized. Expected num classes by model init: {'1 (binary)' if self.is_binary_classification else '37 (multiclass)'}. Actual from checkpoint: {num_classes}"
                )

                # Sanity check if the model's internal num_classes matches what we found
                # The ResNet50 class sets its fc layer based on binary_classification flag.
                # If binary_classification is True, fc_output_features is 1.
                # If binary_classification is False, fc_output_features is 37.
                expected_fc_output_features = 1 if self.is_binary_classification else 37
                if num_classes != expected_fc_output_features:
                    # This can happen if a binary model was saved with 2 output neurons (e.g. for CrossEntropyLoss with 2 classes)
                    # but our ResNet50 class definition for binary uses 1 output neuron (for BCEWithLogitsLoss)
                    # Or if a multiclass model was saved with a different number of classes than 37.
                    print(
                        f"WARNING_DEBUG: Mismatch between num_classes from checkpoint's fc layer ({num_classes}) "
                        f"and expected fc output features from ResNet50 class with binary_classification={self.is_binary_classification} ({expected_fc_output_features})."
                    )
                    print(
                        f"Proceeding with num_classes={num_classes} from checkpoint for state_dict loading."
                    )
                    # The loaded_model was already initialized based on self.is_binary_classification.
                    # The load_state_dict call later should handle the actual layer sizes from the checkpoint.

            else:
                print(
                    f"ERROR_DEBUG: Could not determine ResNet classifier layer key for {self.model_path}."
                )
                raise KeyError(
                    f"Could not determine ResNet classifier layer key. Keys: {list(state_dict.keys())}"
                )

        elif self.model_architecture == "vit":
            print(f"DEBUG: Loading ViT model from checkpoint.")
            possible_keys = [
                "classifier.weight",
                "vit.classifier.weight",
                "head.weight",
            ]
            last_layer_weights = None
            for key in possible_keys:
                if key in state_dict:
                    last_layer_weights = state_dict[key]
                    print(f"DEBUG: ViT classifier key: '{key}'")
                    break

            if last_layer_weights is not None:
                num_classes = last_layer_weights.size(0)
                # Ensure model_name_or_path is available, fallback if necessary
                vit_model_identifier = self.model_name_or_path
                if not vit_model_identifier:
                    vit_model_identifier = "google/vit-base-patch16-224"  # Default
                    print(
                        f"Warning: ViT model_name_or_path not in checkpoint. Using default: {vit_model_identifier}"
                    )
                loaded_model = ViT(
                    model_name_or_path=vit_model_identifier,
                    num_classes=num_classes,
                    pretrained_weights=False,
                )
                print(f"DEBUG: ViT initialized. Num classes: {num_classes}")
            else:
                print(
                    f"ERROR_DEBUG: Could not determine ViT classifier layer key for {self.model_path}."
                )
                raise KeyError(
                    f"Could not determine ViT classifier layer key. Keys: {list(state_dict.keys())}"
                )
        else:
            print(
                f"ERROR_DEBUG: Unknown model architecture: {self.model_architecture} for {self.model_path}"
            )
            raise ValueError(f"Unknown model architecture: {self.model_architecture}")

        if loaded_model and num_classes > 0:
            loaded_model.load_state_dict(state_dict)
            loaded_model.to(self.device)
            loaded_model.eval()
            print(
                f"DEBUG: Model '{self.model_architecture}' loaded, weights set, and in eval mode."
            )
            return loaded_model, num_classes
        else:
            print(
                f"ERROR_DEBUG: Failed to load model or determine num_classes for {self.model_path}"
            )
            # This should ideally be caught by earlier raises if keys aren't found
            raise SystemError(
                f"Failed to correctly load model or determine num_classes for {self.model_path}"
            )

    def _get_test_dataloader(self):
        print(
            f"DEBUG: Getting test dataloader. Num classes for dataset: {self.num_classes}, Binary classification: {self.is_binary_classification}"
        )
        # Determine actual binary_classification flag for dataset loading
        # If num_classes is 2, it's likely binary, but self.is_binary_classification from checkpoint is more reliable
        dataset_binary_flag = self.is_binary_classification

        # OxfordPetDataset's get_dataloaders expects binary_classification to guide label processing.
        # If num_classes is > 2, it MUST be multiclass for the dataset.
        if self.num_classes > 2 and dataset_binary_flag:
            print(
                f"Warning: num_classes is {self.num_classes} but checkpoint says binary. Forcing dataset to multiclass for dataloader."
            )
            dataset_binary_flag = False
        elif self.num_classes == 2 and not dataset_binary_flag:
            print(
                f"Warning: num_classes is 2 but checkpoint says multiclass. Forcing dataset to binary for dataloader."
            )
            dataset_binary_flag = True

        # Get dataloaders - the fourth item is num_classes_from_dataset (an int)
        _, _, test_loader, num_classes_from_dataset = OxfordPetDataset.get_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            binary_classification=dataset_binary_flag,  # Use the determined flag
            data_augmentation=False,  # No augmentation for testing
            model_type=self.model_architecture,  # Pass architecture for potential ViT specific transforms
            vit_model_name=(
                self.model_name_or_path
                if self.model_architecture == "vit"
                else "google/vit-base-patch16-224"
            ),
            random_seed=42,  # Consistent seed for test split
        )
        print(
            f"DEBUG: Dataloader obtained. Num classes from dataset method: {num_classes_from_dataset}"
        )

        # Validate num_classes from model with num_classes_from_dataset
        # Both self.num_classes (from model checkpoint) and num_classes_from_dataset (from dataset loader config) should align.
        if self.num_classes != num_classes_from_dataset:
            print(
                f"ERROR_DEBUG: Mismatch! Num_classes from model checkpoint ({self.num_classes}) vs num_classes from dataset configuration ({num_classes_from_dataset})"
            )
            # This is a significant mismatch and likely indicates a problem in how the model was saved or how the dataset is being loaded for its type.
            # For ResNet binary, model checkpoint num_classes might be 1 (logit) or 2 (softmaxed classes). Dataset should return 1 if binary_classification=True.
            # For ResNet multiclass, model checkpoint num_classes should be 37. Dataset should return 37 if binary_classification=False.
            if (
                self.model_architecture == "resnet"
                and self.is_binary_classification
                and self.num_classes == 1
                and num_classes_from_dataset == 1
            ):
                print(
                    "INFO_DEBUG: ResNet binary (1 output neuron from model, 1 class type from dataset). This is consistent."
                )
            elif (
                self.model_architecture == "resnet"
                and not self.is_binary_classification
                and self.num_classes == 37
                and num_classes_from_dataset == 37
            ):
                print(
                    "INFO_DEBUG: ResNet multiclass (37 output neurons from model, 37 class type from dataset). This is consistent."
                )
            elif (
                self.model_architecture == "vit"
                and self.is_binary_classification
                and self.num_classes == 2
                and num_classes_from_dataset == 1
            ):  # ViT usually outputs 2 for binary_cross_entropy
                print(
                    "INFO_DEBUG: ViT binary (2 output neurons from model, 1 class type from dataset config). This implies dataset is binary."
                )
                # This can be acceptable if the ViT model uses 2 outputs for binary and dataset is configured as binary (num_classes_from_dataset=1)
            elif (
                self.model_architecture == "vit"
                and not self.is_binary_classification
                and self.num_classes == 37
                and num_classes_from_dataset == 37
            ):
                print(
                    "INFO_DEBUG: ViT multiclass (37 output neurons from model, 37 class type from dataset). This is consistent."
                )
            else:
                raise ValueError(
                    f"Critical mismatch between model's num_classes ({self.num_classes}) and dataset's configured num_classes ({num_classes_from_dataset})."
                )

        # Generate class names for reporting based on self.is_binary_classification (from checkpoint)
        # and num_classes_from_dataset (which should align with self.num_classes after the check above)
        class_names_for_report = []
        if self.is_binary_classification:
            # If model is binary (e.g. ResNet outputs 1 logit, or ViT outputs 2 logits for binary CE)
            # And dataset was loaded as binary (num_classes_from_dataset should be 1)
            print(
                "INFO_DEBUG: Binary classification. Generating class names ['Cat', 'Dog'] for reporting."
            )
            class_names_for_report = ["Cat", "Dog"]
        else:  # Multiclass
            # num_classes_from_dataset should be 37 for multiclass
            print(
                f"INFO_DEBUG: Multiclass classification ({num_classes_from_dataset} classes). Generating generic breed names for reporting."
            )
            class_names_for_report = [
                f"Breed_{i}" for i in range(num_classes_from_dataset)
            ]

        return test_loader, class_names_for_report

    def evaluate(self):
        print(f"DEBUG: Entering evaluate() for {self.model_path}")
        print(
            f"DEBUG: self.checkpoint type at start of evaluate: {type(self.checkpoint)}"
        )
        if self.checkpoint is None:
            print(
                f"ERROR_DEBUG: self.checkpoint is None at the start of evaluate() for {self.model_path}. This should have been caught in _load_model."
            )
            # This indicates a logic error if _load_model was supposed to guarantee self.checkpoint is not None
            return  # Or raise an error
        else:
            print(
                f"DEBUG: self.checkpoint keys at start of evaluate (first level): {list(self.checkpoint.keys()) if isinstance(self.checkpoint, dict) else 'Not a dict'}"
            )

        metric_file_path = os.path.join(
            self.evaluation_subdir, "evaluation_metrics.json"
        )
        if os.path.exists(metric_file_path):
            print(
                f"Evaluation results for {self.model_filename} already exist. Skipping re-evaluation, loading existing."
            )
            try:
                with open(metric_file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(
                    f"Error loading existing metrics for {self.model_filename}: {e}. Will re-evaluate."
                )
                # Fall through to re-evaluate

        all_preds = []
        all_labels = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                logits = (
                    outputs.logits if hasattr(outputs, "logits") else outputs
                )  # Handle HuggingFace output object

                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        unique_true_and_pred_labels = sorted(
            list(np.unique(np.concatenate((all_labels, all_preds))))
        )
        print(
            f"DEBUG: Unique labels found in true and preds: {unique_true_and_pred_labels}"
        )

        target_names_for_report = []
        if self.is_binary_classification:
            # For binary, self.class_names is already ['Cat', 'Dog']
            # We need to select the correct names based on actual labels present
            temp_binary_names = {}
            if 0 in unique_true_and_pred_labels:  # Cat
                temp_binary_names[0] = self.class_names[0]  # Should be 'Cat'
            if 1 in unique_true_and_pred_labels:  # Dog
                temp_binary_names[1] = self.class_names[1]  # Should be 'Dog'
            target_names_for_report = [
                temp_binary_names[label]
                for label in unique_true_and_pred_labels
                if label in temp_binary_names
            ]
            if not target_names_for_report and unique_true_and_pred_labels:
                # Fallback if only one class predicted and it wasn't 0 or 1, or self.class_names was wrong
                print(
                    f"WARNING: Binary classification with unusual labels {unique_true_and_pred_labels}. Using generic target names."
                )
                target_names_for_report = [
                    f"ActualClass_{i}" for i in unique_true_and_pred_labels
                ]
        else:  # Multiclass
            # self.class_names should be ['Breed_0', ..., 'Breed_36']
            # We only want names for labels that are actually present.
            target_names_for_report = []
            for label_val in unique_true_and_pred_labels:
                if 0 <= label_val < len(self.class_names):
                    target_names_for_report.append(self.class_names[label_val])
                else:
                    target_names_for_report.append(
                        f"ActualClass_{label_val}"
                    )  # Fallback for unexpected label values

        # Ensure target_names_for_report has same length as unique_true_and_pred_labels
        if len(target_names_for_report) != len(unique_true_and_pred_labels):
            print(
                f"CRITICAL_DEBUG: Length mismatch after generating target_names_for_report. unique_labels: {unique_true_and_pred_labels}, generated_names: {target_names_for_report}"
            )
            # Fallback to generic names based on unique_true_and_pred_labels to prevent crash
            target_names_for_report = [
                f"FallbackClass_{i}" for i in unique_true_and_pred_labels
            ]

        print(
            f"Debug: Classification report will use target_names: {target_names_for_report} and labels: {unique_true_and_pred_labels}"
        )
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=target_names_for_report,  # Must match the order and number of unique_true_and_pred_labels
            labels=unique_true_and_pred_labels,  # Explicitly pass the labels that are present and to be reported
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "model_filename": self.model_filename,
            "model_architecture": self.model_architecture,
            "classification_type": (
                "binary" if self.is_binary_classification else "multiclass"
            ),
            "num_classes_model": self.num_classes,
            "accuracy": np.mean(all_preds == all_labels),
            "classification_report": report_dict,
            "roc_auc": None,
            "pr_auc": None,
            "roc_auc_ovr_weighted": None,
        }

        if self.num_classes == 2 and all_probs.shape[1] == 2:  # Binary classification
            try:
                metrics["roc_auc"] = roc_auc_score(all_labels, all_probs[:, 1])
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
                self.plot_roc_curve(fpr, tpr, metrics["roc_auc"], "roc_curve.png")

                precision, recall, _ = precision_recall_curve(
                    all_labels, all_probs[:, 1]
                )
                metrics["pr_auc"] = auc(recall, precision)
                self.plot_pr_curve(precision, recall, metrics["pr_auc"], "pr_curve.png")
            except Exception as e:
                print(
                    f"Error calculating/plotting binary ROC/PR for {self.model_filename}: {e}"
                )
        elif self.num_classes > 2:  # Multiclass classification
            try:
                metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="weighted"
                )
                # Per-class ROC/PR curves could be plotted here if desired
            except (
                ValueError
            ) as e:  # May happen if a class has no test samples or all predictions are one class
                print(f"Could not calculate OvR ROC AUC for {self.model_filename}: {e}")

        with open(metric_file_path, "w") as f:
            json.dump(
                metrics, f, indent=4, cls=NpEncoder
            )  # Use NpEncoder for numpy types

        self.plot_confusion_matrix(
            all_labels, all_preds, target_names_for_report, "confusion_matrix.png"
        )

        if "history" in self.checkpoint:
            self.plot_training_history(
                self.checkpoint["history"], "training_history.png"
            )

        print(
            f"Evaluation complete for {self.model_filename}. Results saved to {self.evaluation_subdir}"
        )
        return metrics

    def plot_confusion_matrix(self, labels, preds, class_names_for_plot, filename):
        cm = confusion_matrix(labels, preds)
        # Ensure figure size is reasonable based on the number of classes
        fig_width = max(8, self.num_classes * 0.6)
        fig_height = max(6, self.num_classes * 0.5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names_for_plot,
            yticklabels=class_names_for_plot,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {self.model_filename}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.evaluation_subdir, filename))
        plt.close()

    def plot_roc_curve(self, fpr, tpr, roc_auc_value, filename):
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc_value:.3f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.model_filename}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.evaluation_subdir, filename))
        plt.close()

    def plot_pr_curve(self, precision, recall, pr_auc_value, filename):
        plt.figure()
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (area = {pr_auc_value:.3f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f"Precision-Recall Curve - {self.model_filename}")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.evaluation_subdir, filename))
        plt.close()

    def plot_training_history(self, history, filename):
        epochs = range(1, len(history.get("train_loss", [])) + 1)
        if not epochs:
            return

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        if "train_loss" in history and history["train_loss"]:
            axs[0].plot(epochs, history["train_loss"], "bo-", label="Training Loss")
        if "val_loss" in history and history["val_loss"]:
            axs[0].plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
        axs[0].set_title("Training and Validation Loss")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        if axs[0].has_data():
            axs[0].legend()

        if "train_acc" in history and history["train_acc"]:
            axs[1].plot(epochs, history["train_acc"], "bo-", label="Training Accuracy")
        if "val_acc" in history and history["val_acc"]:
            axs[1].plot(epochs, history["val_acc"], "ro-", label="Validation Accuracy")
        axs[1].set_title("Training and Validation Accuracy")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        if axs[1].has_data():
            axs[1].legend()

        fig.suptitle(f"Training History - {self.model_filename}", fontsize=14)
        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make space for suptitle
        plt.savefig(os.path.join(self.evaluation_subdir, filename))
        plt.close()


class NpEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def update_comparison_file(results_dir, all_model_metrics_list):
    comparison_filepath = os.path.join(results_dir, "model_comparison.csv")

    processed_metrics_for_df = []
    for metrics_data in all_model_metrics_list:
        if not metrics_data:
            continue  # Skip if None (e.g. error during eval)

        report = metrics_data.get("classification_report", {})
        weighted_avg = (
            report.get("weighted avg", {}) if isinstance(report, dict) else {}
        )

        # Determine primary ROC AUC value to display
        roc_auc_display = metrics_data.get("roc_auc")  # Binary
        if roc_auc_display is None:
            roc_auc_display = metrics_data.get("roc_auc_ovr_weighted")  # Multiclass

        row = {
            "Model Filename": metrics_data.get("model_filename"),
            "Architecture": metrics_data.get("model_architecture"),
            "Type": metrics_data.get("classification_type"),
            "Num Classes": metrics_data.get("num_classes_model"),
            "Overall Accuracy": f'{metrics_data.get("accuracy", 0):.4f}',
            "Weighted Precision": f'{weighted_avg.get("precision", 0):.4f}',
            "Weighted Recall": f'{weighted_avg.get("recall", 0):.4f}',
            "Weighted F1-Score": f'{weighted_avg.get("f1-score", 0):.4f}',
            "ROC AUC": (
                f"{roc_auc_display:.4f}" if roc_auc_display is not None else "N/A"
            ),
            "PR AUC (Binary)": (
                f'{metrics_data.get("pr_auc"):.4f}'
                if metrics_data.get("pr_auc") is not None
                else "N/A"
            ),
            "Details Path": os.path.splitext(metrics_data.get("model_filename"))[
                0
            ],  # Path relative to evaluation_dir
        }
        processed_metrics_for_df.append(row)

    if not processed_metrics_for_df:
        print("No valid metrics to update comparison file.")
        return

    new_df = pd.DataFrame(processed_metrics_for_df)

    combined_df = new_df
    if os.path.exists(comparison_filepath):
        try:
            existing_df = pd.read_csv(comparison_filepath)
            if not existing_df.empty:
                # Merge: Update existing rows, append new ones
                # Use 'Model Filename' as index for easy update/append
                new_df.set_index("Model Filename", inplace=True)
                existing_df.set_index("Model Filename", inplace=True)
                combined_df = new_df.combine_first(
                    existing_df
                ).reset_index()  # new_df values take precedence
            # If existing_df is empty, combined_df remains new_df
        except pd.errors.EmptyDataError:  # File exists but is empty
            pass  # combined_df is already new_df
        except Exception as e:
            print(
                f"Error reading existing comparison file {comparison_filepath}: {e}. Overwriting."
            )
            # Fall through to save new_df as combined_df

    if not combined_df.empty:
        combined_df.sort_values(by="Model Filename", inplace=True)
        combined_df.to_csv(comparison_filepath, index=False)
        print(f"Model comparison summary updated/created at {comparison_filepath}")
    else:
        print("No data to write to comparison file.")


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Project root is one level up from src where this script is
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DATA_DIR = os.path.join(project_root, "data", "raw")
    MODELS_ROOT_DIR = os.path.join(project_root, "models")
    EVALUATION_ROOT_DIR = os.path.join(project_root, "evaluation")

    print(f"Data directory: {DATA_DIR}")
    print(f"Models root directory: {MODELS_ROOT_DIR}")
    print(f"Evaluation root directory: {EVALUATION_ROOT_DIR}")

    os.makedirs(EVALUATION_ROOT_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found at {DATA_DIR}. Please check the path.")
        return
    if not os.path.exists(MODELS_ROOT_DIR):
        print(
            f"ERROR: Models directory not found at {MODELS_ROOT_DIR}. No models to evaluate."
        )
        return

    all_model_metrics_collected = []

    for arch_folder in os.listdir(MODELS_ROOT_DIR):  # e.g., 'resnet', 'vit'
        arch_path = os.path.join(MODELS_ROOT_DIR, arch_folder)
        if os.path.isdir(arch_path):
            for type_folder in os.listdir(arch_path):  # e.g., 'binary', 'multiclass'
                type_path = os.path.join(arch_path, type_folder)
                if os.path.isdir(type_path):
                    for model_file in os.listdir(type_path):
                        if model_file.endswith(".pth"):
                            model_full_path = os.path.join(type_path, model_file)
                            print(f"\n--- Processing model: {model_full_path} ---")
                            try:
                                evaluator = ModelEvaluator(
                                    model_path=model_full_path,
                                    device=DEVICE,
                                    data_dir=DATA_DIR,
                                    results_dir=EVALUATION_ROOT_DIR,  # Pass the root eval dir
                                )
                                # evaluate() method now handles checking if metrics exist and loading them
                                metrics = evaluator.evaluate()
                                if metrics:
                                    all_model_metrics_collected.append(metrics)
                            except Exception as e:
                                print(
                                    f"ERROR during evaluation setup or execution for {model_file}: {e}"
                                )
                                import traceback

                                traceback.print_exc()

    if all_model_metrics_collected:
        print(f"\n--- Updating overall comparison file ---")
        update_comparison_file(EVALUATION_ROOT_DIR, all_model_metrics_collected)
    else:
        print("\nNo models found or processed. Comparison file not updated.")


if __name__ == "__main__":
    main()
