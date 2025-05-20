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
import csv

# Assuming dataset.py, models/resnet.py, models/vit.py are accessible from this script's location
# If src is in PYTHONPATH or running from src/, these should work.
from dataset import OxfordPetDataset
from models.resnet import ResNet50
from models.vit import ViT


class ModelEvaluator:
    def __init__(
        self,
        model_path,
        device,
        data_dir,
        batch_size=32,
        evaluation_run_specific_dir=None,
    ):
        self.model_path = model_path
        self.device = device
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.results_dir = results_dir # Removed, specific dir is now passed

        # print(f"DEBUG: Initializing ModelEvaluator for {self.model_path}")

        try:
            self.checkpoint = torch.load(self.model_path, map_location=self.device)
            # print(f"DEBUG: Checkpoint loaded successfully for {self.model_path}.")
            # print(f"DEBUG: self.checkpoint type: {type(self.checkpoint)}")
            if self.checkpoint is None:
                raise FileNotFoundError(
                    f"Checkpoint data is None after loading {self.model_path}"
                )
            # print(
            # f"DEBUG: self.checkpoint keys (first level): {list(self.checkpoint.keys()) if isinstance(self.checkpoint, dict) else 'Not a dict'}"
            # )
        except FileNotFoundError as e:
            print(f"ERROR: Could not load checkpoint: {self.model_path}. Error: {e}")
            raise  # Re-raise the exception to stop further execution
        except Exception as e:
            print(
                f"ERROR: An unexpected error occurred while loading checkpoint {self.model_path}. Error: {e}"
            )
            raise  # Re-raise for critical failure

        self.model_architecture = self.checkpoint.get("model_architecture", "unknown")
        self.is_binary_classification = self.checkpoint.get(
            "binary_classification", True
        )  # Default based on original project
        self.model_name_or_path = self.checkpoint.get("model_name_or_path")  # For ViT

        # Existing debug prints (will be updated or preceded by new logic)
        # print(f"DEBUG: Model architecture from checkpoint: {self.model_architecture}")
        # print(
        #     f"DEBUG: Is binary classification from checkpoint: {self.is_binary_classification}"
        # )
        # print(
        #     f"DEBUG: ViT model_name_or_path from checkpoint: {self.model_name_or_path}"
        # )

        # Correction logic for model_architecture
        original_checkpoint_arch = self.model_architecture
        corrected_arch = False

        if (
            self.model_architecture == "resnet"
            and ("vit" in self.model_path.lower() or "/vit/" in self.model_path)
            and self.model_name_or_path
        ):
            print(
                f"WARNING: Checkpoint indicates model_architecture is 'resnet', but model_path ('{self.model_path}') and presence of 'model_name_or_path' ('{self.model_name_or_path}') suggest 'vit'. Overriding to 'vit'."
            )
            self.model_architecture = "vit"
            corrected_arch = True
        elif (
            self.model_architecture == "vit"
            and ("resnet" in self.model_path.lower() or "/resnet/" in self.model_path)
            and not self.model_name_or_path
        ):  # ResNet models wouldn't typically have model_name_or_path set from HuggingFace
            print(
                f"WARNING: Checkpoint indicates model_architecture is 'vit', but model_path ('{self.model_path}') suggests 'resnet' and 'model_name_or_path' is missing. Overriding to 'resnet'."
            )
            self.model_architecture = "resnet"
            corrected_arch = True
        elif self.model_architecture == "unknown":
            inferred_from_path_or_specific_key = False
            if self.model_name_or_path and (
                "vit" in self.model_path.lower()
                or "/vit/" in self.model_path
                or not (
                    "resnet" in self.model_path.lower() or "/resnet/" in self.model_path
                )
            ):
                # If model_name_or_path exists, and path is vit-like or neutral, lean towards vit.
                self.model_architecture = "vit"
                inferred_from_path_or_specific_key = True
                print(
                    f"INFO: model_architecture was 'unknown'. Inferred as 'vit' based on model_name_or_path ('{self.model_name_or_path}') and/or model_path."
                )
            elif "vit" in self.model_path.lower() or "/vit/" in self.model_path:
                self.model_architecture = "vit"
                inferred_from_path_or_specific_key = True
                print(
                    f"INFO: model_architecture was 'unknown'. Inferred as 'vit' based on model_path ('{self.model_path}')."
                )
            elif "resnet" in self.model_path.lower() or "/resnet/" in self.model_path:
                self.model_architecture = "resnet"
                inferred_from_path_or_specific_key = True
                print(
                    f"INFO: model_architecture was 'unknown'. Inferred as 'resnet' based on model_path ('{self.model_path}')."
                )

            if inferred_from_path_or_specific_key:
                corrected_arch = True  # Technically an inference, but changes 'unknown'

        print(
            f"DEBUG: Original model_architecture from checkpoint: {original_checkpoint_arch}"
        )
        if corrected_arch:
            print(
                f"DEBUG: Effective model_architecture after correction/inference: {self.model_architecture}"
            )
        else:
            print(
                f"DEBUG: Model architecture from checkpoint (no correction applied): {self.model_architecture}"
            )

        print(
            f"DEBUG: Is binary classification from checkpoint: {self.is_binary_classification}"
        )
        print(
            f"DEBUG: ViT model_name_or_path (used if arch is ViT): {self.model_name_or_path}"
        )

        self.model_filename = os.path.basename(self.model_path)

        if evaluation_run_specific_dir is None:
            # This case should ideally not be hit if called from perform_and_save_evaluation
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            fallback_dir_base = os.path.join(
                project_root,
                "evaluation",
                self.model_architecture,
                "binary" if self.is_binary_classification else "multiclass",
            )
            self.evaluation_subdir = os.path.join(
                fallback_dir_base, os.path.splitext(self.model_filename)[0]
            )
            print(
                f"WARNING: evaluation_run_specific_dir not provided to ModelEvaluator, using fallback: {self.evaluation_subdir}"
            )
        else:
            self.evaluation_subdir = evaluation_run_specific_dir

        os.makedirs(self.evaluation_subdir, exist_ok=True)
        # print(f"DEBUG: Evaluation artifacts will be saved to: {self.evaluation_subdir}")

        # self.num_classes will be set by _load_model_and_set_classes
        # self.class_names will be set by _get_test_dataloader
        self.model, self.num_classes = self._load_model_and_set_classes()

        if self.model:
            # print(f"DEBUG: Model loaded and num_classes set to: {self.num_classes}")
            self.test_loader, self.class_names = self._get_test_dataloader()
            # print(
            #     f"DEBUG: Test dataloader and class_names ({self.class_names}) obtained."
            # )
        else:
            print(
                f"ERROR: Model could not be loaded in __init__. Aborting further setup for {self.model_path}."
            )
            # Handle cases where model loading might fail gracefully if needed, or ensure _load_model_and_set_classes raises
            self.test_loader = None
            self.class_names = []

    def _load_model_and_set_classes(self):
        # print(f"DEBUG: Entering _load_model_and_set_classes for {self.model_path}")
        if not self.checkpoint or not isinstance(self.checkpoint, dict):
            print(
                f"ERROR: Checkpoint not loaded or not a dict in _load_model_and_set_classes for {self.model_path}."
            )
            raise ValueError(
                "Checkpoint not loaded correctly before calling _load_model_and_set_classes."
            )

        state_dict = self.checkpoint.get("model_state_dict")
        if not state_dict:
            print(
                f"ERROR: model_state_dict not found in checkpoint for {self.model_path}"
            )
            raise ValueError(
                f"model_state_dict not found in checkpoint: {self.model_path}"
            )
        # print(f"DEBUG: model_state_dict obtained. Keys count: {len(state_dict.keys())}")

        num_classes = 0
        loaded_model = None

        if self.model_architecture == "resnet":
            # print(f"DEBUG: Loading ResNet model from checkpoint.")
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
                # print(f"DEBUG: ResNet classifier key: '{determined_key_name}'")
                last_layer_weights = state_dict[determined_key_name]
                num_classes = last_layer_weights.shape[0]
                # Use self.is_binary_classification which is read from the checkpoint
                # freeze_backbone=False as we are loading a saved model for evaluation, not training setup
                # print(
                #     f"DEBUG: Initializing ResNet50 with binary_classification={self.is_binary_classification} (derived from checkpoint)"
                # )
                loaded_model = ResNet50(
                    binary_classification=self.is_binary_classification,
                    freeze_backbone=False,  # For eval, backbone state is as saved
                )
                # print(
                #     f"DEBUG: ResNet initialized. Expected num classes by model init: {'1 (binary)' if self.is_binary_classification else '37 (multiclass)'}. Actual from checkpoint: {num_classes}"
                # )

                expected_fc_output_features = 1 if self.is_binary_classification else 37
                if num_classes != expected_fc_output_features:
                    print(
                        f"WARNING: Mismatch between num_classes from checkpoint's fc layer ({num_classes}) "
                        f"and expected fc output features from ResNet50 class with binary_classification={self.is_binary_classification} ({expected_fc_output_features})."
                    )
                    # print(
                    #     f"Proceeding with num_classes={num_classes} from checkpoint for state_dict loading."
                    # )

            else:
                print(
                    f"ERROR: Could not determine ResNet classifier layer key for {self.model_path}."
                )
                raise KeyError(
                    f"Could not determine ResNet classifier layer key. Keys: {list(state_dict.keys())}"
                )

        elif self.model_architecture == "vit":
            # print(f"DEBUG: Loading ViT model from checkpoint.")
            possible_keys = [
                "classifier.weight",
                "vit.classifier.weight",
                "head.weight",
                "vit_model.classifier.weight",
                "vit_model.vit.classifier.weight",
                "vit_model.head.weight",
            ]
            last_layer_weights = None
            for key in possible_keys:
                if key in state_dict:
                    last_layer_weights = state_dict[key]
                    # print(f"DEBUG: ViT classifier key: '{key}'")
                    break

            if last_layer_weights is not None:
                num_classes = last_layer_weights.size(0)
                vit_model_identifier = self.model_name_or_path
                if not vit_model_identifier:
                    vit_model_identifier = "google/vit-base-patch16-224"  # Default
                    print(
                        f"Warning: ViT model_name_or_path not in checkpoint. Using default: {vit_model_identifier}"
                    )
                loaded_model = ViT(
                    model_name_or_path=vit_model_identifier,
                    binary_classification=self.is_binary_classification,
                )
                # print(f"DEBUG: ViT initialized. Num classes: {num_classes}")
            else:
                print(
                    f"ERROR: Could not determine ViT classifier layer key for {self.model_path}."
                )
                raise KeyError(
                    f"Could not determine ViT classifier layer key. Keys: {list(state_dict.keys())}"
                )
        else:
            print(
                f"ERROR: Unknown model architecture: {self.model_architecture} for {self.model_path}"
            )
            raise ValueError(f"Unknown model architecture: {self.model_architecture}")

        if loaded_model and num_classes > 0:
            loaded_model.load_state_dict(state_dict)
            loaded_model.to(self.device)
            loaded_model.eval()
            # print(
            #     f"DEBUG: Model '{self.model_architecture}' loaded, weights set, and in eval mode."
            # )
            return loaded_model, num_classes
        else:
            print(
                f"ERROR: Failed to load model or determine num_classes for {self.model_path}"
            )
            raise SystemError(
                f"Failed to correctly load model or determine num_classes for {self.model_path}"
            )

    def _get_test_dataloader(self):
        # print(
        #     f"DEBUG: Getting test dataloader. Num classes for dataset: {self.num_classes}, Binary classification: {self.is_binary_classification}"
        # )
        dataset_binary_flag = self.is_binary_classification

        if self.num_classes > 2 and dataset_binary_flag:
            print(
                f"Warning: num_classes is {self.num_classes} but checkpoint says binary. Forcing dataset to multiclass for dataloader."
            )
            dataset_binary_flag = False
        elif self.num_classes == 2 and not dataset_binary_flag:  # Typically ViT binary
            print(
                f"Warning: num_classes is 2 (often ViT binary) but checkpoint says multiclass. Forcing dataset to binary for dataloader based on is_binary_classification flag."
            )
            # Keep dataset_binary_flag as self.is_binary_classification from checkpoint is king
        elif (
            self.num_classes == 1 and not dataset_binary_flag
        ):  # ResNet binary (1 logit) but checkpoint somehow multiclass
            print(
                f"Warning: num_classes is 1 (ResNet binary) but checkpoint indicates multiclass. Using binary for dataloader based on num_classes=1."
            )
            dataset_binary_flag = True

        _, _, test_loader, num_classes_from_dataset = OxfordPetDataset.get_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            binary_classification=dataset_binary_flag,
            data_augmentation=False,
            model_type=self.model_architecture,
            vit_model_name=(
                self.model_name_or_path
                if self.model_architecture == "vit"
                else "google/vit-base-patch16-224"
            ),
            random_seed=42,
        )
        # print(
        #     f"DEBUG: Dataloader obtained. Num classes from dataset method: {num_classes_from_dataset}"
        # )

        if self.is_binary_classification:
            # For binary, model (ResNet) might output 1 logit (num_classes=1), ViT might output 2 (num_classes=2).
            # Dataset loader with binary_classification=True gives num_classes_from_dataset=1 (labels are 0 or 1).
            # This is generally fine. The important thing is that labels are 0/1.
            if self.num_classes not in [1, 2]:
                print(
                    f"WARNING: Binary classification model has unexpected num_classes: {self.num_classes}"
                )
            if num_classes_from_dataset != 1:
                raise ValueError(
                    f"Binary classification mode, but dataset loader returned {num_classes_from_dataset} classes instead of 1."
                )
        else:  # Multiclass
            if self.num_classes != num_classes_from_dataset:
                raise ValueError(
                    f"Multiclass mismatch: Model num_classes ({self.num_classes}) vs Dataset num_classes ({num_classes_from_dataset})"
                )

        class_names_for_report = []
        if self.is_binary_classification:
            # print(
            #     "INFO_DEBUG: Binary classification. Generating class names ['Cat', 'Dog'] for reporting."
            # )
            class_names_for_report = ["Cat", "Dog"]
        else:
            # print(
            #     f"INFO_DEBUG: Multiclass classification ({num_classes_from_dataset} classes). Generating generic breed names for reporting."
            # )
            class_names_for_report = [
                f"Breed_{i}" for i in range(num_classes_from_dataset)
            ]

        return test_loader, class_names_for_report

    def evaluate(self):
        # print(f"DEBUG: Entering evaluate() for {self.model_path}")
        # print(
        #     f"DEBUG: self.checkpoint type at start of evaluate: {type(self.checkpoint)}"
        # )
        if self.checkpoint is None:
            print(
                f"ERROR: self.checkpoint is None at the start of evaluate() for {self.model_path}."
            )
            return
        # else:
        # print(
        #     f"DEBUG: self.checkpoint keys at start of evaluate (first level): {list(self.checkpoint.keys()) if isinstance(self.checkpoint, dict) else 'Not a dict'}"
        # )

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

        all_preds_list = []
        all_labels_list = []
        # For binary: store P(class_1). For multiclass: store [P(class_0), ..., P(class_N)]
        all_probs_for_metrics_list = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels_batch = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                logits_for_eval = (
                    outputs.logits if hasattr(outputs, "logits") else outputs
                )

                current_batch_preds = None
                current_batch_probs_for_roc_pr = None

                if self.is_binary_classification:
                    if (
                        self.model_architecture == "resnet"
                    ):  # ResNet binary outputs [batch, 1] logit
                        # Sigmoid gives P(class_1) for the single logit output
                        probs_class_1 = torch.sigmoid(logits_for_eval).squeeze(
                            -1
                        )  # Shape: [batch]
                        current_batch_preds = (probs_class_1 > 0.5).float()
                        current_batch_probs_for_roc_pr = (
                            probs_class_1.cpu().numpy()
                        )  # Store P(class_1)
                    else:  # Assumes ViT binary or other binary models output [batch, 2] logits
                        probs_all_classes = torch.softmax(
                            logits_for_eval, dim=1
                        )  # Shape: [batch, 2]
                        current_batch_preds = torch.argmax(probs_all_classes, dim=1)
                        current_batch_probs_for_roc_pr = (
                            probs_all_classes[:, 1].cpu().numpy()
                        )  # Store P(class_1)
                else:  # Multiclass
                    probs_all_classes = torch.softmax(
                        logits_for_eval, dim=1
                    )  # Shape: [batch, num_classes]
                    current_batch_preds = torch.argmax(probs_all_classes, dim=1)
                    current_batch_probs_for_roc_pr = (
                        probs_all_classes.cpu().numpy()
                    )  # Store all class probabilities

                all_preds_list.extend(
                    current_batch_preds.cpu().numpy()
                )  # Store predictions
                all_labels_list.extend(labels_batch.cpu().numpy())  # Store true labels
                # For binary, this list will extend with 1D arrays (P(class_1) per batch item)
                # For multiclass, this list will extend with 2D arrays (P(all classes) per batch item)
                all_probs_for_metrics_list.extend(current_batch_probs_for_roc_pr)

        all_preds = np.array(all_preds_list)
        all_labels = np.array(all_labels_list)

        # Convert all_probs_for_metrics_list to a single NumPy array
        # If binary, it was a list of scalars (P(class_1)), so np.array makes it (n_samples,)
        # If multiclass, it was a list of 1D arrays (one per sample, if extend was used on individual sample probs)
        # or it was a list of batch_sized arrays. If it was extend(probs_all_classes.cpu().numpy()), then it is a list of arrays.
        # Need to ensure all_probs_np is (n_samples,) for binary P(class_1) or (n_samples, n_classes) for multiclass.

        if self.is_binary_classification:
            # all_probs_for_metrics_list should be a flat list of P(class_1) values
            all_probs_np = np.array(
                all_probs_for_metrics_list
            )  # Expected shape (n_samples,)
        else:  # Multiclass
            # all_probs_for_metrics_list contains arrays of shape (batch_size_i, num_classes) if extend was on batch arrays
            # or it might be a flat list if extend was on individual sample arrays - check extend logic.
            # current_batch_probs_for_roc_pr = probs_all_classes.cpu().numpy() # This is (batch_size, num_classes)
            # all_probs_for_metrics_list.extend(current_batch_probs_for_roc_pr) <-- extends with rows of the array not the array itself
            # Correction for extend: it should be a list of arrays, then vstack OR append rows
            # Let's re-do the loop for all_probs_for_metrics_list for clarity
            pass  # Revisit all_probs_np formation after correcting the loop

        # --- Corrected loop and all_probs_np formation --- START
        all_preds_list_corrected = []
        all_labels_list_corrected = []
        all_probs_for_metrics_list_corrected = (
            []
        )  # This will be a list of appropriate items

        self.model.eval()
        with torch.no_grad():
            for inputs, labels_batch in self.test_loader:
                inputs, labels_batch = inputs.to(self.device), labels_batch.to(
                    self.device
                )
                outputs = self.model(inputs)
                logits_for_eval = (
                    outputs.logits if hasattr(outputs, "logits") else outputs
                )

                current_batch_preds_corrected = None

                if self.is_binary_classification:
                    if self.model_architecture == "resnet":
                        probs_class_1 = torch.sigmoid(logits_for_eval).squeeze(-1)
                        current_batch_preds_corrected = (probs_class_1 > 0.5).float()
                        all_probs_for_metrics_list_corrected.extend(
                            probs_class_1.cpu().numpy()
                        )  # Extends with individual P(class_1) floats
                    else:  # ViT binary
                        probs_all_classes = torch.softmax(logits_for_eval, dim=1)
                        current_batch_preds_corrected = torch.argmax(
                            probs_all_classes, dim=1
                        )
                        all_probs_for_metrics_list_corrected.extend(
                            probs_all_classes[:, 1].cpu().numpy()
                        )  # Extends with individual P(class_1) floats
                else:  # Multiclass
                    probs_all_classes = torch.softmax(logits_for_eval, dim=1)
                    current_batch_preds_corrected = torch.argmax(
                        probs_all_classes, dim=1
                    )
                    # For multiclass, store the entire probability array for each sample
                    for i in range(probs_all_classes.shape[0]):
                        all_probs_for_metrics_list_corrected.append(
                            probs_all_classes[i].cpu().numpy()
                        )  # Append each sample's prob array

                all_preds_list_corrected.extend(
                    current_batch_preds_corrected.cpu().numpy()
                )
                all_labels_list_corrected.extend(labels_batch.cpu().numpy())

        all_preds = np.array(all_preds_list_corrected)
        all_labels = np.array(all_labels_list_corrected)

        if self.is_binary_classification:
            all_probs_np = np.array(
                all_probs_for_metrics_list_corrected
            )  # Shape (n_samples,)
        else:  # Multiclass
            if all_probs_for_metrics_list_corrected:  # Check if not empty
                all_probs_np = np.vstack(
                    all_probs_for_metrics_list_corrected
                )  # Shape (n_samples, n_classes)
            else:
                all_probs_np = np.array([])  # Empty array if no samples
        # --- Corrected loop and all_probs_np formation --- END

        unique_true_and_pred_labels = sorted(
            list(np.unique(np.concatenate((all_labels, all_preds))))
        )
        # print(
        #     f"DEBUG: Unique labels found in true and preds: {unique_true_and_pred_labels}"
        # )

        target_names_for_report = []
        if self.is_binary_classification:
            temp_binary_names = {}
            if 0 in unique_true_and_pred_labels:
                temp_binary_names[0] = self.class_names[0]
            if 1 in unique_true_and_pred_labels:
                temp_binary_names[1] = self.class_names[1]
            target_names_for_report = [
                temp_binary_names[label]
                for label in unique_true_and_pred_labels
                if label in temp_binary_names
            ]
            if not target_names_for_report and unique_true_and_pred_labels:
                print(
                    f"WARNING: Binary classification with unusual labels {unique_true_and_pred_labels}. Using generic target names."
                )
                target_names_for_report = [
                    f"ActualClass_{i}" for i in unique_true_and_pred_labels
                ]
        else:  # Multiclass
            target_names_for_report = []
            for label_val in unique_true_and_pred_labels:
                if 0 <= label_val < len(self.class_names):
                    target_names_for_report.append(self.class_names[label_val])
                else:
                    target_names_for_report.append(f"ActualClass_{label_val}")

        if len(target_names_for_report) != len(unique_true_and_pred_labels):
            print(
                f"CRITICAL WARNING: Length mismatch after generating target_names_for_report. unique_labels: {unique_true_and_pred_labels}, generated_names: {target_names_for_report}"
            )
            target_names_for_report = [
                f"FallbackClass_{i}" for i in unique_true_and_pred_labels
            ]

        # print(
        #     f"Debug: Classification report will use target_names: {target_names_for_report} and labels: {unique_true_and_pred_labels}"
        # )
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=target_names_for_report,
            labels=unique_true_and_pred_labels,
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

        if self.is_binary_classification:
            if all_probs_np.ndim == 1 and len(all_probs_np) == len(all_labels):
                try:
                    metrics["roc_auc"] = roc_auc_score(all_labels, all_probs_np)
                    fpr, tpr, _ = roc_curve(all_labels, all_probs_np)
                    self.plot_roc_curve(fpr, tpr, metrics["roc_auc"], "roc_curve.png")

                    precision, recall, _ = precision_recall_curve(
                        all_labels, all_probs_np
                    )
                    metrics["pr_auc"] = auc(recall, precision)
                    self.plot_pr_curve(
                        precision, recall, metrics["pr_auc"], "pr_curve.png"
                    )
                except Exception as e:
                    print(
                        f"Error calculating/plotting binary ROC/PR for {self.model_filename}: {e}"
                    )
            else:
                print(
                    f"Skipping binary ROC/PR for {self.model_filename} due to probability array shape mismatch. Probs shape: {all_probs_np.shape if isinstance(all_probs_np, np.ndarray) else 'not an ndarray'}"
                )
        elif self.num_classes > 1:  # Multiclass (check all_probs_np shape)
            if (
                all_probs_np.ndim == 2
                and all_probs_np.shape[0] == len(all_labels)
                and all_probs_np.shape[1] == self.num_classes
            ):
                try:
                    metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                        all_labels, all_probs_np, multi_class="ovr", average="weighted"
                    )
                except ValueError as e:
                    print(
                        f"Could not calculate OvR ROC AUC for {self.model_filename}: {e}"
                    )
            else:
                print(
                    f"Skipping multiclass ROC_AUC for {self.model_filename} due to probability array shape mismatch. Probs shape: {all_probs_np.shape if isinstance(all_probs_np, np.ndarray) else 'not an ndarray'}"
                )

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


def append_to_evaluation_csv(metrics_data, csv_filepath):
    """Appends a dictionary of metrics to a CSV file."""
    fieldnames = [
        "model_filename",
        "evaluation_run_path",
        "model_architecture",
        "classification_type",
        "num_classes_model",
        "epochs",
        "epochs_labeled",
        "epochs_combined",
        "learning_rate",
        "learning_rate_config",
        "batch_size",
        "loss_function_type",
        "imbalance_strategy",
        "label_fraction",
        "training_option_resnet_e2",
        "num_layers_trained_excluding_fc",
        "data_augmentation",
        "finetune_bn",
        "l2_lambda",
        "monitor_gradients",
        "gradient_monitor_interval",
        "vit_model_checkpoint",
        "training_type",  # General training type (supervised, semi-supervised, etc.)
        "training_time_seconds",
        "accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1_score",
        "roc_auc",
        "pr_auc",
        "roc_auc_ovr_weighted",
    ]

    row_to_write = {}
    # Populate row_to_write from metrics_data, using .get() for safety
    for key in fieldnames:
        row_to_write[key] = metrics_data.get(key)

        # Specific handling for metrics from classification_report
        report = metrics_data.get("classification_report", {})
    weighted_avg = report.get("weighted avg", {}) if isinstance(report, dict) else {}
    row_to_write["weighted_precision"] = weighted_avg.get("precision")
    row_to_write["weighted_recall"] = weighted_avg.get("recall")
    row_to_write["weighted_f1_score"] = weighted_avg.get("f1-score")

    # Handle learning rate specifically if it's stored as a list in config
    lr_config = metrics_data.get(
        "learning_rate_config", metrics_data.get("learning_rate")
    )
    if isinstance(lr_config, list):
        row_to_write["learning_rate_config"] = str(lr_config)
        if not metrics_data.get("learning_rate"):  # If single LR wasn't also stored
            row_to_write["learning_rate"] = lr_config[0] if lr_config else None
    elif lr_config is not None:  # It's a single LR value
        row_to_write["learning_rate"] = lr_config
        row_to_write["learning_rate_config"] = None

    file_exists = os.path.isfile(csv_filepath)
    try:
        with open(csv_filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if (
                not file_exists or os.path.getsize(csv_filepath) == 0
            ):  # Check if file is empty
                writer.writeheader()
            writer.writerow(row_to_write)
        print(
            f"Evaluation results for {metrics_data.get('model_filename')} appended to {csv_filepath}"
        )
    except IOError as e:
        print(f"Error writing to CSV {csv_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing to CSV: {e}")


def perform_and_save_evaluation(
    model_path,
    model_eval_output_base_path,
    experiment_params_dict,
    training_time=None,
    device_str="cpu",
):
    """
    Performs evaluation for a given model, saves results in a unique directory,
    and appends key metrics to a global CSV file.

    Args:
        model_path (str): Path to the saved .pth model file.
        model_eval_output_base_path (str): The base path for the evaluation output directory.
                                           Example: ../evaluation/resnet/binary/resnet_binary_10ep_lr0.001_sup
                                           A suffix like '_1', '_2' will be added if it already exists.
        experiment_params_dict (dict): Dictionary containing parameters of the experiment.
        training_time (float, optional): Training time in seconds. Defaults to None.
        device_str (str, optional): Device to use ('cuda' or 'cpu'). Defaults to "cpu".
    """
    print(f"\n--- Starting evaluation for model: {model_path} ---")
    print(
        f"Experiment params: {json.dumps(experiment_params_dict, indent=2, cls=NpEncoder)}"
    )
    if training_time is not None:
        print(f"Training time: {training_time:.2f}s")

    # 1. Determine unique evaluation directory
    eval_run_dir = model_eval_output_base_path
    counter = 1
    # Check if the base path itself (without _N) exists and is a directory.
    # If it exists and is NOT a directory, or if we need a new numbered one.
    while os.path.exists(eval_run_dir):
        if os.path.isdir(eval_run_dir) and not os.listdir(
            eval_run_dir
        ):  # If it's an empty dir, use it
            print(f"Found existing empty directory, will use: {eval_run_dir}")
            break
        eval_run_dir = f"{model_eval_output_base_path}_{counter}"
        counter += 1

    try:
        os.makedirs(eval_run_dir, exist_ok=True)
        print(f"Evaluation run directory set to: {eval_run_dir}")
    except OSError as e:
        print(f"Error creating directory {eval_run_dir}: {e}")
        return

    # 2. Instantiate ModelEvaluator
    # Determine project root and data_dir relative to this script's location (src/)
    # Assuming this script (evaluation.py) is in 'src' folder
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
    data_dir = os.path.join(project_root, "data", "raw")

    # Default batch size, can be overridden by experiment_params_dict
    batch_size = experiment_params_dict.get("batch_size", 32)

    try:
        evaluator = ModelEvaluator(
            model_path=model_path,
            device=torch.device(device_str),
            data_dir=data_dir,
            batch_size=batch_size,
            evaluation_run_specific_dir=eval_run_dir,
        )

        # 3. Run evaluation
        metrics = evaluator.evaluate()  # This saves plots and json to eval_run_dir
    except Exception as e:
        print(
            f"ERROR during ModelEvaluator instantiation or evaluation for {model_path}: {e}"
        )
        import traceback

        traceback.print_exc()
        return  # Stop if evaluation fails

    if metrics:
        # 4. Augment metrics with experiment params and training time for CSV
        metrics_for_csv = metrics.copy()
        metrics_for_csv.update(
            experiment_params_dict
        )  # Add/overwrite with experiment parameters

        if training_time is not None:
            metrics_for_csv["training_time_seconds"] = training_time

        # Add relative path to the evaluation run directory for the CSV
        try:
            # project_root for all_evaluations.csv is one level above src
            csv_project_root = os.path.abspath(
                os.path.join(project_root)
            )  # project_root IS the project root
            metrics_for_csv["evaluation_run_path"] = os.path.relpath(
                eval_run_dir, csv_project_root
            )
        except ValueError:  # If paths are on different drives on Windows
            metrics_for_csv["evaluation_run_path"] = (
                eval_run_dir  # Fallback to absolute path
            )

        # 5. Append to CSV
        # CSV path is PROJECT_ROOT/evaluation/all_evaluations.csv
        csv_filepath = os.path.join(project_root, "evaluation", "all_evaluations.csv")
        append_to_evaluation_csv(metrics_for_csv, csv_filepath)
        print(f"--- Evaluation completed for model: {model_path} ---")
    else:
        print(
            f"WARNING: Evaluation did not return metrics for {model_path}. CSV not updated."
        )


if __name__ == "__main__":
    print("--- Running Test Evaluation ---")

    # --- Configuration for the test ---
    # Absolute path to the model you want to test
    # IMPORTANT: Replace this with the correct ABSOLUTE path to your model if it's different
    # or ensure your script's execution context allows this relative path from project root.
    # Assuming evaluation.py is in src, and models is ../models from src.

    # Let's try to make the model path more robust for testing from src/
    current_script_dir_for_test = os.path.dirname(os.path.abspath(__file__))
    project_root_for_test = os.path.abspath(
        os.path.join(current_script_dir_for_test, "..")
    )

    test_model_relative_path = "models/resnet/binary/resnet_binary_1ep_lr0.001_sup.pth"
    test_model_full_path = os.path.join(project_root_for_test, test_model_relative_path)

    if not os.path.exists(test_model_full_path):
        print(f"ERROR: Test model not found at {test_model_full_path}")
        print("Please ensure the path is correct and the model file exists.")
    else:
        print(f"Attempting to evaluate model: {test_model_full_path}")

        # Base name for the evaluation output directory (without _1, _2 suffixes yet)
        # This should match the logic in main.py: ../evaluation/<arch>/<class_type>/<model_filename_base>
        model_filename_base_for_test = "resnet_binary_1ep_lr0.001_sup"
        test_eval_output_base_relative_path = (
            f"evaluation/resnet/binary/{model_filename_base_for_test}"
        )
        test_eval_output_base_full_path = os.path.join(
            project_root_for_test, test_eval_output_base_relative_path
        )

        # Experiment parameters mimicking how they would be in main.py for this run
        test_experiment_params = {
            "model_architecture": "resnet",
            "classification_type": "binary",
            "epochs": 1,
            "learning_rate": 0.001,  # or "learning_rate_config": [0.001]
            "training_type": "supervised",
            "monitor_gradients": False,
            # "gradient_monitor_interval": 100, # Not relevant if monitor_gradients is False
            "batch_size": 32,  # Assuming default
            "model_path": test_model_full_path,  # Path to the model being evaluated
            # Add any other parameters that would be in experiment_params for this specific run
            # and are expected by the CSV header or logic in ModelEvaluator/append_to_evaluation_csv
            "label_fraction": None,
            "imbalance_strategy": None,
            "data_augmentation": False,  # Common default
            "finetune_bn": True,  # Common default for ResNet supervised
            "l2_lambda": 0.0,  # Common default
        }

        test_training_time = 60.5  # Example training time in seconds

        # Determine device
        test_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for test: {test_device_str}")

        perform_and_save_evaluation(
            model_path=test_model_full_path,
            model_eval_output_base_path=test_eval_output_base_full_path,
            experiment_params_dict=test_experiment_params,
            training_time=test_training_time,
            device_str=test_device_str,
        )
        print("--- Test Evaluation Finished ---")
