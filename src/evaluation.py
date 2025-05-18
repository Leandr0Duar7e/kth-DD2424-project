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
from models.vit import VisionTransformerModel


class ModelEvaluator:
    def __init__(
        self, model_path, device, data_dir, batch_size=32, results_dir="../evaluation"
    ):
        self.model_path = model_path
        self.device = device
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.results_dir = (
            results_dir  # This is EVALUATION_ROOT_DIR, e.g., ../evaluation
        )
        os.makedirs(self.results_dir, exist_ok=True)

        self.checkpoint = torch.load(model_path, map_location=device)
        self.model_architecture = self.checkpoint.get("model_architecture", "unknown")
        # self.model_type derived from binary_classification flag later
        self.is_binary_classification = self.checkpoint.get(
            "binary_classification", True
        )
        self.model_name_or_path = self.checkpoint.get("model_name_or_path")  # For ViT

        self.model_filename = os.path.basename(model_path)
        # Subdirectory for this specific model's evaluation files
        self.evaluation_subdir = os.path.join(
            self.results_dir, os.path.splitext(self.model_filename)[0]
        )
        os.makedirs(self.evaluation_subdir, exist_ok=True)

        self.num_classes = 0  # Will be set in _load_model
        self.class_names = []  # Will be set in _get_test_dataloader

        self.model = self._load_model()  # Sets self.num_classes
        self.test_loader = (
            self._get_test_dataloader()
        )  # Sets self.class_names based on num_classes and dataset

    def _load_model(self):
        # Infer num_classes from the model's state_dict's final layer
        last_layer_weights = None
        state_dict = self.checkpoint.get("model_state_dict")
        if not state_dict:
            raise ValueError(
                f"model_state_dict not found in checkpoint: {self.model_path}"
            )

        if self.model_architecture == "resnet":
            if "fc.weight" in state_dict:
                last_layer_weights = state_dict["fc.weight"]
        elif self.model_architecture == "vit":
            possible_keys = [
                "classifier.weight",
                "vit.classifier.weight",
                "head.weight",
            ]  # Common ViT classifier layer names
            for key in possible_keys:
                if key in state_dict:
                    last_layer_weights = state_dict[key]
                    break

        if last_layer_weights is None:
            raise ValueError(
                f"Could not determine the number of classes from the model checkpoint's final layer: {self.model_path}"
            )

        self.num_classes = last_layer_weights.size(0)

        if self.model_architecture == "resnet":
            model = ResNet50(num_classes=self.num_classes, pretrained=False)
        elif self.model_architecture == "vit":
            if not self.model_name_or_path:
                # Fallback for older models that might not have saved model_name_or_path
                # Defaulting based on common ViT if binary/multiclass output matches. This is a guess.
                print(
                    f"Warning: ViT model_name_or_path not found in checkpoint {self.model_path}. Using default 'google/vit-base-patch16-224'."
                )
                self.model_name_or_path = "google/vit-base-patch16-224"
            model = VisionTransformerModel(
                model_name_or_path=self.model_name_or_path,
                num_classes=self.num_classes,
                pretrained_weights=False,  # We load from checkpoint
            )
        else:
            raise ValueError(f"Unknown model architecture: {self.model_architecture}")

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _get_test_dataloader(self):
        # Ensure data_augmentation is False for testing.
        # Use a consistent random_seed for the test split.
        _, _, test_loader = OxfordPetDataset.get_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            binary_classification=self.is_binary_classification,
            data_augmentation=False,
            model_type=self.model_architecture,
            vit_model_name=(
                self.model_name_or_path
                if self.model_architecture == "vit"
                else "google/vit-base-patch16-224"
            ),  # Default for non-ViT call
            random_seed=42,  # Crucial for consistent test set
        )

        # Determine class names
        # test_loader.dataset is a Subset, test_loader.dataset.dataset is the OxfordPetDataset instance
        full_dataset = test_loader.dataset.dataset
        if self.is_binary_classification:
            self.class_names = ["Cat", "Dog"]  # Explicit for binary task
            if self.num_classes != 2:  # Sanity check
                print(
                    f"Warning: Binary classification flag is True, but model output units are {self.num_classes}. Adjusting class names based on model output."
                )
                self.class_names = [f"Class {i}" for i in range(self.num_classes)]

        elif hasattr(full_dataset, "classes") and full_dataset.classes:
            self.class_names = full_dataset.classes
            if len(self.class_names) != self.num_classes:  # Sanity check
                print(
                    f"Warning: Mismatch between dataset classes ({len(self.class_names)}) and model output units ({self.num_classes}). Using model's num_classes."
                )
                self.class_names = [
                    f"Class {i}" for i in range(self.num_classes)
                ]  # Fallback
        else:
            self.class_names = [
                f"Class {i}" for i in range(self.num_classes)
            ]  # Generic fallback

        return test_loader

    def evaluate(self):
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

        # Ensure class_names for classification_report matches num_classes
        report_class_names = self.class_names
        if len(self.class_names) != self.num_classes:
            print(
                f"Adjusting class names for report: using {self.num_classes} generic names as there's a mismatch."
            )
            report_class_names = [f"Class {i}" for i in range(self.num_classes)]

        accuracy = np.mean(all_preds == all_labels)
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=report_class_names,
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
            "accuracy": accuracy,
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
            all_labels, all_preds, report_class_names, "confusion_matrix.png"
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

    DATA_DIR = os.path.join(project_root, "data", "raw", "oxford-iiit-pet")
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
