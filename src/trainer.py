import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class ModelTrainer:
    """
    A class to handle model training and evaluation

    Args:
        model: The PyTorch model to train
        device: The device to use for training (cuda or cpu)
        binary_classification: Whether the model is for binary classification
        learning_rate: Learning rate for the optimizer
    """

    def __init__(self, model, device, binary_classification=True, learning_rate=[0.001], lam=0.0):
        self.model = model.to(device)
        self.device = device
        self.binary_classification = binary_classification
        
        param_groups = []
    
        if len(learning_rate) == 1:
            learning_rate = learning_rate[0]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            weighted_layers = self.model.get_index_weighted_layers()
            weighted_layers.reverse()
            
            backbone_layers = list(self.model.backbone.children())
            
            for i in range(len(weighted_layers)):
                param_groups.append({
                    'params': backbone_layers[weighted_layers[i]].parameters(),
                    'lr': learning_rate[i],
                    'weight_decay': lam
                })

            self.optimizer = torch.optim.Adam(param_groups)
        
        
        self.criterion = (
            nn.BCEWithLogitsLoss() if binary_classification else nn.CrossEntropyLoss()
        )
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset

        Args:
            data_loader: The DataLoader for the dataset to evaluate on

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_pred = 0
        n_sample = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)

                if self.binary_classification:
                    labels = labels.float().to(self.device)
                else:
                    labels = labels.long().to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)

                total_loss += loss.item()
                predicted = (
                    (torch.sigmoid(outputs) > 0.5).float()
                    if self.binary_classification
                    else outputs.argmax(dim=1)
                )
                n_sample += labels.size(0)
                correct_pred += (predicted.squeeze() == labels).sum().item()

        return total_loss / len(data_loader), 100 * correct_pred / n_sample

    def train(self, train_loader, val_loader, num_epochs=10, print_graph=False):
        """
        Train the model

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            print_graph: If True, plots training/validation loss and accuracy

        Returns:
            Trained model and training history
        """
        print(
            f" Start Training {self.model.__class__.__name__} for {num_epochs} epochs"
        )

        # Reset history for a new training session
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Use tqdm for a progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                # Convert labels to appropriate type based on classification type
                if self.binary_classification:
                    labels = labels.float().to(self.device)
                else:
                    labels = labels.long().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                predicted = (
                    (torch.sigmoid(outputs) > 0.5).float()
                    if self.binary_classification
                    else outputs.argmax(dim=1)
                )
                train_total += labels.size(0)
                train_correct += (predicted.squeeze() == labels).sum().item()

                # Update progress bar
                current_loss = train_loss / (progress_bar.n + 1)
                current_acc = 100 * train_correct / train_total
                progress_bar.set_postfix(
                    {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"}
                )

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Record metrics
            self.history["train_loss"].append(train_loss / len(train_loader))
            self.history["train_acc"].append(100 * train_correct / train_total)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Print epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(
                f'Train Loss: {self.history["train_loss"][-1]:.4f}, '
                f'Train Acc: {self.history["train_acc"][-1]:.2f}%'
            )
            print(
                f'Val Loss: {self.history["val_loss"][-1]:.4f}, '
                f'Val Acc: {self.history["val_acc"][-1]:.2f}%'
            )
            print("-" * 60)

        if print_graph:
            self._plot_history()

        return self.model, self.history

    def train_gradual_unfreezing(
        self, train_loader, val_loader, num_epochs=10, print_graph=False
    ):
        """
        Train the model with gradual unfreezing of layers.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            print_graph: If True, plots training/validation loss and accuracy

        Returns:
            Trained model and training history
        """
        print(
            f" Start Training {self.model.__class__.__name__} with Gradual Unfreezing for {num_epochs} epochs"
        )

        # Reset history for a new training session
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        tot_num_steps = len(train_loader) * num_epochs
        print(
            f"Total number of steps: {tot_num_steps}, length of train_loader: {len(train_loader)}"
        )

        # Identify layers in the backbone (excluding the final classifier layer)
        # Assuming self.model has a 'backbone' attribute as in ResNet50 class
        if hasattr(self.model, "backbone") and self.model.backbone is not None:
            # children() gives direct children; modules() gives all recursively.
            # We want the main blocks of the backbone, typically direct children.
            all_layers_in_backbone = list(self.model.backbone.children())[
                :-1
            ]  # Exclude the final fc layer which is part of backbone

            # Filter for layers that have parameters (e.g., Conv2d, BatchNorm, Linear)
            # Some children might be containers like Sequential, so we need to be careful
            # This part might need adjustment based on the exact structure of ResNet backbone layers
            param_layers_indices = []
            actual_param_layers = []
            for i, layer_module in enumerate(all_layers_in_backbone):
                if list(
                    layer_module.parameters()
                ):  # Check if the module itself has parameters
                    param_layers_indices.append(i)
                    actual_param_layers.append(layer_module)

            
            print(
                f"Identified {len(actual_param_layers)} parameter-containing layer groups in backbone to unfreeze gradually."
            )

            if (
                not actual_param_layers or len(actual_param_layers) <= 1
            ):  # Need at least 2 layers to unfreeze one by one excluding classifier
                print(
                    "Warning: Not enough layers in backbone for gradual unfreezing or backbone structure not as expected. Proceeding with normal training."
                )
                return self.train(train_loader, val_loader, num_epochs, print_graph)

            # Ensure the final FC layer of the model (not just backbone) is trainable
            if hasattr(self.model.backbone, "fc"):
                for param in self.model.backbone.fc.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, "fc"):  # If fc is a direct attribute of the model
                for param in self.model.fc.parameters():
                    param.requires_grad = True

            # Initially freeze all backbone layers except the classifier (which should already be trainable or handled by ResNet50 class)
            for layer_module in actual_param_layers:
                for param in layer_module.parameters():
                    param.requires_grad = False

            unfreeze_step_interval = tot_num_steps // len(
                actual_param_layers
            )  # unfreeze one layer group per interval
            unfreeze_layer_group_idx_to_unfreeze = (
                len(actual_param_layers) - 1
            )  # Start from the layer group closest to classifier

        else:
            print(
                "Warning: Model does not have a 'backbone' attribute. Gradual unfreezing cannot be applied. Proceeding with normal training."
            )
            return self.train(train_loader, val_loader, num_epochs, print_graph)

        step_counter = 0
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Gradual Unfreeze)"
            )

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.binary_classification:
                    labels = labels.float()
                else:
                    labels = labels.long()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                predicted = (
                    (torch.sigmoid(outputs) > 0.5).float()
                    if self.binary_classification
                    else outputs.argmax(dim=1)
                )
                train_total += labels.size(0)
                train_correct += (predicted.squeeze() == labels).sum().item()

                current_loss = train_loss / (progress_bar.n + 1)
                current_acc = 100 * train_correct / train_total
                progress_bar.set_postfix(
                    {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"}
                )

                step_counter += 1
                if (
                    hasattr(self.model, "backbone")
                    and self.model.backbone is not None
                    and unfreeze_layer_group_idx_to_unfreeze >= 0
                    and step_counter % unfreeze_step_interval == 0
                ):
                    layer_to_unfreeze = actual_param_layers[
                        unfreeze_layer_group_idx_to_unfreeze
                    ]
                    print(
                        f"\nUnfreezing layer group {unfreeze_layer_group_idx_to_unfreeze+1}/{len(actual_param_layers)} (step {step_counter}) name: {layer_to_unfreeze.__class__.__name__}"
                    )
                    for param in layer_to_unfreeze.parameters():
                        param.requires_grad = True
                    # Re-create optimizer to include newly unfrozen parameters if its Adam, Adagrad etc.
                    # For SGD it might not be strictly necessary but good practice.
                    self.optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                    unfreeze_layer_group_idx_to_unfreeze -= 1

            val_loss, val_acc = self.evaluate(val_loader)
            self.history["train_loss"].append(train_loss / len(train_loader))
            self.history["train_acc"].append(100 * train_correct / train_total)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch+1}/{num_epochs}: Train Loss: {self.history['train_loss'][-1]:.4f}, Train Acc: {self.history['train_acc'][-1]:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            print("-" * 60)

        if print_graph:
            self._plot_history()

        return self.model, self.history

    def _plot_history(self):
        if not self.history["train_loss"]:  # Check if history is empty
            print("No training history to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history["train_loss"], label="Train Loss")
        ax1.plot(self.history["val_loss"], label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.history["train_acc"], label="Train Accuracy")
        ax2.plot(self.history["val_acc"], label="Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        # Try to show plot, but handle environments where it might fail (e.g., headless server)
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot (matplotlib.pyplot.show() failed): {e}")
            # Optionally save the plot to a file if showing fails
            plot_filename = "training_plot.png"
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        plt.close(fig)  # Close the figure to free memory

    def save_model(self, model_type="binary"):
        """
        Save the trained model

        Args:
            model_type: Type of model ('binary', 'multiclass', or 'pretrained')
        """
        # Create the directory if it doesn't exist
        base_dir = "../models/resnet"
        save_dir = os.path.join(base_dir, model_type)
        os.makedirs(save_dir, exist_ok=True)

        # Save the model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "binary_classification": self.binary_classification,
            },
            os.path.join(save_dir, f"resnet50_{model_type}_classifier.pth"),
        )

        print(f"Model saved to {save_dir}/resnet50_{model_type}_classifier.pth")
