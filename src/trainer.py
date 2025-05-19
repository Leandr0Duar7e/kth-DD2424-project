import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset


class ModelTrainer:
    """
    A class to handle model training and evaluation

    Args:
        model: The PyTorch model to train
        device: The device to use for training (cuda or cpu)
        binary_classification: Whether the model is for binary classification
        learning_rate: Learning rate for the optimizer
        monitor_gradients (bool): If True, logs gradient norms during training.
        gradient_monitor_interval (int): Interval (in batches) for logging gradient norms.
        finetune_bn (bool): If True, batch normalization parameters will be fine-tuned during training.
        use_scheduler (bool): If True, uses OneCycleLR scheduler with warm-up and cosine annealing.
        scheduler_params (dict): Parameters for the scheduler:
            - max_lr: Maximum learning rate to reach during training
            - pct_start: Percentage of training for warm-up (default: 0.3)
    """

    def __init__(
        self,
        model,
        device,
        binary_classification=True,
        learning_rate=[5e-5],
        loss_fn=None,
        lam=0.0,
        monitor_gradients=False,
        gradient_monitor_interval=100,
        finetune_bn=True,
        use_scheduler=False,
        scheduler_params=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.binary_classification = binary_classification
        self.loss_fn = loss_fn
        self.finetune_bn = finetune_bn
        self.learning_rates = learning_rate
        self.weight_decay = lam
        self.use_scheduler = use_scheduler
        self.scheduler_params = scheduler_params or {}

        if self.loss_fn is None:
            if self.binary_classification:
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()

        # Apply batch normalization freezing if specified
        if not self.finetune_bn:
            self._freeze_bn_params()

        if len(learning_rate) == 1:
            learning_rate = learning_rate[0]
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=lam
            )

        else:
            weighted_layers = self.model.get_index_weighted_layers(
                finetune_bn=finetune_bn
            )
            weighted_layers.reverse()

            backbone_layers = list(self.model.backbone.children())

            param_groups = []

            for i in range(len(backbone_layers)):
                layer_params = list(backbone_layers[i].parameters())

                if i in weighted_layers:

                    param_groups.append(
                        {
                            "params": layer_params,
                            "lr": learning_rate[weighted_layers.index(i)],
                            "weight_decay": lam,
                        }
                    )

                else:
                    param_groups.append(
                        {
                            "params": layer_params,
                            "lr": 1e-5,
                            "weight_decay": lam,
                        }
                    )

            self.optimizer = torch.optim.Adam(param_groups)

        # Initialize scheduler if requested (will be properly initialized in train())
        self.scheduler = None

        self.criterion = (
            nn.BCEWithLogitsLoss() if binary_classification else nn.CrossEntropyLoss()
        )
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }
        self.monitor_gradients = monitor_gradients
        self.gradient_monitor_interval = gradient_monitor_interval

    def _freeze_bn_params(self):
        """Freeze batch normalization parameters in the model."""
        print("Freezing batch normalization parameters")
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()  # Set BN layers to evaluation mode

                for param in module.parameters():
                    param.requires_grad = False

            for name, param in module.named_parameters():
                if "bn" in name:
                    param.requires_grad = False

    def _log_gradient_norms(self, epoch, batch_idx):
        """Logs the L2 norm of gradients for each parameter and the total norm."""
        print(f"--- Gradient Norms at Epoch {epoch+1}, Batch {batch_idx+1} ---")
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                param_norm = param.grad.norm(2).item()
                total_norm += param_norm**2
                print(f"  Param: {name:<50} | Grad Norm: {param_norm:.4e}")
            elif param.requires_grad:
                print(f"  Param: {name:<50} | Grad Norm: None (param.grad is None)")
        total_norm = total_norm**0.5
        print(f"  Total gradient norm for all trainable parameters: {total_norm:.4e}")
        print(f"--- End of Gradient Norms ---")

    def combine_loaders(self, labeled_loader, pseudo_loader):
        """
        Combine labeled and pseudo-labeled data into one DataLoader
        """
        combined_dataset = ConcatDataset(
            [labeled_loader.dataset, pseudo_loader.dataset]
        )
        return DataLoader(combined_dataset, batch_size=32, shuffle=True)

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

        if not self.finetune_bn:
            print("Batch normalization parameters are frozen (not being fine-tuned)")

        if self.use_scheduler:
            print("Using OneCycleLR scheduler with warm-up and cosine annealing")
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * num_epochs
            
            # Handle multiple learning rates
            if isinstance(self.learning_rates, list) and len(self.learning_rates) > 1:
                print("Multiple learning rates detected for different parameter groups")
                # Get the base learning rates for each parameter group
                base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                # Calculate max_lrs for each group (10x their base learning rate)
                max_lrs = [lr * 10 for lr in base_lrs]
                
                print("Learning rate ranges for each parameter group:")
                for i, (base_lr, max_lr) in enumerate(zip(base_lrs, max_lrs)):
                    print(f"Group {i}: {base_lr:.2e} -> {max_lr:.2e}")
            else:
                # Single learning rate case
                base_lr = self.optimizer.param_groups[0]['lr']
                max_lr = self.scheduler_params.get('max_lr', base_lr * 10)
                max_lrs = [max_lr]  # OneCycleLR expects a list even for single lr
                print(f"Base learning rate: {base_lr:.2e}")
                print(f"Maximum learning rate: {max_lr:.2e}")
            
            pct_start = self.scheduler_params.get('pct_start', 0.3)
            print(f"Warm-up percentage: {pct_start*100:.0f}%")
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,  # Now handles both single and multiple learning rates
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0,
            )

        # Reset history for a new training session
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        for epoch in range(num_epochs):
            # Training
            if self.finetune_bn:
                self.model.train()  # Set model to training mode (including BN layers)
            else:
                # If not fine-tuning BN, we need to set the model to train mode but keep BN in eval mode
                self.model.train()
                # Re-freeze BN layers as model.train() would have set them to train mode
                for module in self.model.modules():
                    if isinstance(module, nn.BatchNorm2d) or isinstance(
                        module, nn.BatchNorm1d
                    ):
                        module.eval()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Use tqdm for a progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (inputs, labels) in enumerate(progress_bar):
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

                # Monitor gradient norms
                if (
                    self.monitor_gradients
                    and (batch_idx + 1) % self.gradient_monitor_interval == 0
                ):
                    self._log_gradient_norms(epoch, batch_idx)

                self.optimizer.step()

                # Step the scheduler (per batch)
                if self.use_scheduler:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.history["learning_rates"].append(current_lr)
                    #if batch_idx % 10 == 0:  # Print every 10 batches to avoid too much output
                    #progress_bar.set_postfix({"lr": f"{current_lr:.2e}"})

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
                    {
                        "current_loss": f"{current_loss:.4f}",
                        "current_acc": f"{current_acc:.2f}%",
                        "lr": f"{current_lr:.2e}",
                    }
                )

            # Evaluate on train set
            train_loss, train_acc = self.evaluate(train_loader)

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Record metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
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

        if not self.finetune_bn:
            print("Batch normalization parameters are frozen (not being fine-tuned)")

        # Initialize scheduler if requested
        if self.use_scheduler:
            print("Using OneCycleLR scheduler with warm-up and cosine annealing")
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * num_epochs
            
            # Handle multiple learning rates
            if isinstance(self.learning_rates, list) and len(self.learning_rates) > 1:
                print("Multiple learning rates detected for different parameter groups")
                # Get the base learning rates for each parameter group
                base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                # Calculate max_lrs for each group (10x their base learning rate)
                max_lrs = [lr * 10 for lr in base_lrs]
                
                print("Learning rate ranges for each parameter group:")
                for i, (base_lr, max_lr) in enumerate(zip(base_lrs, max_lrs)):
                    print(f"Group {i}: {base_lr:.2e} -> {max_lr:.2e}")
            else:
                # Single learning rate case
                base_lr = self.optimizer.param_groups[0]['lr']
                max_lr = self.scheduler_params.get('max_lr', base_lr * 10)
                max_lrs = [max_lr]  # OneCycleLR expects a list even for single lr
                print(f"Base learning rate: {base_lr:.2e}")
                print(f"Maximum learning rate: {max_lr:.2e}")
            
            pct_start = self.scheduler_params.get('pct_start', 0.3)
            print(f"Warm-up percentage: {pct_start*100:.0f}%")
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0,
            )

        # Reset history for a new training session
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],  # Add this to track learning rates
        }

        tot_num_steps = len(train_loader) * num_epochs

        # Identify layers in the backbone (excluding the final classifier layer)
        if hasattr(self.model, "backbone") and self.model.backbone is not None:

            all_layers_in_backbone = list(self.model.backbone.children())[
                :-1
            ]  # Exclude the final fc layer

            param_layers_indices = self.model.get_index_weighted_layers(
                self.finetune_bn
            )[:-1]

            actual_param_layers = []

            for i in param_layers_indices:
                actual_param_layers.append(all_layers_in_backbone[i])

            print(
                f"Identified {len(actual_param_layers)} parameter-containing layer groups in backbone to unfreeze gradually (fc already unfrozen)."
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

            # Redundant but just to be sure
            for layer_module in actual_param_layers:
                for param in layer_module.parameters():
                    param.requires_grad = False

            unfreeze_step_interval = tot_num_steps // (
                len(actual_param_layers) + 1
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
            if self.finetune_bn:
                self.model.train()  # Set model to training mode (including BN layers)
            else:
                # If not fine-tuning BN, we need to set the model to train mode but keep BN in eval mode
                self.model.train()

                # Re-freeze BN layers as model.train() would have set them to train mode
                self._freeze_bn_params()

            train_loss = 0.0
            train_correct = 0
            train_total = 0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Gradual Unfreeze)"
            )

            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.binary_classification:
                    labels = labels.float()
                else:
                    labels = labels.long()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()

                # Monitor gradient norms
                if (
                    self.monitor_gradients
                    and (batch_idx + 1) % self.gradient_monitor_interval == 0
                ):
                    self._log_gradient_norms(epoch, batch_idx)

                self.optimizer.step()

                # Step the scheduler (per batch)
                if self.use_scheduler:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.history["learning_rates"].append(current_lr)
                    
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
                    {
                        "curr_loss": f"{current_loss:.4f}",
                        "curr_acc": f"{current_acc:.2f}%",
                    }
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
                        f"\nUnfreezing layer group idx:{unfreeze_layer_group_idx_to_unfreeze} (step {step_counter}) name: {layer_to_unfreeze.__class__.__name__}"
                    )

                    for name, param in layer_to_unfreeze.named_parameters():
                        if not param.requires_grad:
                            if not self.finetune_bn:
                                # BatchNorm parameter names in PyTorch always include "bn" by convention
                                if "bn" not in name:
                                    param.requires_grad = True
                            else:
                                param.requires_grad = True

                    unfreeze_layer_group_idx_to_unfreeze -= 1

            # Evaluate on train set
            train_loss, train_acc = self.evaluate(train_loader)
            # Evaluate on val set
            val_loss, val_acc = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch+1}/{num_epochs}: Train Loss: {self.history['train_loss'][-1]:.4f}, Train Acc: {self.history['train_acc'][-1]:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            print("-" * 60)

        if print_graph:
            self._plot_history()

        return self.model, self.history

    def generate_pseudo_labels(self, model, unlabeled_loader):
        """
        Generate pseudo-labels for unlabeled data
        """
        model.eval()
        pseudo_images, pseudo_labels = [], []

        with torch.no_grad():
            for images, _ in unlabeled_loader:
                images = images.to(self.device)
                outputs = model(images)

                preds = (
                    (torch.sigmoid(outputs) > 0.5).float().squeeze()
                    if self.binary_classification
                    else outputs.argmax(dim=1)
                )

                pseudo_images.extend(images.cpu())
                pseudo_labels.extend(preds.cpu())

        dataset = torch.utils.data.TensorDataset(
            torch.stack(pseudo_images),
            torch.tensor(
                pseudo_labels,
                dtype=torch.float32 if self.binary_classification else torch.long,
            ),
        )
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def _plot_history(self):
        if not self.history["train_loss"]:  # Check if history is empty
            print("No training history to plot.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

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

        # Plot learning rate
        if self.use_scheduler and self.history["learning_rates"]:
            ax3.plot(self.history["learning_rates"], label="Learning Rate")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Learning Rate Schedule")
            ax3.legend()
            ax3.grid(True)
            ax3.set_yscale('log')  # Use log scale for better visualization

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

    def save_model(
        self, model_type="binary", model_architecture="resnet", full_save_path=None
    ):
        """
        Save the trained model

        Args:
            model_type: Type of model ('binary', 'multiclass')
            model_architecture: Architecture of the model ('resnet', 'vit')
            full_save_path: Optional. Full path (including filename) to save the model.
                           If None, constructs path based on model_type and model_architecture.
        """
        save_path = full_save_path
        if save_path:
            # Ensure the directory for the custom path exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            # Fallback to original behavior if full_save_path is not provided
            print(
                "Warning: full_save_path not provided to save_model. Using default naming and location."
            )
            base_dir = os.path.join(
                "..", "models", model_architecture
            )  # Use os.path.join for robustness
            save_dir = os.path.join(base_dir, model_type)
            os.makedirs(save_dir, exist_ok=True)
            model_filename = f"{model_architecture}_{model_type}_classifier_fallback.pth"  # Added fallback to distinguish
            save_path = os.path.join(save_dir, model_filename)

        # Save the model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "binary_classification": self.binary_classification,
                "finetune_bn": self.finetune_bn,  # Save BN fine-tuning setting
                "model_architecture": model_architecture,
                "model_name_or_path": getattr(self.model, "model_name_or_path", None),
            },
            save_path,
        )

        print(f"Model saved to {save_path}")
