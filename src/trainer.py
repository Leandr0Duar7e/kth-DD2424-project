import torch
import torch.nn as nn
import os
from tqdm import tqdm


class ModelTrainer:
    """
    A class to handle model training and evaluation

    Args:
        model: The PyTorch model to train
        device: The device to use for training (cuda or cpu)
        binary_classification: Whether the model is for binary classification
        learning_rate: Learning rate for the optimizer
    """

    def __init__(self, model, device, binary_classification=True, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.binary_classification = binary_classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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

    def train(self, train_loader, val_loader, num_epochs=10):
        """
        Train the model

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs

        Returns:
            Trained model and training history
        """
        print(
            f" Start Training {self.model.__class__.__name__} for {num_epochs} epochs"
        )

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

        return self.model, self.history

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
