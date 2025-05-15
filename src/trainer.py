import torch
import torch.nn as nn
from models.resnet import get_ResNet50_model
from dataset import get_oxford_pet_dataloaders
import os


def evaluate_model(model, device, test_loader, criterion, binary_classification):
    model.eval()
    total_loss = 0.0
    correct_pred = 0
    n_sample = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            if binary_classification:
                labels = labels.float().to(device)
            else:
                labels = labels.long().to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            total_loss += loss.item()
            predicted = (
                (torch.sigmoid(outputs) > 0.5).float()
                if binary_classification
                else outputs.argmax(dim=1)
            )
            n_sample += labels.size(0)
            correct_pred += (predicted.squeeze() == labels).sum().item()

    return total_loss / len(test_loader), 100 * correct_pred / n_sample


def train_model_with_adam(
    model,
    device,
    train_loader,
    val_loader,
    binary_classification,
    num_epochs=10,
    learning_rate=0.001,
):

    # Set up model and training
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Use appropriate loss function based on classification type
    criterion = (
        nn.BCEWithLogitsLoss() if binary_classification else nn.CrossEntropyLoss()
    )

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop
    print(f" Start Training {model.__class__.__name__} for {num_epochs} epochs")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            # Convert labels to appropriate type based on classification type
            if binary_classification:
                labels = labels.float().to(device)
            else:
                labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (
                (torch.sigmoid(outputs) > 0.5).float()
                if binary_classification
                else outputs.argmax(dim=1)
            )
            train_total += labels.size(0)
            train_correct += (predicted.squeeze() == labels).sum().item()

        # Validation
        val_loss, val_acc = evaluate_model(
            model, device, val_loader, criterion, binary_classification
        )

        # Record metrics
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(100 * train_correct / train_total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(
            f'Train Loss: {history["train_loss"][-1]:.4f}, '
            f'Train Acc: {history["train_acc"][-1]:.2f}%'
        )
        print(
            f'Val Loss: {history["val_loss"][-1]:.4f}, '
            f'Val Acc: {history["val_acc"][-1]:.2f}%'
        )
        print("-" * 60)

    # # Save the trained model
    # os.makedirs('checkpoints', exist_ok=True)
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'history': history,
    #     'binary_classification': binary_classification
    # }, f'checkpoints/resnet50_{"binary" if binary_classification else "multi"}_classifier.pth')

    return model, history
