import torch
import torch.nn as nn
import sys
import time
import random
from tqdm import tqdm

# Local imports
from models.resnet import ResNet50
from models.vit import ViT
from trainer import ModelTrainer
from dataset import OxfordPetDataset
from utils import get_device, get_swedish_waiting_message, create_directories

# Create required directories
create_directories("../models/resnet", ["binary", "multiclass", "pretrained"])
create_directories("../models/vit", ["binary", "multiclass"])


def display_welcome_message():
    """Display a welcome message with ASCII art"""
    welcome_message = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   OXFORD PET DATASET CLASSIFICATION                       ║
    ║   KTH Royal Institute of Technology - DD2424              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(welcome_message)


def display_experiment_options():
    """Display available experiment options"""
    options = [
        "1. ResNet50 binary classification with Adam optimizer (E.1)",
        "2. Multi-class classification of dogs and cats with ResNet50 (E.2)",
        "3. ViT binary classification (Dog vs Cat)",
        "4. ViT multi-class classification (37 Breeds)",
        "5. Exit",
    ]

    print("\nAvailable experiments:")
    for option in options:
        print(f"  {option}")

    return options


def get_user_choice(max_choice):
    """Get user choice with input validation"""
    while True:
        try:
            choice = int(
                input("\nEnter the number of the experiment you want to run: ")
            )
            if 1 <= choice <= max_choice:
                return choice
            else:
                print(f"Please enter a number between 1 and {max_choice}")
        except ValueError:
            print("Please enter a valid number")


def run_experiment_1():
    """Run binary classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 1: ResNet50 binary classification")
    print("=" * 70)

    # Load data
    print("\nLoading Oxford-IIIT Pet Dataset...")
    train_loader, val_loader, test_loader, num_classes = (
        OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw", batch_size=32, binary_classification=True
        )
    )
    print(
        f"Dataset loaded successfully! ({len(train_loader.dataset)} training samples)"
    )

    # Get number of layers to train
    num_layers = int(
        input("\nInsert the number of layers to train (last layer excluded): ")
    )

    # Load model
    print("\nInitializing ResNet50 model...")
    for _ in tqdm(range(5), desc="Loading model"):
        time.sleep(0.5)  # Simulate loading time
    model = ResNet50(
        binary_classification=True, freeze_backbone=True, num_train_layers=num_layers
    )

    # Get device
    device = get_device()

    # Ask for gradient monitoring
    monitor_grads_choice = input("\nDo you want to monitor gradients? (y/n): ").lower()
    monitor_gradients = monitor_grads_choice == "y"
    gradient_monitor_interval = 100  # Default
    if monitor_gradients:
        try:
            interval = int(input("Monitor gradients every N batches (e.g., 50, 100): "))
            if interval > 0:
                gradient_monitor_interval = interval
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")

    # Create trainer
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=True,
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    model, history = trainer.train(
        train_loader, val_loader, num_epochs=3, print_graph=True
    )

    # Save model
    save_choice = input("\nDo you want to save the model? (y/n): ").lower()
    if save_choice == "y":
        trainer.save_model(model_type="binary")

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%")

    print("\nExperiment 1 completed!")


def run_experiment_2():
    """Run multi-class classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 2: ResNet50 multi-class classification")
    print("=" * 70)

    # Get number of layers to train
    num_train_layers = int(
        input("\nHow many layers to train? (-1 for gradient unfreezing): ")
    )

    # Load data
    print("\nLoading Oxford-IIIT Pet Dataset for multi-class classification...")
    train_loader, val_loader, test_loader, num_classes = (
        OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw", batch_size=32, binary_classification=False
        )
    )
    print(
        f"Dataset loaded successfully! ({len(train_loader.dataset)} training samples)"
    )

    # Load model
    print("\nInitializing ResNet50 model...")
    for _ in tqdm(range(5), desc="Loading model"):
        time.sleep(0.5)  # Simulate loading time

    if num_train_layers == -1:
        # For gradual unfreezing, backbone is initially frozen, and trainer handles unfreezing
        model = ResNet50(
            binary_classification=False,
            freeze_backbone=True,  # Important: Trainer will unfreeze layers gradually
            num_train_layers=0,  # Initially, only classifier is unfrozen by ResNet50 class
        )
    else:
        model = ResNet50(
            binary_classification=False,
            freeze_backbone=True,  # ResNet50 handles unfreezing based on num_train_layers
            num_train_layers=num_train_layers,
        )

    # Get device
    device = get_device()

    # Ask for gradient monitoring
    monitor_grads_choice = input("\nDo you want to monitor gradients? (y/n): ").lower()
    monitor_gradients = monitor_grads_choice == "y"
    gradient_monitor_interval = 100  # Default
    if monitor_gradients:
        try:
            interval = int(input("Monitor gradients every N batches (e.g., 50, 100): "))
            if interval > 0:
                gradient_monitor_interval = interval
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")

    # Create trainer
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    if num_train_layers == -1:
        print("\nStarting training with Gradual Unfreezing...")
        model, history = trainer.train_gradual_unfreezing(
            train_loader, val_loader, num_epochs=3, print_graph=True
        )
    else:
        model, history = trainer.train(
            train_loader, val_loader, num_epochs=3, print_graph=True
        )

    # Save model
    save_choice = input("\nDo you want to save the model? (y/n): ").lower()
    if save_choice == "y":
        trainer.save_model(model_type="multiclass")

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("\nExperiment 2 completed!")


def run_experiment_vit_binary():
    """Run ViT binary classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 3: ViT binary classification (Dog vs Cat)")
    print("=" * 70)

    vit_model_checkpoint = "google/vit-base-patch16-224"
    num_epochs_vit = 3  # Example, can be configured
    batch_size_vit = 32  # Adjust based on GPU memory

    # Load data
    train_loader, val_loader, test_loader, _ = OxfordPetDataset.get_dataloaders(
        data_dir="../data/raw",
        batch_size=batch_size_vit,
        binary_classification=True,
        model_type="vit",
        vit_model_name=vit_model_checkpoint,
    )
    print(
        f"Dataset loaded for ViT binary classification! ({len(train_loader.dataset)} training samples)"
    )

    # Load model
    print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
    model = ViT(model_name_or_path=vit_model_checkpoint, binary_classification=True)

    # Get device
    device = get_device()

    # Ask for gradient monitoring
    monitor_grads_choice = input("\nDo you want to monitor gradients? (y/n): ").lower()
    monitor_gradients = monitor_grads_choice == "y"
    gradient_monitor_interval = 100  # Default
    if monitor_gradients:
        try:
            interval = int(input("Monitor gradients every N batches (e.g., 50, 100): "))
            if interval > 0:
                gradient_monitor_interval = interval
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")

    # Create trainer
    # Note: ViT models often benefit from smaller learning rates e.g. 5e-5 or 2e-5
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=True,
        learning_rate=5e-5,
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    model, history = trainer.train(
        train_loader, val_loader, num_epochs=num_epochs_vit, print_graph=True
    )

    # Save model
    save_choice = input("\nDo you want to save the model? (y/n): ").lower()
    if save_choice == "y":
        trainer.save_model(model_type="binary", model_architecture="vit")

    # Evaluate on test set
    print("\nEvaluating ViT model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("\nFinal Test Results (ViT Binary):")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%")

    print("\nViT Binary Experiment completed!")


def run_experiment_vit_multiclass():
    """Run ViT multi-class classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 4: ViT multi-class classification (37 Breeds)")
    print("=" * 70)

    vit_model_checkpoint = "google/vit-base-patch16-224"
    num_epochs_vit = (
        3  # Example, can be configured. More epochs might be needed for multi-class.
    )
    batch_size_vit = 32  # Adjust based on GPU memory

    # Load data
    train_loader, val_loader, test_loader, _ = OxfordPetDataset.get_dataloaders(
        data_dir="../data/raw",
        batch_size=batch_size_vit,
        binary_classification=False,
        model_type="vit",
        vit_model_name=vit_model_checkpoint,
    )
    print(
        f"Dataset loaded for ViT multi-class classification! ({len(train_loader.dataset)} training samples)"
    )

    # Load model
    print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
    model = ViT(model_name_or_path=vit_model_checkpoint, binary_classification=False)

    # Get device
    device = get_device()

    # Ask for gradient monitoring
    monitor_grads_choice = input("\nDo you want to monitor gradients? (y/n): ").lower()
    monitor_gradients = monitor_grads_choice == "y"
    gradient_monitor_interval = 100  # Default
    if monitor_gradients:
        try:
            interval = int(input("Monitor gradients every N batches (e.g., 50, 100): "))
            if interval > 0:
                gradient_monitor_interval = interval
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")

    # Create trainer
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        learning_rate=5e-5,
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    # For multi-class ViT, gradual unfreezing could be explored later if needed.
    # For now, standard fine-tuning.
    model, history = trainer.train(
        train_loader, val_loader, num_epochs=num_epochs_vit, print_graph=True
    )

    # Save model
    save_choice = input("\nDo you want to save the model? (y/n): ").lower()
    if save_choice == "y":
        trainer.save_model(model_type="multiclass", model_architecture="vit")

    # Evaluate on test set
    print("\nEvaluating ViT model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("\nFinal Test Results (ViT Multi-class):")
    print(
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
    )  # .2f for multiclass as in run_exp2

    print("\nViT Multi-class Experiment completed!")


def main():
    """Main function"""
    display_welcome_message()

    while True:
        options = display_experiment_options()
        choice = get_user_choice(len(options))

        if choice == 1:
            run_experiment_1()
        elif choice == 2:
            run_experiment_2()
        elif choice == 3:
            run_experiment_vit_binary()
        elif choice == 4:
            run_experiment_vit_multiclass()
        elif choice == 5:
            print("\nExiting program. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
