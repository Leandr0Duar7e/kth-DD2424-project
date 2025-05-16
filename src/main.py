import torch
import torch.nn as nn
import sys
import time
import random
from tqdm import tqdm

# Local imports
from models.resnet import ResNet50
from trainer import ModelTrainer
from dataset import OxfordPetDataset
from utils import get_device, get_swedish_waiting_message, create_directories

# Create required directories
create_directories("../models/resnet", ["binary", "multiclass", "pretrained"])


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
        "3. Exit",
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

    # Create trainer
    trainer = ModelTrainer(model, device, binary_classification=True)

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
    user_input = int(
        input("\nSelect training option: \n n>0: train n layers \n '-1': gradually unfreeze each layer \n '-2': different learning rate for each layer and no data augmentation \n '-3': different learning rates for each layer and data augmentation")
    )               
             
    # Load data
    print("\nLoading Oxford-IIIT Pet Dataset for multi-class classification...")
    train_loader, val_loader, test_loader, num_classes = (
        OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw", batch_size=32, binary_classification=False, data_augmentation=True if user_input == -3 else False
        )
    )
    print(
        f"Dataset loaded successfully! ({len(train_loader.dataset)} training samples)"
    )

    # Load model
    print("\nInitializing ResNet50 model...")
    for _ in tqdm(range(5), desc="Loading model"):
        time.sleep(0.5)  # Simulate loading time


    
    if user_input == -2 or user_input == -3:
        # For gradual unfreezing, backbone is initially frozen, and trainer handles unfreezing
        model = ResNet50(
            binary_classification=False,
            freeze_backbone=False,  # Important: Trainer will unfreeze layers gradually
            num_train_layers=0,  # Initially, only classifier is unfrozen by ResNet50 class
        )
    else:
        
        model = ResNet50(
            binary_classification=False,
            freeze_backbone=True,  # ResNet50 handles unfreezing based on num_train_layers
            num_train_layers=user_input if user_input > 0 else 0,
        )

    # Get device
    device = get_device()

    # Create trainer
    if user_input == -2 or user_input == -3: # Different learning rates for each layer
        learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
    else:
        learning_rates = [0.001]
    trainer = ModelTrainer(model, device, binary_classification=False, learning_rate=learning_rates)

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    if user_input == -1:
        print("\nStarting training with Gradual Unfreezing...")
        model, history = trainer.train_gradual_unfreezing(
            train_loader, val_loader, num_epochs=3, print_graph=True
        )
    else:        
        
        # TODO: ADD LEARNING RATES
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
            print("\nExiting program. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
