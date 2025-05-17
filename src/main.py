import torch
import torch.nn as nn
import sys
import time
import random
from tqdm import tqdm
import csv

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

    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    if choice == "1":
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
    if choice == "2":
        run_experiment_1_semi_supervised()
    else:
        print("Invalid option.")

def run_experiment_1_semi_supervised():
    print("\nRunning semi-supervised experiment (binary classification)...")

    num_layers = int(input("Insert the number of layers to train (last layer excluded): "))
    
    label_fraction = float(input("Enter labeled data fraction (e.g., 0.1 for 10%): "))
    labeled_loader, unlabeled_loader, val_loader, test_loader = OxfordPetDataset.get_semi_supervised_loaders(
        data_dir="../data/raw",
        batch_size=32,
        label_fraction=label_fraction,
        binary_classification=True,
    )
    print(
        f"Dataset loaded successfully! ({len(labeled_loader.dataset)} labeled samples and {len(unlabeled_loader.dataset)} unlabeled samples)"
        )

    model = ResNet50(binary_classification=True, freeze_backbone=True, num_train_layers=num_layers)
    device = get_device()
    trainer = ModelTrainer(model, device, binary_classification=True)

    print("\nTraining on labeled subset...")
    model, _ = trainer.train(labeled_loader, val_loader, num_epochs=3)

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nTraining on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)
    model, _ = trainer.train(combined_loader, val_loader, num_epochs=3)

    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")
    
    with open("semi_supervised_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label fraction", label_fraction, "Test Acc", test_acc, "Test Loss", test_loss])

def run_experiment_2_semi_supervised():
    print("\nRunning semi-supervised experiment (multi-class classification)...")

    label_fraction = float(input("Enter labeled data fraction (e.g., 0.1 for 10%): "))
    num_layers = int(input("Insert the number of layers to train (last layer excluded): "))

    # Load semi-supervised splits
    labeled_loader, unlabeled_loader, val_loader, test_loader = OxfordPetDataset.get_semi_supervised_loaders(
        data_dir="../data/raw",
        batch_size=32,
        label_fraction=label_fraction,
        binary_classification=False,
    )

    model = ResNet50(binary_classification=False, freeze_backbone=True, num_train_layers=num_layers)
    device = get_device()
    trainer = ModelTrainer(model, device, binary_classification=False)

    print("\nTraining on labeled subset...")
    model, _ = trainer.train(labeled_loader, val_loader, num_epochs=3)

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nTraining on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)
    model, _ = trainer.train(combined_loader, val_loader, num_epochs=3)

    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")

    # Optional: Save results
    import csv
    with open("semi_supervised_results_multiclass.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label fraction", label_fraction, "Test Acc", test_acc, "Test Loss", test_loss])
        
def run_experiment_2():
    """Run multi-class classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 2: ResNet50 multi-class classification")
    print("=" * 70)

    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    if choice == "1":
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

        # Create trainer
        trainer = ModelTrainer(model, device, binary_classification=False)

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
    if choice == "2":
        run_experiment_2_semi_supervised()
    else:
        print("Invalid option.")


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
