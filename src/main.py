import torch
import torch.nn as nn
import sys
import time
import random
from tqdm import tqdm
import csv

# Local imports
from models.resnet import ResNet50
from models.vit import ViT
from trainer import ModelTrainer
from dataset import OxfordPetDataset
from utils import get_device, get_swedish_waiting_message, create_directories
from utils import compute_class_weights, get_oversampled_loader

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend (no GUI required)

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
        "5. ResNet50 multi-class classification (E.2) with an imbalanced training set",
        "6. ViT multi-class classification with an imbalanced training set",
        "7. Exit",
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
        #PUT SOMEWHERE THE IMBALANCED
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

        # Load model
        print("\nInitializing ResNet50 model...")
        for _ in tqdm(range(5), desc="Loading model"):
            time.sleep(0.2)  # Simulate loading time
        model = ResNet50(binary_classification=True, freeze_backbone=True)

        # Get device
        device = get_device()

        # Ask for gradient monitoring
        monitor_grads_choice = input(
            "\nDo you want to monitor gradients? (y/n): "
        ).lower()
        monitor_gradients = monitor_grads_choice == "y"
        gradient_monitor_interval = 100  # Default
        if monitor_gradients:
            try:
                interval = int(
                    input("Monitor gradients every N batches (e.g., 50, 100): ")
                )
                if interval > 0:
                    gradient_monitor_interval = interval
                else:
                    print("Invalid interval, using default 100.")
            except ValueError:
                print("Invalid input, using default interval 100.")

        # Get scheduler preferences
        
        use_scheduler = input("\nWould you like to use the learning rate scheduler? (y/n): ").lower() == 'y'

        # Create trainer
        trainer = ModelTrainer(
            model,
            device,
            binary_classification=True,
            learning_rate=[0.001],
            monitor_gradients=monitor_gradients,
            gradient_monitor_interval=gradient_monitor_interval,
            use_scheduler=use_scheduler,
            scheduler_params={'max_lr': 1e-2, 'pct_start': 0.3} if use_scheduler else None
        )

        # Display Swedish humor
        print(f"\n{get_swedish_waiting_message()}")

        # Train model
        model, history = trainer.train(
            train_loader, val_loader, num_epochs=2, print_graph=True
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

    num_layers = int(
        input("Insert the number of layers to train (last layer excluded): ")
    )

    label_fraction = float(input("Enter labeled data fraction (e.g., 0.1 for 10%): "))
    labeled_loader, unlabeled_loader, val_loader, test_loader = (
        OxfordPetDataset.get_semi_supervised_loaders(
            data_dir="../data/raw",
            batch_size=32,
            label_fraction=label_fraction,
            binary_classification=True,
        )
    )
    print(
        f"Dataset loaded successfully! ({len(labeled_loader.dataset)} labeled samples and {len(unlabeled_loader.dataset)} unlabeled samples)"
    )

    model = ResNet50(
        binary_classification=True, freeze_backbone=True, num_train_layers=num_layers
    )
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

    # Ask for batch normalization fine-tuning
    finetune_bn_choice = input(
        "\nDo you want to fine-tune batch normalization parameters? (y/n): "
    ).lower()
    finetune_bn = finetune_bn_choice == "y"

    # Create trainer
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=True,
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
        finetune_bn=finetune_bn,
    )

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
        writer.writerow(
            [
                "Label fraction",
                label_fraction,
                "Test Acc",
                test_acc,
                "Test Loss",
                test_loss,
            ]
        )

def run_experiment_imbalanced_multiclass():
    print("\n" + "=" * 70)
    print("Starting experiment: ResNet50 MULTICLASS with imbalanced training set")
    print("=" * 70)

    print("\nStrategy options to handle imbalance:")
    print("1. No strategy (baseline)")
    print("2. Weighted CrossEntropyLoss")
    print("3. Oversampling minority classes")

    strategy = int(input("Choose strategy (1/2/3): ").strip())

    user_input = int(
                input(
                    "\nInsert the number of layers to train (last layer excluded): "
                )
            )
    # Load imbalanced data
    train_loader, val_loader, test_loader, num_classes = OxfordPetDataset.get_dataloaders(
        data_dir="../data/raw",
        batch_size=32,
        binary_classification=False,
        apply_imbalance=True,
    )

    # Apply oversampling if selected
    if strategy == 3:
        print("Applying oversampling to rebalance classes...")
        train_loader = get_oversampled_loader(train_loader.dataset, batch_size=32)

    # Load model
    print("\nInitializing ResNet50 model...")
    for _ in tqdm(range(5), desc="Loading model"):
        time.sleep(0.2)
    model = ResNet50(binary_classification=False, freeze_backbone=True, num_train_layers=user_input)

    device = get_device()

    # Weighted loss
    if strategy == 2:
        print("Using weighted CrossEntropyLoss...")
        class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Create trainer
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        learning_rate=[1e-4],
        loss_fn=loss_fn,
    )

    # Train
    model, _ = trainer.train(train_loader, val_loader, num_epochs=3, print_graph=True)

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nTest Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")


def run_experiment_2_semi_supervised():
    print("\nRunning semi-supervised experiment (multi-class classification)...")

    user_input = int(
        input(
            "\nSelect training option: \n n>0: train n layers \n '-1': gradually unfreeze each layer \n '-2': different learning rate for each layer and no data augmentation \n '-3': different learning rates for each layer and data augmentation \n User input: "
        )
    )
    label_fraction = float(input("Enter labeled data fraction (e.g., 0.1 for 10%): "))

    data_augmentation = user_input == -3

    # Load semi-supervised splits
    labeled_loader, unlabeled_loader, val_loader, test_loader = (
        OxfordPetDataset.get_semi_supervised_loaders(
            data_dir="../data/raw",
            batch_size=32,
            label_fraction=label_fraction,
            binary_classification=False,
            data_augmentation=data_augmentation,
        )
    )

    # Load model
    print("\nInitializing ResNet50 model...")
    for _ in tqdm(range(5), desc="Loading model"):
        time.sleep(0.2)  # Simulate loading time

    if user_input == -2 or user_input == -3:

        model = ResNet50(
            binary_classification=False,
            freeze_backbone=False,
            num_train_layers=0,
        )

    else:

        # Freeze all layers except the selected ones
        model = ResNet50(
            binary_classification=False,
            freeze_backbone=True,  # ResNet50 handles unfreezing based on num_train_layers
            num_train_layers=user_input if user_input > 0 else 0,
        )

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

    # Ask for batch normalization fine-tuning
    finetune_bn_choice = input(
        "\nDo you want to fine-tune batch normalization parameters? (y/n): "
    ).lower()
    finetune_bn = finetune_bn_choice == "y"

    # Create trainer
    if user_input == -2 or user_input == -3:  # Different learning rates for each layer
        learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
    else:
        learning_rates = [0.00001]
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        learning_rate=learning_rates,
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
        finetune_bn=finetune_bn,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    if user_input == -1:
        print("\nStarting training with Gradual Unfreezing on labeled data...")
        model, _ = trainer.train_gradual_unfreezing(
            labeled_loader, val_loader, num_epochs=3, print_graph=True
        )

    else:

        model, history = trainer.train(
            labeled_loader, val_loader, num_epochs=3, print_graph=True
        )

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nTraining on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)
    # Train model with all the data
    if user_input == -1:
        print("\nStarting training with Gradual Unfreezing on labeled data...")
        model, _ = trainer.train_gradual_unfreezing(
            combined_loader, val_loader, num_epochs=3, print_graph=True
        )
    else:
        # TODO: ADD LEARNING RATES
        model, history = trainer.train(
            combined_loader, val_loader, num_epochs=3, print_graph=True
        )

    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")

    # Optional: Save results
    import csv

    with open("semi_supervised_results_multiclass.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Label fraction",
                label_fraction,
                "Test Acc",
                test_acc,
                "Test Loss",
                test_loss,
            ]
        )


def run_experiment_2():
    """Run multi-class classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 2: ResNet50 multi-class classification")
    print("=" * 70)
    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    if choice == "1":

        # Get number of layers to train
        user_input = int(
            input(
                "\nSelect training option: \n n>0: train n layers \n '-1': gradually unfreeze each layer (fixed learning rate) \n '-2': different learning rate for each layer and no data augmentation \n '-3': different learning rates for each layer and data augmentation \n User input: "
            )
        )

        # Load data
        print("\nLoading Oxford-IIIT Pet Dataset for multi-class classification...")
        train_loader, val_loader, test_loader, num_classes = (
            OxfordPetDataset.get_dataloaders(
                data_dir="../data/raw",
                batch_size=32,
                binary_classification=False,
                data_augmentation=True if user_input == -3 else False,
            )
        )
        print(
            f"Dataset loaded successfully! ({len(train_loader.dataset)} training samples)"
        )

        # Load model
        print("\nInitializing ResNet50 model...")
        for _ in tqdm(range(5), desc="Loading model"):
            time.sleep(0.2)  # Simulate loading time

        if user_input == -2 or user_input == -3:
            # All layers unfrozen
            model = ResNet50(
                binary_classification=False,
                freeze_backbone=False,
                num_train_layers=0,
            )

        else:
            model = ResNet50(
                binary_classification=False,
                freeze_backbone=True,  # ResNet50 handles unfreezing based on num_train_layers
                num_train_layers=user_input if user_input > 0 else 0,
            )

        # Get device
        device = get_device()

        # Ask for gradient monitoring
        monitor_grads_choice = input(
            "\nDo you want to monitor gradients? (y/n): "
        ).lower()
        monitor_gradients = monitor_grads_choice == "y"
        gradient_monitor_interval = 100  # Default

        if monitor_gradients:
            try:
                interval = int(
                    input("Monitor gradients every N batches (e.g., 50, 100): ")
                )
                if interval > 0:
                    gradient_monitor_interval = interval
                else:
                    print("Invalid interval, using default 100.")
            except ValueError:
                print("Invalid input, using default interval 100.")

        # Ask for batch normalization fine-tuning
        finetune_bn_choice = input(
            "\nDo you want to fine-tune batch normalization parameters? (y/n): "
        ).lower()
        finetune_bn = finetune_bn_choice == "y"

        # Create trainer
        if (
            user_input == -2 or user_input == -3
        ):  # Different learning rates for each layer
            learning_rates = [
                1e-3,
                5e-4,
                1e-4,
                5e-5,
                1e-5,
                5e-6,
                1e-6,
                5e-7,
                1e-7,
                5e-8,
            ]
        else:
            learning_rates = [5e-5]
            
        use_scheduler = input("\nWould you like to use the learning rate scheduler? (y/n): ").lower() == 'y'

        # Create trainer
        trainer = ModelTrainer(
            model,
            device,
            binary_classification=False,
            learning_rate=[0.001],
            monitor_gradients=monitor_gradients,
            gradient_monitor_interval=gradient_monitor_interval,
            finetune_bn=finetune_bn,
            use_scheduler=use_scheduler,
            scheduler_params={'max_lr': 1e-2, 'pct_start': 0.3} if use_scheduler else None
        )


        # Display Swedish humor
        print(f"\n{get_swedish_waiting_message()}")

        # Train model
        if user_input == -1:

            model, history = trainer.train_gradual_unfreezing(
                train_loader, val_loader, num_epochs=1, print_graph=True
            )
        else:

            model, history = trainer.train(
                train_loader, val_loader, num_epochs=1, print_graph=True
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
    elif choice == "2":
        run_experiment_2_semi_supervised()
    else:
        print("Invalid option.")


def run_experiment_vit_binary():
    """Run ViT binary classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 3: ViT binary classification (Dog vs Cat)")
    print("=" * 70)

    vit_model_checkpoint = "google/vit-base-patch16-224"
    num_epochs_vit = 3  # can be configured
    batch_size_vit = 32

    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    if choice == "1":
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
        monitor_grads_choice = input(
            "\nDo you want to monitor gradients? (y/n): "
        ).lower()
        monitor_gradients = monitor_grads_choice == "y"
        gradient_monitor_interval = 100  # Default
        if monitor_gradients:
            try:
                interval = int(
                    input("Monitor gradients every N batches (e.g., 50, 100): ")
                )
                if interval > 0:
                    gradient_monitor_interval = interval
                else:
                    print("Invalid interval, using default 100.")
            except ValueError:
                print("Invalid input, using default interval 100.")

        # # Ask for batch normalization fine-tuning
        # finetune_bn_choice = input(
        #     "\nDo you want to fine-tune batch normalization parameters? (y/n): "
        # ).lower()
        # finetune_bn = finetune_bn_choice == "y"

        # Create trainer
        # Note: ViT models often benefit from smaller learning rates e.g. 5e-5 or 2e-5
        trainer = ModelTrainer(
            model,
            device,
            binary_classification=True,
            learning_rate=[5e-5],
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
    elif choice == "2":
        run_experiment_vit_binary_semi()
    else:
        print("Invalid option.")
    print("\nViT Binary Experiment completed!")


def run_experiment_vit_binary_semi():
    """Run ViT binary classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 3: ViT binary classification (Dog vs Cat)")
    print("=" * 70)

    vit_model_checkpoint = "google/vit-base-patch16-224"
    num_epochs_vit = 3  # can be configured
    batch_size_vit = 32

    label_fraction = float(input("Enter labeled data fraction (e.g., 0.1 for 10%): "))

    labeled_loader, unlabeled_loader, val_loader, test_loader = (
        OxfordPetDataset.get_semi_supervised_loaders(
            data_dir="../data/raw",
            batch_size=32,
            label_fraction=label_fraction,
            binary_classification=True,
            model_type="vit",
            vit_model_name=vit_model_checkpoint,
        )
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

    # # Ask for batch normalization fine-tuning
    # finetune_bn_choice = input(
    #     "\nDo you want to fine-tune batch normalization parameters? (y/n): "
    # ).lower()
    # finetune_bn = finetune_bn_choice == "y"

    # Create trainer
    trainer = ModelTrainer(
        model,
        device,
        binary_classification=True,
        learning_rate=[5e-5],
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    model, _ = trainer.train(
        labeled_loader, val_loader, num_epochs=num_epochs_vit, print_graph=True
    )

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nTraining on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)

    model, _ = trainer.train(
        combined_loader, val_loader, num_epochs=3, print_graph=True
    )
    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")


def run_experiment_vit_multiclass_semi():
    vit_model_checkpoint = "google/vit-base-patch16-224"
    num_epochs_vit = (
        3  # Example, can be configured. More epochs might be needed for multi-class.
    )
    batch_size_vit = 32  # Adjust based on GPU memory

    label_fraction = float(input("Enter labeled data fraction (e.g., 0.1 for 10%): "))

    labeled_loader, unlabeled_loader, val_loader, test_loader = (
        OxfordPetDataset.get_semi_supervised_loaders(
            data_dir="../data/raw",
            batch_size=batch_size_vit,
            binary_classification=False,
            model_type="vit",
            vit_model_name=vit_model_checkpoint,
        )
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
        learning_rate=[5e-5],
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    # For multi-class ViT, gradual unfreezing could be explored later if needed.
    # For now, standard fine-tuning.
    model, history = trainer.train(
        labeled_loader, val_loader, num_epochs=num_epochs_vit, print_graph=True
    )

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nTraining on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)

    model, _ = trainer.train(
        combined_loader, val_loader, num_epochs=3, print_graph=True
    )
    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")

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

def run_experiment_vit_multiclass_imbalanced():
    print("\n" + "=" * 70)
    print("Starting experiment: ViT MULTICLASS with imbalanced training set")
    print("=" * 70)
    
    vit_model_checkpoint = "google/vit-base-patch16-224"
    num_epochs_vit = 3  # Example, can be configured. More epochs might be needed for multi-class.
    batch_size_vit = 32  # Adjust based on GPU memory

    print("\nStrategy options to handle imbalance:")
    print("1. No strategy (baseline)")
    print("2. Weighted CrossEntropyLoss")
    print("3. Oversampling minority classes")
    
    strategy = int(input("Choose strategy (1/2/3): ").strip())
    vit_model_checkpoint = "google/vit-base-patch16-224"
    num_epochs_vit = 3
    batch_size = 32
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = OxfordPetDataset.get_dataloaders(
        data_dir="../data/raw",
        batch_size=batch_size_vit,
        binary_classification=False,
        model_type="vit",
        vit_model_name=vit_model_checkpoint,
        apply_imbalance=True,
    )
    print(
        f"Dataset loaded for ViT multi-class classification! ({len(train_loader.dataset)} training samples)"
    )
    
    if strategy == 3:
        print("Applying oversampling to rebalance classes...")
        train_loader = get_oversampled_loader(train_loader.dataset, batch_size=batch_size)
    
    # Load model
    print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
    model = ViT(
        model_name_or_path=vit_model_checkpoint, binary_classification=False
    )

    # Get device
    device = get_device()

    # Use class weights if selected
    if strategy == 2:
        print("Using weighted CrossEntropyLoss...")
        class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
        
    # Ask for gradient monitoring
    monitor_grads_choice = input(
        "\nDo you want to monitor gradients? (y/n): "
    ).lower()
    monitor_gradients = monitor_grads_choice == "y"
    gradient_monitor_interval = 100  # Default
    if monitor_gradients:
        try:
            interval = int(
                input("Monitor gradients every N batches (e.g., 50, 100): ")
            )
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
        learning_rate=[5e-5],
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
        loss_fn=loss_fn,
    )

    # Display Swedish humor
    print(f"\n{get_swedish_waiting_message()}")

    # Train model
    # For multi-class ViT, gradual unfreezing could be explored later if needed.
    # For now, standard fine-tuning.
    model, _ = trainer.train(
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
    )

def run_experiment_vit_multiclass():
    """Run ViT multi-class classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 4: ViT multi-class classification (37 Breeds)")
    print("=" * 70)

    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    if choice == "1":
        vit_model_checkpoint = "google/vit-base-patch16-224"
        num_epochs_vit = 3  # Example, can be configured. More epochs might be needed for multi-class.
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
        model = ViT(
            model_name_or_path=vit_model_checkpoint, binary_classification=False
        )

        # Get device
        device = get_device()

        # Ask for gradient monitoring
        monitor_grads_choice = input(
            "\nDo you want to monitor gradients? (y/n): "
        ).lower()
        monitor_gradients = monitor_grads_choice == "y"
        gradient_monitor_interval = 100  # Default
        if monitor_gradients:
            try:
                interval = int(
                    input("Monitor gradients every N batches (e.g., 50, 100): ")
                )
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
            learning_rate=[5e-5],
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
    elif choice == "2":
        run_experiment_vit_multiclass_semi()
    else:
        print("Invalid option.")
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
            run_experiment_imbalanced_multiclass()
        elif choice == 6:
            run_experiment_vit_multiclass_imbalanced()
        elif choice == 7:
            print("\nExiting program. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
