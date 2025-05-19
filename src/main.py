import torch
import torch.nn as nn
import sys
import time
import random
from tqdm import tqdm
import csv
import os

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
    model_architecture = "resnet"
    classification_type = "binary"
    experiment_params = {}

    if choice == "1":
        training_type_str = "sup"
        experiment_params["training_type"] = "supervised"
        try:
            num_epochs = int(
                input("Enter the number of epochs for supervised training (e.g., 2): ")
            )
            if num_epochs <= 0:
                print("Number of epochs must be positive. Using default: 2.")
                num_epochs = 2
        except ValueError:
            print("Invalid input for epochs. Using default: 2.")
            num_epochs = 2
        experiment_params["epochs"] = num_epochs

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

        # Get learning rate
        try:
            lr_input = float(
                input("Enter the learning rate for supervised training (e.g., 0.001): ")
            )
            if lr_input <= 0:
                print("Learning rate must be positive. Using default: 0.001.")
                lr_input = 0.001
        except ValueError:
            print("Invalid input for learning rate. Using default: 0.001.")
            lr_input = 0.001
        lr_config = [lr_input]
        experiment_params["learning_rate"] = lr_config[0]

        # Create trainer
        trainer = ModelTrainer(
            model,
            device,
            binary_classification=True,
            learning_rate=lr_config,
            monitor_gradients=monitor_gradients,
            gradient_monitor_interval=gradient_monitor_interval,
        )

        # Display Swedish humor
        print(f"\n{get_swedish_waiting_message()}")

        # Train model
        start_time = time.time()
        model, history = trainer.train(
            train_loader, val_loader, num_epochs=num_epochs, print_graph=True
        )
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds.")

        # Save model
        save_choice = input("\nDo you want to save the model? (y/n): ").lower()
        if save_choice == "y":
            # Construct model name and path
            name_parts = [
                model_architecture,
                classification_type,
                f"{num_epochs}ep",
                f"lr{lr_config[0]}",
                training_type_str,
            ]
            if monitor_gradients:
                name_parts.append("gradmon")
            # finetune_bn is True by default in ModelTrainer if not specified otherwise by user for this experiment
            # We'll assume it's true unless explicitly set to false and track that in more complex experiments

            model_filename_base = "_".join(name_parts)
            model_save_path = os.path.join(
                "..",
                "models",
                model_architecture,
                classification_type,
                model_filename_base + ".pth",
            )

            print(f"Attempting to save model to: {model_save_path}")
            trainer.save_model(full_save_path=model_save_path)
            experiment_params["model_path"] = model_save_path

            # Placeholder for evaluation call
            evaluation_output_dir = os.path.join(
                "..",
                "evaluation",
                model_architecture,
                classification_type,
                model_filename_base,
            )
            print(
                f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}, time: {training_time:.2f}s"
            )
            # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, training_time=training_time, experiment_params_dict=experiment_params)

        # Evaluate on test set
        print("\nEvaluating model on test set...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        print("\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%")

        print("\nExperiment 1 completed!")
    elif choice == "2":
        run_experiment_1_semi_supervised()
    else:
        print("Invalid option.")


def run_experiment_1_semi_supervised():
    print("\nRunning semi-supervised experiment (binary classification)...")

    model_architecture = "resnet"
    classification_type = "binary"
    experiment_params = {"training_type": "semi-supervised"}

    try:
        num_epochs_labeled = int(
            input("Enter epochs for initial training on labeled data (e.g., 3): ")
        )
        if num_epochs_labeled <= 0:
            print("Number of epochs must be positive. Using default: 3.")
            num_epochs_labeled = 3
    except ValueError:
        print("Invalid input for epochs. Using default: 3.")
        num_epochs_labeled = 3
    try:
        num_epochs_combined = int(
            input("Enter epochs for training on combined data (e.g., 3): ")
        )
        if num_epochs_combined <= 0:
            print("Number of epochs must be positive. Using default: 3.")
            num_epochs_combined = 3
    except ValueError:
        print("Invalid input for epochs. Using default: 3.")
        num_epochs_combined = 3
    experiment_params["epochs_labeled"] = num_epochs_labeled
    experiment_params["epochs_combined"] = num_epochs_combined

    label_fraction = float(input("Enter labeled data fraction (e.g., 0.1 for 10%): "))
    experiment_params["label_fraction"] = label_fraction
    training_type_str = f"semi_frac{label_fraction}"

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

    model = ResNet50(binary_classification=True, freeze_backbone=True)
    device = get_device()

    monitor_grads_choice = input("\nDo you want to monitor gradients? (y/n): ").lower()
    monitor_gradients = monitor_grads_choice == "y"
    experiment_params["monitor_gradients"] = monitor_gradients
    gradient_monitor_interval = 100
    if monitor_gradients:
        try:
            interval = int(input("Monitor gradients every N batches (e.g., 50, 100): "))
            if interval > 0:
                gradient_monitor_interval = interval
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")
        experiment_params["gradient_monitor_interval"] = gradient_monitor_interval

    # Get learning rate
    try:
        lr_input = float(input("Enter the learning rate (e.g., 0.001): "))
        if lr_input <= 0:
            print("Learning rate must be positive. Using default: 0.001.")
            lr_input = 0.001
    except ValueError:
        print("Invalid input for learning rate. Using default: 0.001.")
        lr_input = 0.001
    lr_config = [lr_input]
    experiment_params["learning_rate"] = lr_config[0]

    trainer = ModelTrainer(
        model,
        device,
        binary_classification=True,
        learning_rate=lr_config,
        monitor_gradients=monitor_gradients,
        gradient_monitor_interval=gradient_monitor_interval,
    )

    print("\nTraining on labeled subset...")
    model, _ = trainer.train(labeled_loader, val_loader, num_epochs=num_epochs_labeled)

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nTraining on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)
    model, _ = trainer.train(
        combined_loader, val_loader, num_epochs=num_epochs_combined
    )

    total_training_time = None
    # Save model (after all training stages)
    save_choice = input("\nDo you want to save the final model? (y/n): ").lower()
    if save_choice == "y":
        name_parts = [
            model_architecture,
            classification_type,
            f"{num_epochs_labeled}l+{num_epochs_combined}c_ep",  # Distinguish labeled and combined epochs
            f"lr{lr_config[0]}",
            training_type_str,
        ]
        if monitor_gradients:
            name_parts.append("gradmon")

        model_filename_base = "_".join(name_parts)
        model_save_path = os.path.join(
            "..",
            "models",
            model_architecture,
            classification_type,
            model_filename_base + ".pth",
        )
        experiment_params["model_path"] = model_save_path

        print(f"Attempting to save model to: {model_save_path}")
        trainer.save_model(full_save_path=model_save_path)

        evaluation_output_dir = os.path.join(
            "..",
            "evaluation",
            model_architecture,
            classification_type,
            model_filename_base,
        )
        print(
            f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}, total time: {total_training_time:.2f}s"
        )
        # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, training_time=total_training_time, experiment_params_dict=experiment_params)

    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")


def run_experiment_imbalanced_multiclass():
    print("\n" + "=" * 70)
    print("Starting experiment: ResNet50 MULTICLASS with imbalanced training set")
    print("=" * 70)

    model_architecture = "resnet"
    classification_type = "multiclass"  # Main type for folder structure
    experiment_params = {}

    print("\nStrategy options to handle imbalance:")
    print("1. No strategy (baseline)")
    print("2. Weighted CrossEntropyLoss")
    print("3. Oversampling minority classes")
    strategy = int(input("Choose strategy (1/2/3): ").strip())
    experiment_params["imbalance_strategy"] = strategy
    training_type_str = f"sup_imbal_strat{strategy}"
    experiment_params["training_type"] = "supervised_imbalanced"

    try:
        num_epochs = int(input("Enter the number of epochs for training (e.g., 3): "))
        if num_epochs <= 0:
            print("Number of epochs must be positive. Using default: 3.")
            num_epochs = 3
    except ValueError:
        print("Invalid input for epochs. Using default: 3.")
        num_epochs = 3
    experiment_params["epochs"] = num_epochs

    user_input_layers = int(
        input("\nInsert the number of layers to train (last layer excluded): ")
    )
    experiment_params["num_layers_trained_excluding_fc"] = user_input_layers

    # Defaults for this function as they are not explicitly prompted here
    current_data_augmentation = False
    experiment_params["data_augmentation"] = current_data_augmentation
    current_l2_lambda = 0.0  # Default L2, ModelTrainer applies it if lam > 0
    experiment_params["l2_lambda"] = current_l2_lambda
    current_finetune_bn = True  # Default ModelTrainer behavior
    experiment_params["finetune_bn"] = current_finetune_bn
    current_monitor_gradients = False  # Not prompted in this specific flow
    experiment_params["monitor_gradients"] = current_monitor_gradients
    # gradient_monitor_interval not stored if current_monitor_gradients is False

    # Load imbalanced data
    # Note: apply_imbalance=True is crucial here
    train_loader, val_loader, test_loader, num_classes_data = (
        OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw",
            batch_size=32,
            binary_classification=False,
            apply_imbalance=True,  # This is key for this experiment
        )
    )

    # Apply oversampling if selected (modifies train_loader)
    if strategy == 3:
        print("Applying oversampling to rebalance classes...")
        train_loader = get_oversampled_loader(train_loader.dataset, batch_size=32)

    # Load model
    print("\nInitializing ResNet50 model...")
    for _ in tqdm(range(5), desc="Loading model"):
        time.sleep(0.2)
    model = ResNet50(
        binary_classification=False,
        freeze_backbone=True,
        num_train_layers=user_input_layers,
    )
    device = get_device()

    loss_fn_choice = None
    loss_desc_for_filename = "CE"  # Default to CrossEntropy
    if strategy == 2:
        print("Using weighted CrossEntropyLoss...")
        # Ensure num_classes_data is correctly passed if different from model's internal default for multiclass
        class_weights = compute_class_weights(
            train_loader.dataset, num_classes_data
        ).to(device)
        loss_fn_choice = nn.CrossEntropyLoss(weight=class_weights)
        experiment_params["loss_function_type"] = "weighted_ce"
        loss_desc_for_filename = "WeightedCE"
    else:
        loss_fn_choice = nn.CrossEntropyLoss()
        experiment_params["loss_function_type"] = "ce"
    # experiment_params["learning_rate"] is already set

    # Get learning rate
    try:
        lr_input = float(input("Enter the learning rate (e.g., 0.0001): "))
        if lr_input <= 0:
            print("Learning rate must be positive. Using default: 0.0001.")
            lr_input = 0.0001
    except ValueError:
        print("Invalid input for learning rate. Using default: 0.0001.")
        lr_input = 0.0001
    lr_config = [lr_input]
    experiment_params["learning_rate"] = lr_config[0]

    # Monitor gradients (optional for this experiment, can be added)
    monitor_gradients = False  # Default for this exp unless prompted
    # gradient_monitor_interval = 100 # Not used if monitor_gradients is False
    experiment_params["monitor_gradients"] = monitor_gradients
    # finetune_bn is True by default for ModelTrainer, captured above

    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        learning_rate=lr_config,
        loss_fn=loss_fn_choice,  # Pass the chosen loss function
        # lam=0.0, # L2 regularization, can be added as input
        # monitor_gradients=monitor_gradients,
        # gradient_monitor_interval=gradient_monitor_interval,
        # finetune_bn=True, # Default
    )

    # Train
    start_time = time.time()
    model, _ = trainer.train(
        train_loader, val_loader, num_epochs=num_epochs, print_graph=True
    )
    end_time = time.time()
    training_time = end_time - start_time
    experiment_params["training_time_seconds"] = training_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Save model
    save_choice = input("\nDo you want to save the model? (y/n): ").lower()
    if save_choice == "y":
        name_parts = [
            model_architecture,
            classification_type,
            f"{num_epochs}ep",
            f"lr{lr_config[0]}",
            training_type_str,  # e.g. "sup_imbal_strat1" (includes strategy)
            f"loss{loss_desc_for_filename}",  # Describes the loss function used
            f"layers{user_input_layers}",
            f"aug{current_data_augmentation}",
            f"bn{current_finetune_bn}",
            f"L2reg{current_l2_lambda}",
        ]
        if current_monitor_gradients:
            name_parts.append("gradmon")

        model_filename_base = "_".join(name_parts)
        model_save_path = os.path.join(
            "..",
            "models",
            model_architecture,
            classification_type,
            model_filename_base + ".pth",
        )
        experiment_params["model_path"] = model_save_path

        print(f"Attempting to save model to: {model_save_path}")
        trainer.save_model(full_save_path=model_save_path)

        evaluation_output_dir = os.path.join(
            "..",
            "evaluation",
            model_architecture,
            classification_type,
            model_filename_base,  # Subfolder named after the model base name
        )
        print(
            f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}, time: {training_time:.2f}s"
        )
        # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, training_time=training_time, experiment_params_dict=experiment_params)

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("\nFinal Test Results (Multi-class Imbalanced):")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


def run_experiment_2_semi_supervised():
    print("\nRunning semi-supervised experiment (multi-class classification)...")
    model_architecture = "resnet"
    classification_type = "multiclass"
    experiment_params = {"training_type": "semi-supervised"}

    try:
        num_epochs_labeled = int(
            input("Enter epochs for initial training on labeled data (e.g., 3): ")
        )
        if num_epochs_labeled <= 0:
            print("Number of epochs must be positive. Using default: 3.")
            num_epochs_labeled = 3
    except ValueError:
        print("Invalid input for epochs. Using default: 3.")
        num_epochs_labeled = 3
    experiment_params["epochs_labeled"] = num_epochs_labeled

    try:
        num_epochs_combined = int(
            input("Enter epochs for training on combined data (e.g., 3): ")
        )
        if num_epochs_combined <= 0:
            print("Number of epochs must be positive. Using default: 3.")
            num_epochs_combined = 3
    except ValueError:
        print("Invalid input for epochs. Using default: 3.")
        num_epochs_combined = 3
    experiment_params["epochs_combined"] = num_epochs_combined

    user_input_train_opt = int(
        input(
            "\nSelect training option: \n n>0: train n layers \n '-1': gradually unfreeze each layer \n '-2': different learning rate for each layer and no data augmentation \n '-3': different learning rates for each layer and data augmentation \n User input: "
        )
    )
    experiment_params["training_option_resnet_e2"] = user_input_train_opt

    label_fraction_val = float(
        input("Enter labeled data fraction (e.g., 0.1 for 10%): ")
    )
    experiment_params["label_fraction"] = label_fraction_val

    data_augmentation_val = user_input_train_opt == -3
    experiment_params["data_augmentation"] = data_augmentation_val

    num_layers_to_train_for_model_init = 0
    train_opt_filename_desc = ""
    if user_input_train_opt > 0:
        train_opt_filename_desc = f"layers{user_input_train_opt}"
        num_layers_to_train_for_model_init = user_input_train_opt
        experiment_params["num_layers_trained_excluding_fc"] = user_input_train_opt
    elif user_input_train_opt == -1:
        train_opt_filename_desc = "gradUnfreeze"
        experiment_params["num_layers_trained_excluding_fc"] = "N/A (gradual unfreeze)"
    elif user_input_train_opt == -2:
        train_opt_filename_desc = "diffLR"
        experiment_params["num_layers_trained_excluding_fc"] = (
            "N/A (all layers, diffLR)"
        )
    elif user_input_train_opt == -3:
        train_opt_filename_desc = "diffLRAug"
        experiment_params["num_layers_trained_excluding_fc"] = (
            "N/A (all layers, diffLR with aug)"
        )

    # Load semi-supervised splits
    labeled_loader, unlabeled_loader, val_loader, test_loader = (
        OxfordPetDataset.get_semi_supervised_loaders(
            data_dir="../data/raw",
            batch_size=32,
            label_fraction=label_fraction_val,
            binary_classification=False,
            data_augmentation=data_augmentation_val,
        )
    )

    # Load model
    print("\nInitializing ResNet50 model...")
    for _ in tqdm(range(5), desc="Loading model"):
        time.sleep(0.2)  # Simulate loading time

    if user_input_train_opt == -2 or user_input_train_opt == -3:
        model = ResNet50(
            binary_classification=False,
            freeze_backbone=False,
            num_train_layers=0,
        )
    else:
        model = ResNet50(
            binary_classification=False,
            freeze_backbone=True,
            num_train_layers=num_layers_to_train_for_model_init,
        )

    device = get_device()

    monitor_grads_choice = input("\nDo you want to monitor gradients? (y/n): ").lower()
    monitor_gradients_val = monitor_grads_choice == "y"
    experiment_params["monitor_gradients"] = monitor_gradients_val
    gradient_monitor_interval_val = 100  # Default
    if monitor_gradients_val:
        try:
            interval = int(input("Monitor gradients every N batches (e.g., 50, 100): "))
            if interval > 0:
                gradient_monitor_interval_val = interval
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")
        experiment_params["gradient_monitor_interval"] = gradient_monitor_interval_val

    finetune_bn_choice = input(
        "\nDo you want to fine-tune batch normalization parameters? (y/n): "
    ).lower()
    finetune_bn_val = finetune_bn_choice == "y"
    experiment_params["finetune_bn"] = finetune_bn_val

    lr_config_val = []
    lr_filename_desc = ""
    if (
        user_input_train_opt == -2 or user_input_train_opt == -3
    ):  # Different learning rates for each layer
        lr_config_val = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
        lr_filename_desc = "diffLRprofile"
        experiment_params["learning_rate_config"] = lr_config_val
    else:
        default_lr = 0.00001
        try:
            lr_input = float(
                input(f"Enter learning rate for both phases (default {default_lr}): ")
                or str(default_lr)
            )
            if lr_input <= 0:
                print(f"Learning rate must be positive. Using default {default_lr}.")
                lr_input = default_lr
        except ValueError:
            print(f"Invalid input for learning rate. Using default {default_lr}.")
            lr_input = default_lr
        lr_config_val = [lr_input]
        lr_filename_desc = f"lr{lr_input}"
        experiment_params["learning_rate_config"] = lr_config_val[0]

    l2_lambda_val = 0.0  # Default for this function
    experiment_params["l2_lambda"] = l2_lambda_val

    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        learning_rate=lr_config_val,
        monitor_gradients=monitor_gradients_val,
        gradient_monitor_interval=(
            gradient_monitor_interval_val if monitor_gradients_val else 100
        ),
        finetune_bn=finetune_bn_val,
        lam=l2_lambda_val,
    )

    print(f"\n{get_swedish_waiting_message()}")

    print("\nPhase 1: Training on labeled subset...")
    if user_input_train_opt == -1:
        print("\nStarting training with Gradual Unfreezing on labeled data...")
        model, _ = trainer.train_gradual_unfreezing(
            labeled_loader, val_loader, num_epochs=num_epochs_labeled, print_graph=True
        )
    else:
        model, _ = trainer.train(
            labeled_loader, val_loader, num_epochs=num_epochs_labeled, print_graph=True
        )

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nPhase 2: Training on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)
    if user_input_train_opt == -1:
        print("\nStarting training with Gradual Unfreezing on combined data...")
        model, _ = trainer.train_gradual_unfreezing(
            combined_loader,
            val_loader,
            num_epochs=num_epochs_combined,
            print_graph=True,
        )
    else:
        model, _ = trainer.train(
            combined_loader,
            val_loader,
            num_epochs=num_epochs_combined,
            print_graph=True,
        )

    save_choice = input("\nDo you want to save the final model? (y/n): ").lower()
    if save_choice == "y":
        name_parts = [
            model_architecture,
            classification_type,
            f"{num_epochs_labeled}l+{num_epochs_combined}c_ep",
            lr_filename_desc,
            f"frac{label_fraction_val:.2f}".replace(".", "p"),
            train_opt_filename_desc,
            f"aug{data_augmentation_val}",
            f"bn{finetune_bn_val}",
            "semi",
        ]
        # Optionally add L2reg and gradmon if they were prompted/non-default and deemed essential for filename
        # For now, sticking to user's request for "essential to identify"

        model_filename_base = "_".join(name_parts)
        model_save_path = os.path.join(
            "..",
            "models",
            model_architecture,
            classification_type,
            model_filename_base + ".pth",
        )
        experiment_params["model_path"] = model_save_path

        print(f"Attempting to save final model to: {model_save_path}")
        trainer.save_model(full_save_path=model_save_path)

        evaluation_output_dir = os.path.join(
            "..",
            "evaluation",
            model_architecture,
            classification_type,
            model_filename_base,
        )
        print(
            f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}"
        )
        # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, experiment_params_dict=experiment_params)

    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")


def run_experiment_2():
    """Run multi-class classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 2: ResNet50 multi-class classification")
    print("=" * 70)
    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    # Moved arch and class type here as they are common if choice is '1'
    model_architecture = "resnet"
    classification_type = "multiclass"
    experiment_params = {}

    if choice == "1":
        training_type_str_suffix = "sup"
        experiment_params["training_type"] = "supervised"

        try:
            num_epochs_r2 = int(
                input("Enter the number of epochs for supervised training (e.g., 3): ")
            )
            if num_epochs_r2 <= 0:
                print("Number of epochs must be positive. Using default: 3.")
                num_epochs_r2 = 3
        except ValueError:
            print("Invalid input for epochs. Using default: 3.")
            num_epochs_r2 = 3
        experiment_params["epochs"] = num_epochs_r2

        # Get number of layers to train / training strategy
        user_input_train_opt_r2 = int(
            input(
                "\nSelect training option: \n n>0: train n layers \n '-1': gradually unfreeze each layer (fixed learning rate) \n '-2': different learning rate for each layer and no data augmentation \n '-3': different learning rates for each layer and data augmentation \n User input: "
            )
        )
        experiment_params["training_option_resnet_e2"] = user_input_train_opt_r2

        num_layers_to_train_val_r2 = 0
        train_opt_filename_desc_r2 = ""
        if user_input_train_opt_r2 > 0:
            train_opt_filename_desc_r2 = f"layers{user_input_train_opt_r2}"
            num_layers_to_train_val_r2 = user_input_train_opt_r2
            experiment_params["num_layers_trained_excluding_fc"] = (
                user_input_train_opt_r2
            )
        elif user_input_train_opt_r2 == -1:
            train_opt_filename_desc_r2 = "gradUnfreeze"
            experiment_params["num_layers_trained_excluding_fc"] = (
                "N/A (gradual unfreeze)"
            )
        elif user_input_train_opt_r2 == -2:
            train_opt_filename_desc_r2 = "diffLR"
            experiment_params["num_layers_trained_excluding_fc"] = (
                "N/A (all layers, diffLR)"
            )
        elif user_input_train_opt_r2 == -3:
            train_opt_filename_desc_r2 = "diffLRAug"
            experiment_params["num_layers_trained_excluding_fc"] = (
                "N/A (all layers, diffLR with aug)"
            )

        data_aug_flag_r2 = user_input_train_opt_r2 == -3
        experiment_params["data_augmentation"] = data_aug_flag_r2

        # Load data
        print("\nLoading Oxford-IIIT Pet Dataset for multi-class classification...")
        train_loader, val_loader, test_loader, num_classes = (
            OxfordPetDataset.get_dataloaders(
                data_dir="../data/raw",
                batch_size=32,
                binary_classification=False,
                data_augmentation=data_aug_flag_r2,
            )
        )
        print(
            f"Dataset loaded successfully! ({len(train_loader.dataset)} training samples)"
        )

        if user_input_train_opt_r2 == -2 or user_input_train_opt_r2 == -3:
            model = ResNet50(
                binary_classification=False,
                freeze_backbone=False,
                num_train_layers=0,
            )
        else:
            model = ResNet50(
                binary_classification=False,
                freeze_backbone=True,
                num_train_layers=num_layers_to_train_val_r2,
            )

        device = get_device()

        monitor_grads_choice_r2 = input(
            "\nDo you want to monitor gradients? (y/n): "
        ).lower()
        monitor_gradients_r2 = monitor_grads_choice_r2 == "y"
        experiment_params["monitor_gradients"] = monitor_gradients_r2
        gradient_monitor_interval_r2 = 100  # Default
        if monitor_gradients_r2:
            try:
                interval_r2 = int(
                    input("Monitor gradients every N batches (e.g., 50, 100): ")
                )
                if interval_r2 > 0:
                    gradient_monitor_interval_r2 = interval_r2
                else:
                    print("Invalid interval, using default 100.")
            except ValueError:
                print("Invalid input, using default interval 100.")
            experiment_params["gradient_monitor_interval"] = (
                gradient_monitor_interval_r2
            )

        finetune_bn_choice_r2 = input(
            "\nDo you want to fine-tune batch normalization parameters? (y/n): "
        ).lower()
        finetune_bn_r2 = finetune_bn_choice_r2 == "y"
        experiment_params["finetune_bn"] = finetune_bn_r2

        actual_lr_config_r2 = []
        lr_filename_desc_r2 = ""
        if user_input_train_opt_r2 == -2 or user_input_train_opt_r2 == -3:
            actual_lr_config_r2 = [
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
            lr_filename_desc_r2 = "diffLRprofile"
            experiment_params["learning_rate_config"] = actual_lr_config_r2
        else:
            default_lr_r2 = 5e-5
            try:
                lr_val_r2 = float(
                    input(f"Enter learning rate (default {default_lr_r2}): ")
                    or str(default_lr_r2)
                )
                if lr_val_r2 <= 0:
                    print(
                        f"Learning rate must be positive. Using default {default_lr_r2}."
                    )
                    lr_val_r2 = default_lr_r2
            except ValueError:
                print(f"Invalid input. Using default {default_lr_r2}.")
                lr_val_r2 = default_lr_r2
            actual_lr_config_r2 = [lr_val_r2]
            lr_filename_desc_r2 = f"lr{lr_val_r2}"
            experiment_params["learning_rate_config"] = actual_lr_config_r2[0]

        l2_lambda_r2 = 0.0  # Default for this experiment flow
        experiment_params["l2_lambda"] = l2_lambda_r2

        trainer = ModelTrainer(
            model,
            device,
            binary_classification=False,
            learning_rate=actual_lr_config_r2,
            monitor_gradients=monitor_gradients_r2,
            gradient_monitor_interval=(
                gradient_monitor_interval_r2 if monitor_gradients_r2 else 100
            ),
            finetune_bn=finetune_bn_r2,
            lam=l2_lambda_r2,
        )

        print(f"\n{get_swedish_waiting_message()}")

        if user_input_train_opt_r2 == -1:
            start_time = time.time()
            model, history = trainer.train_gradual_unfreezing(
                train_loader, val_loader, num_epochs=num_epochs_r2, print_graph=True
            )
            end_time = time.time()
            training_time = end_time - start_time
            print(
                f"Training (gradual unfreezing) completed in {training_time:.2f} seconds."
            )
        else:
            start_time = time.time()
            model, history = trainer.train(
                train_loader, val_loader, num_epochs=num_epochs_r2, print_graph=True
            )
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training completed in {training_time:.2f} seconds.")

        save_choice = input("\nDo you want to save the model? (y/n): ").lower()
        if save_choice == "y":
            name_parts = [
                model_architecture,
                classification_type,
                f"{num_epochs_r2}ep",
                lr_filename_desc_r2,
                train_opt_filename_desc_r2,
                f"aug{data_aug_flag_r2}",
                f"bn{finetune_bn_r2}",
                f"L2reg{l2_lambda_r2}",  # Included for completeness, though 0.0 here
                training_type_str_suffix,  # "sup"
            ]
            if monitor_gradients_r2:
                name_parts.append("gradmon")

            model_filename_base = "_".join(name_parts)
            model_save_path = os.path.join(
                "..",
                "models",
                model_architecture,
                classification_type,
                model_filename_base + ".pth",
            )
            experiment_params["model_path"] = model_save_path

            print(f"Attempting to save model to: {model_save_path}")
            trainer.save_model(full_save_path=model_save_path)

            evaluation_output_dir = os.path.join(
                "..",
                "evaluation",
                model_architecture,
                classification_type,
                model_filename_base,
            )
            print(
                f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}"
            )
            # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir,training_time=training_time, experiment_params_dict=experiment_params)

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
    # num_epochs_vit = 2 # Will be prompted
    batch_size_vit = 32
    model_architecture = "vit"
    classification_type = "binary"
    experiment_params = {}

    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    if choice == "1":
        training_type_str = "sup"
        experiment_params["training_type"] = "supervised"
        try:
            num_epochs_vit = int(
                input("Enter the number of epochs for ViT training (e.g., 2): ")
            )
            if num_epochs_vit <= 0:
                print("Number of epochs must be positive. Using default: 2.")
                num_epochs_vit = 2
        except ValueError:
            print("Invalid input for epochs. Using default: 2.")
            num_epochs_vit = 2
        experiment_params["epochs"] = num_epochs_vit

        # Load data
        train_loader, val_loader, test_loader, _ = OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw",
            batch_size=batch_size_vit,
            binary_classification=True,
            model_type="vit",  # Ensures ViT compatible transforms
            vit_model_name=vit_model_checkpoint,
            data_augmentation=False,  # Explicitly False for this supervised ViT path
        )
        experiment_params["data_augmentation"] = False

        print(
            f"Dataset loaded for ViT binary classification! ({len(train_loader.dataset)} training samples)"
        )

        # Load model
        print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
        model = ViT(model_name_or_path=vit_model_checkpoint, binary_classification=True)
        experiment_params["vit_model_checkpoint"] = vit_model_checkpoint

        device = get_device()

        monitor_grads_choice = input(
            "\nDo you want to monitor gradients? (y/n): "
        ).lower()
        monitor_gradients_vit = monitor_grads_choice == "y"
        experiment_params["monitor_gradients"] = monitor_gradients_vit
        gradient_monitor_interval_vit = 100  # Default
        if monitor_gradients_vit:
            try:
                interval = int(
                    input("Monitor gradients every N batches (e.g., 50, 100): ")
                )
                if interval > 0:
                    gradient_monitor_interval_vit = interval
                else:
                    print("Invalid interval, using default 100.")
            except ValueError:
                print("Invalid input, using default interval 100.")
            experiment_params["gradient_monitor_interval"] = (
                gradient_monitor_interval_vit
            )

        # For ViT, BN fine-tuning is not typically a user-configurable hyperparameter like in ResNets
        finetune_bn_vit = False  # Reflects no specific BN fine-tuning for ViT here
        experiment_params["finetune_bn"] = finetune_bn_vit

        # Learning rate for ViT often needs to be small
        default_lr_vit = 5e-5
        try:
            lr_input_vit = float(
                input(
                    f"Enter learning rate for ViT training (default {default_lr_vit}): "
                )
                or str(default_lr_vit)
            )
            if lr_input_vit <= 0:
                print(
                    f"Learning rate must be positive. Using default {default_lr_vit}."
                )
                lr_input_vit = default_lr_vit
        except ValueError:
            print(f"Invalid input for learning rate. Using default {default_lr_vit}.")
            lr_input_vit = default_lr_vit
        lr_config_vit = [lr_input_vit]
        experiment_params["learning_rate"] = lr_config_vit[0]

        l2_lambda_vit = 0.0  # Default
        experiment_params["l2_lambda"] = l2_lambda_vit

        trainer = ModelTrainer(
            model,
            device,
            binary_classification=True,
            learning_rate=lr_config_vit,
            monitor_gradients=monitor_gradients_vit,
            gradient_monitor_interval=(
                gradient_monitor_interval_vit if monitor_gradients_vit else 100
            ),
            finetune_bn=finetune_bn_vit,  # Passed, though ModelTrainer's BN logic primarily targets ResNet BatchNorm layers
            lam=l2_lambda_vit,
        )

        print(f"\n{get_swedish_waiting_message()}")

        start_time = time.time()
        model, history = trainer.train(
            train_loader, val_loader, num_epochs=num_epochs_vit, print_graph=True
        )
        end_time = time.time()
        # training_time = end_time - start_time # Not adding to params as per new concise request
        # print(f"Training completed in {training_time:.2f} seconds.")

        save_choice = input("\nDo you want to save the model? (y/n): ").lower()
        if save_choice == "y":
            name_parts = [
                model_architecture,  # vit
                classification_type,  # binary
                f"{num_epochs_vit}ep",
                f"lr{lr_config_vit[0]}",
                training_type_str,  # sup
                f"aug{experiment_params['data_augmentation']}",
                f"bn{finetune_bn_vit}",  # Will show bnFalse or bnNA
            ]
            # Optionally add L2reg{l2_lambda_vit} or gradmon if deemed essential for filename

            model_filename_base = "_".join(name_parts)
            model_save_path = os.path.join(
                "..",
                "models",
                model_architecture,
                classification_type,
                model_filename_base + ".pth",
            )
            experiment_params["model_path"] = model_save_path

            print(f"Attempting to save model to: {model_save_path}")
            trainer.save_model(
                full_save_path=model_save_path
            )  # trainer.save_model now expects full_save_path

            evaluation_output_dir = os.path.join(
                "..",
                "evaluation",
                model_architecture,
                classification_type,
                model_filename_base,
            )
            print(
                f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}"
            )
            # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, experiment_params_dict=experiment_params)

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
    """Run ViT binary classification semi-supervised experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment: ViT binary classification (SEMI-SUPERVISED)")
    print("=" * 70)

    vit_model_checkpoint = "google/vit-base-patch16-224"
    batch_size_vit = 32
    model_architecture = "vit"
    classification_type = "binary"
    experiment_params = {"training_type": "semi-supervised"}
    experiment_params["vit_model_checkpoint"] = vit_model_checkpoint

    try:
        num_epochs_labeled_vit = int(
            input("Enter epochs for initial ViT training on labeled data (e.g., 2): ")
        )
        if num_epochs_labeled_vit <= 0:
            print("Number of epochs must be positive. Using default: 2.")
            num_epochs_labeled_vit = 2
    except ValueError:
        print("Invalid input for epochs. Using default: 2.")
        num_epochs_labeled_vit = 2
    experiment_params["epochs_labeled"] = num_epochs_labeled_vit

    try:
        num_epochs_combined_vit = int(
            input("Enter epochs for ViT training on combined data (e.g., 2): ")
        )
        if num_epochs_combined_vit <= 0:
            print("Number of epochs must be positive. Using default: 2.")
            num_epochs_combined_vit = 2
    except ValueError:
        print("Invalid input for epochs. Using default: 2.")
        num_epochs_combined_vit = 2
    experiment_params["epochs_combined"] = num_epochs_combined_vit

    label_fraction_vit = float(
        input("Enter labeled data fraction (e.g., 0.1 for 10%): ")
    )
    experiment_params["label_fraction"] = label_fraction_vit

    # Data augmentation is False for ViT semi-supervised path in dataset.py by default
    data_aug_vit_semi = False
    experiment_params["data_augmentation"] = data_aug_vit_semi

    labeled_loader, unlabeled_loader, val_loader, test_loader = (
        OxfordPetDataset.get_semi_supervised_loaders(
            data_dir="../data/raw",
            batch_size=batch_size_vit,
            label_fraction=label_fraction_vit,
            binary_classification=True,
            model_type="vit",
            vit_model_name=vit_model_checkpoint,
            data_augmentation=data_aug_vit_semi,
        )
    )

    print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
    model = ViT(model_name_or_path=vit_model_checkpoint, binary_classification=True)

    device = get_device()

    monitor_grads_choice_vit = input(
        "\nDo you want to monitor gradients? (y/n): "
    ).lower()
    monitor_gradients_vit_semi = monitor_grads_choice_vit == "y"
    experiment_params["monitor_gradients"] = monitor_gradients_vit_semi
    gradient_monitor_interval_vit_semi = 100  # Default
    if monitor_gradients_vit_semi:
        try:
            interval_vit = int(
                input("Monitor gradients every N batches (e.g., 50, 100): ")
            )
            if interval_vit > 0:
                gradient_monitor_interval_vit_semi = interval_vit
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")
        experiment_params["gradient_monitor_interval"] = (
            gradient_monitor_interval_vit_semi
        )

    finetune_bn_vit_semi = (
        False  # ViT doesn't have typical BN layers for this kind of fine-tuning
    )
    experiment_params["finetune_bn"] = finetune_bn_vit_semi

    default_lr_vit_semi = 5e-5
    try:
        lr_input_vit_semi = float(
            input(
                f"Enter learning rate for ViT (both phases, default {default_lr_vit_semi}): "
            )
            or str(default_lr_vit_semi)
        )
        if lr_input_vit_semi <= 0:
            print(
                f"Learning rate must be positive. Using default {default_lr_vit_semi}."
            )
            lr_input_vit_semi = default_lr_vit_semi
    except ValueError:
        print(f"Invalid input for learning rate. Using default {default_lr_vit_semi}.")
        lr_input_vit_semi = default_lr_vit_semi
    lr_config_vit_semi = [lr_input_vit_semi]
    experiment_params["learning_rate"] = lr_config_vit_semi[0]

    l2_lambda_vit_semi = 0.0  # Default
    experiment_params["l2_lambda"] = l2_lambda_vit_semi

    trainer = ModelTrainer(
        model,
        device,
        binary_classification=True,
        learning_rate=lr_config_vit_semi,
        monitor_gradients=monitor_gradients_vit_semi,
        gradient_monitor_interval=(
            gradient_monitor_interval_vit_semi if monitor_gradients_vit_semi else 100
        ),
        finetune_bn=finetune_bn_vit_semi,
        lam=l2_lambda_vit_semi,
    )

    print(f"\n{get_swedish_waiting_message()}")

    print("\nPhase 1: ViT training on labeled subset...")
    model, _ = trainer.train(
        labeled_loader, val_loader, num_epochs=num_epochs_labeled_vit, print_graph=True
    )

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print("\nPhase 2: ViT training on combined labeled + pseudo-labeled data...")
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)
    model, _ = trainer.train(
        combined_loader,
        val_loader,
        num_epochs=num_epochs_combined_vit,
        print_graph=True,
    )

    save_choice = input("\nDo you want to save the final model? (y/n): ").lower()
    if save_choice == "y":
        name_parts = [
            model_architecture,
            classification_type,
            f"{num_epochs_labeled_vit}l+{num_epochs_combined_vit}c_ep",
            f"lr{lr_config_vit_semi[0]}",
            f"frac{label_fraction_vit:.2f}".replace(".", "p"),
            f"aug{data_aug_vit_semi}",
            f"bn{finetune_bn_vit_semi}",
            "semi",
        ]
        # Add L2reg or gradmon if they become prompted/essential for filename identification

        model_filename_base = "_".join(name_parts)
        model_save_path = os.path.join(
            "..",
            "models",
            model_architecture,
            classification_type,
            model_filename_base + ".pth",
        )
        experiment_params["model_path"] = model_save_path

        print(f"Attempting to save final model to: {model_save_path}")
        trainer.save_model(full_save_path=model_save_path)

        evaluation_output_dir = os.path.join(
            "..",
            "evaluation",
            model_architecture,
            classification_type,
            model_filename_base,
        )
        print(
            f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}"
        )
        # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, experiment_params_dict=experiment_params)

    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")


def run_experiment_vit_multiclass_semi():
    print("\n" + "=" * 70)
    print("Starting experiment: ViT MULTICLASS classification (SEMI-SUPERVISED)")
    print("=" * 70)

    vit_model_checkpoint = "google/vit-base-patch16-224"
    batch_size_vit = 32
    model_architecture = "vit"
    classification_type = "multiclass"
    experiment_params = {"training_type": "semi-supervised"}
    experiment_params["vit_model_checkpoint"] = vit_model_checkpoint

    try:
        num_epochs_labeled_vit = int(
            input("Enter epochs for initial ViT training on labeled data (e.g., 6): ")
        )
        if num_epochs_labeled_vit <= 0:
            print("Number of epochs must be positive. Using default: 6.")
            num_epochs_labeled_vit = 6
    except ValueError:
        print("Invalid input for epochs. Using default: 6.")
        num_epochs_labeled_vit = 6
    try:
        num_epochs_combined_vit = int(
            input("Enter epochs for ViT training on combined data (e.g., 6): ")
        )
        if num_epochs_combined_vit <= 0:
            print("Number of epochs must be positive. Using default: 6.")
            num_epochs_combined_vit = 6
    except ValueError:
        print("Invalid input for epochs. Using default: 6.")
        num_epochs_combined_vit = 6
    experiment_params["epochs_combined"] = num_epochs_combined_vit

    label_fraction_vit_mc = float(
        input("Enter labeled data fraction (e.g., 0.1 for 10%): ")
    )
    experiment_params["label_fraction"] = label_fraction_vit_mc

    # Data augmentation is False for ViT semi-supervised path in dataset.py by default
    data_aug_vit_mc_semi = False
    experiment_params["data_augmentation"] = data_aug_vit_mc_semi

    labeled_loader, unlabeled_loader, val_loader, test_loader = (
        OxfordPetDataset.get_semi_supervised_loaders(
            data_dir="../data/raw",
            batch_size=batch_size_vit,
            label_fraction=label_fraction_vit_mc,
            binary_classification=False,
            model_type="vit",
            vit_model_name=vit_model_checkpoint,
            data_augmentation=data_aug_vit_mc_semi,
        )
    )

    print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
    model = ViT(model_name_or_path=vit_model_checkpoint, binary_classification=False)

    device = get_device()

    monitor_grads_choice_vit_mc = input(
        "\nDo you want to monitor gradients? (y/n): "
    ).lower()
    monitor_gradients_vit_mc_semi = monitor_grads_choice_vit_mc == "y"
    experiment_params["monitor_gradients"] = monitor_gradients_vit_mc_semi
    gradient_monitor_interval_vit_mc_semi = 100
    if monitor_gradients_vit_mc_semi:
        try:
            interval_vit_mc = int(
                input("Monitor gradients every N batches (e.g., 50, 100): ")
            )
            if interval_vit_mc > 0:
                gradient_monitor_interval_vit_mc_semi = interval_vit_mc
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")
        experiment_params["gradient_monitor_interval"] = (
            gradient_monitor_interval_vit_mc_semi
        )

    finetune_bn_vit_mc_semi = False
    experiment_params["finetune_bn"] = finetune_bn_vit_mc_semi

    default_lr_vit_mc_semi = 5e-5
    try:
        lr_input_vit_mc_semi = float(
            input(
                f"Enter learning rate for ViT (both phases, default {default_lr_vit_mc_semi}): "
            )
            or str(default_lr_vit_mc_semi)
        )
        if lr_input_vit_mc_semi <= 0:
            print(
                f"Learning rate must be positive. Using default {default_lr_vit_mc_semi}."
            )
            lr_input_vit_mc_semi = default_lr_vit_mc_semi
    except ValueError:
        print(
            f"Invalid input for learning rate. Using default {default_lr_vit_mc_semi}."
        )
        lr_input_vit_mc_semi = default_lr_vit_mc_semi
    lr_config_vit_mc_semi = [lr_input_vit_mc_semi]
    experiment_params["learning_rate"] = lr_config_vit_mc_semi[0]

    l2_lambda_vit_mc_semi = 0.0
    experiment_params["l2_lambda"] = l2_lambda_vit_mc_semi

    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        learning_rate=lr_config_vit_mc_semi,
        monitor_gradients=monitor_gradients_vit_mc_semi,
        gradient_monitor_interval=(
            gradient_monitor_interval_vit_mc_semi
            if monitor_gradients_vit_mc_semi
            else 100
        ),
        finetune_bn=finetune_bn_vit_mc_semi,
        lam=l2_lambda_vit_mc_semi,
    )

    print(f"\n{get_swedish_waiting_message()}")

    print("\nPhase 1: ViT Multiclass training on labeled subset...")
    model, _ = trainer.train(
        labeled_loader, val_loader, num_epochs=num_epochs_labeled_vit, print_graph=True
    )

    print("\nGenerating pseudo-labels...")
    pseudo_loader = trainer.generate_pseudo_labels(model, unlabeled_loader)

    print(
        "\nPhase 2: ViT Multiclass training on combined labeled + pseudo-labeled data..."
    )
    combined_loader = trainer.combine_loaders(labeled_loader, pseudo_loader)
    model, _ = trainer.train(
        combined_loader,
        val_loader,
        num_epochs=num_epochs_combined_vit,
        print_graph=True,
    )

    save_choice = input("\nDo you want to save the final model? (y/n): ").lower()
    if save_choice == "y":
        name_parts = [
            model_architecture,
            classification_type,
            f"{num_epochs_labeled_vit}l+{num_epochs_combined_vit}c_ep",
            f"lr{lr_config_vit_mc_semi[0]}",
            f"frac{label_fraction_vit_mc:.2f}".replace(".", "p"),
            f"aug{data_aug_vit_mc_semi}",
            f"bn{finetune_bn_vit_mc_semi}",
            "semi",
        ]
        # L2reg and gradmon could be added if they become prompted/non-default

        model_filename_base = "_".join(name_parts)
        model_save_path = os.path.join(
            "..",
            "models",
            model_architecture,
            classification_type,
            model_filename_base + ".pth",
        )
        experiment_params["model_path"] = model_save_path

        print(f"Attempting to save final model to: {model_save_path}")
        trainer.save_model(full_save_path=model_save_path)

        evaluation_output_dir = os.path.join(
            "..",
            "evaluation",
            model_architecture,
            classification_type,
            model_filename_base,
        )
        print(
            f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}"
        )
        # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, experiment_params_dict=experiment_params)

    print("\nEvaluating final model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")


def run_experiment_vit_multiclass_imbalanced():
    print("\n" + "=" * 70)
    print("Starting experiment: ViT MULTICLASS with imbalanced training set")
    print("=" * 70)

    vit_model_checkpoint = "google/vit-base-patch16-224"
    model_architecture = "vit"
    classification_type = "multiclass"
    experiment_params = {"training_type": "supervised_imbalanced_vit"}
    experiment_params["vit_model_checkpoint"] = vit_model_checkpoint

    try:
        num_epochs_vit = int(
            input("Enter the number of epochs for ViT training (e.g., 6): ")
        )
        if num_epochs_vit <= 0:
            print("Number of epochs must be positive. Using default: 6.")
            num_epochs_vit = 6
    except ValueError:
        print("Invalid input for epochs. Using default: 6.")
        num_epochs_vit = 6
    experiment_params["epochs"] = num_epochs_vit

    batch_size_vit = 32
    experiment_params["batch_size"] = batch_size_vit

    print("\nStrategy options to handle imbalance:")
    print("1. No strategy (baseline)")
    print("2. Weighted CrossEntropyLoss")
    print("3. Oversampling minority classes")

    strategy = int(input("Choose strategy (1/2/3): ").strip())
    experiment_params["imbalance_strategy"] = strategy

    training_type_str_detail = f"imbalStrat{strategy}"

    # Data augmentation: For ViT with imbalance, get_dataloaders does not apply augmentation by default.
    # If it were to be added, it would need to be handled in dataset.py or by passing a flag.
    # For now, assume no augmentation unless explicitly added to the get_dataloaders call for this path.
    current_data_augmentation_vit_imbal = False
    experiment_params["data_augmentation"] = current_data_augmentation_vit_imbal

    # Load data
    train_loader, val_loader, test_loader, num_classes = (
        OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw",
            batch_size=batch_size_vit,
            binary_classification=False,  # Multiclass
            model_type="vit",
            vit_model_name=vit_model_checkpoint,
            apply_imbalance=True,
            data_augmentation=current_data_augmentation_vit_imbal,  # Pass this flag
        )
    )
    print(
        f"Dataset loaded for ViT multi-class classification! ({len(train_loader.dataset)} training samples)"
    )

    if strategy == 3:  # Oversampling
        print("Applying oversampling to rebalance classes...")
        train_loader = get_oversampled_loader(
            train_loader.dataset, batch_size=batch_size_vit  # Use batch_size_vit
        )

    print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
    model = ViT(model_name_or_path=vit_model_checkpoint, binary_classification=False)

    device = get_device()

    loss_fn_choice = None
    loss_desc_for_filename = "CE"
    if strategy == 2:  # Weighted CE
        print("Using weighted CrossEntropyLoss...")
        class_weights = compute_class_weights(train_loader.dataset, num_classes).to(
            device
        )
        loss_fn_choice = nn.CrossEntropyLoss(weight=class_weights)
        experiment_params["loss_function_type"] = "weighted_ce"
        loss_desc_for_filename = "WeightedCE"
    else:  # No strategy or Oversampling (uses standard CE)
        loss_fn_choice = nn.CrossEntropyLoss()
        experiment_params["loss_function_type"] = "ce"

    monitor_grads_choice_vit_imbal = input(
        "\nDo you want to monitor gradients? (y/n): "
    ).lower()
    monitor_gradients_vit_imbal = monitor_grads_choice_vit_imbal == "y"
    experiment_params["monitor_gradients"] = monitor_gradients_vit_imbal
    gradient_monitor_interval_vit_imbal = 100
    if monitor_gradients_vit_imbal:
        try:
            interval_vit_imbal = int(
                input("Monitor gradients every N batches (e.g., 50, 100): ")
            )
            if interval_vit_imbal > 0:
                gradient_monitor_interval_vit_imbal = interval_vit_imbal
            else:
                print("Invalid interval, using default 100.")
        except ValueError:
            print("Invalid input, using default interval 100.")
        experiment_params["gradient_monitor_interval"] = (
            gradient_monitor_interval_vit_imbal
        )

    finetune_bn_vit_imbal = False  # ViT doesn't have typical BN layers
    experiment_params["finetune_bn"] = finetune_bn_vit_imbal

    default_lr_vit_imbal = 5e-5
    try:
        lr_input_vit_imbal = float(
            input(
                f"Enter learning rate for ViT training (default {default_lr_vit_imbal}): "
            )
            or str(default_lr_vit_imbal)
        )
        if lr_input_vit_imbal <= 0:
            print(
                f"Learning rate must be positive. Using default {default_lr_vit_imbal}."
            )
            lr_input_vit_imbal = default_lr_vit_imbal
    except ValueError:
        print(f"Invalid input for learning rate. Using default {default_lr_vit_imbal}.")
        lr_input_vit_imbal = default_lr_vit_imbal
    lr_config_vit_imbal = [lr_input_vit_imbal]
    experiment_params["learning_rate"] = lr_config_vit_imbal[0]

    l2_lambda_vit_imbal = 0.0  # Default
    experiment_params["l2_lambda"] = l2_lambda_vit_imbal

    trainer = ModelTrainer(
        model,
        device,
        binary_classification=False,
        learning_rate=lr_config_vit_imbal,
        monitor_gradients=monitor_gradients_vit_imbal,
        gradient_monitor_interval=(
            gradient_monitor_interval_vit_imbal if monitor_gradients_vit_imbal else 100
        ),
        loss_fn=loss_fn_choice,  # Pass the chosen loss function
        finetune_bn=finetune_bn_vit_imbal,
        lam=l2_lambda_vit_imbal,
    )

    print(f"\n{get_swedish_waiting_message()}")

    # start_time = time.time() # No time tracking for params
    model, _ = trainer.train(
        train_loader, val_loader, num_epochs=num_epochs_vit, print_graph=True
    )
    # end_time = time.time()
    # training_time = end_time - start_time
    # print(f"Training completed in {training_time:.2f} seconds.")

    save_choice = input("\nDo you want to save the model? (y/n): ").lower()
    if save_choice == "y":
        name_parts = [
            model_architecture,  # vit
            classification_type,  # multiclass
            f"{num_epochs_vit}ep",
            f"lr{lr_config_vit_imbal[0]}",
            training_type_str_detail,  # e.g., imbalStrat1
            f"loss{loss_desc_for_filename}",
            f"aug{current_data_augmentation_vit_imbal}",  # augFalse
            f"bn{finetune_bn_vit_imbal}",  # bnFalse
        ]
        if monitor_gradients_vit_imbal:
            name_parts.append("gradmon")
        # L2reg not added to filename if 0.0 by default

        model_filename_base = "_".join(name_parts)
        model_save_path = os.path.join(
            "..",
            "models",
            model_architecture,
            classification_type,
            model_filename_base + ".pth",
        )
        experiment_params["model_path"] = model_save_path

        print(f"Attempting to save model to: {model_save_path}")
        trainer.save_model(full_save_path=model_save_path)

        evaluation_output_dir = os.path.join(
            "..",
            "evaluation",
            model_architecture,
            classification_type,
            model_filename_base,
        )
        print(
            f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}"
        )
        # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, experiment_params_dict=experiment_params)

    print("\nEvaluating ViT model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("\nFinal Test Results (ViT Multi-class Imbalanced):")  # Clarified title
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


def run_experiment_vit_multiclass():
    """Run ViT multi-class classification experiment"""
    print("\n" + "=" * 70)
    print("Starting experiment 4: ViT multi-class classification (37 Breeds)")
    print("=" * 70)

    choice = input("Choose training type:\n1. Supervised\n2. Semi-supervised\n> ")
    model_architecture = "vit"
    classification_type = "multiclass"
    experiment_params = {}

    if choice == "1":
        training_type_str = "sup"
        experiment_params["training_type"] = "supervised"
        vit_model_checkpoint = "google/vit-base-patch16-224"
        experiment_params["vit_model_checkpoint"] = vit_model_checkpoint
        batch_size_vit = 32  # Adjust based on GPU memory
        experiment_params["batch_size"] = batch_size_vit

        try:
            num_epochs_vit_mc = int(
                input("Enter the number of epochs for ViT training (default: 6): ")
            )
            if num_epochs_vit_mc <= 0:
                print("Number of epochs must be positive. Using default: 6.")
                num_epochs_vit_mc = 6
        except ValueError:
            print("Invalid input for epochs. Using default: 6.")
            num_epochs_vit_mc = 6
        experiment_params["epochs"] = num_epochs_vit_mc

        # Data augmentation is False for this supervised ViT path
        data_augmentation_vit_mc = False
        experiment_params["data_augmentation"] = data_augmentation_vit_mc

        # Load data
        train_loader, val_loader, test_loader, _ = OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw",
            batch_size=batch_size_vit,
            binary_classification=False,  # Multiclass
            model_type="vit",
            vit_model_name=vit_model_checkpoint,
            data_augmentation=data_augmentation_vit_mc,  # Pass the flag
        )
        print(
            f"Dataset loaded for ViT multi-class classification! ({len(train_loader.dataset)} training samples)"
        )

        # Load model
        print(f"\nInitializing ViT model ({vit_model_checkpoint})...")
        model = ViT(
            model_name_or_path=vit_model_checkpoint, binary_classification=False
        )

        device = get_device()

        monitor_grads_choice_mc = input(
            "\nDo you want to monitor gradients? (y/n): "
        ).lower()
        monitor_gradients_vit_mc = monitor_grads_choice_mc == "y"
        experiment_params["monitor_gradients"] = monitor_gradients_vit_mc
        gradient_monitor_interval_vit_mc = 100  # Default
        if monitor_gradients_vit_mc:
            try:
                interval_mc = int(
                    input("Monitor gradients every N batches (e.g., 50, 100): ")
                )
                if interval_mc > 0:
                    gradient_monitor_interval_vit_mc = interval_mc
                else:
                    print("Invalid interval, using default 100.")
            except ValueError:
                print("Invalid input, using default interval 100.")
            experiment_params["gradient_monitor_interval"] = (
                gradient_monitor_interval_vit_mc
            )

        # For ViT, BN fine-tuning is not typically a user-configurable hyperparameter
        finetune_bn_vit_mc = False
        experiment_params["finetune_bn"] = finetune_bn_vit_mc

        default_lr_vit_mc = 5e-5
        try:
            lr_input_vit_mc = float(
                input(
                    f"Enter learning rate for ViT training (default {default_lr_vit_mc}): "
                )
                or str(default_lr_vit_mc)
            )
            if lr_input_vit_mc <= 0:
                print(
                    f"Learning rate must be positive. Using default {default_lr_vit_mc}."
                )
                lr_input_vit_mc = default_lr_vit_mc
        except ValueError:
            print(
                f"Invalid input for learning rate. Using default {default_lr_vit_mc}."
            )
            lr_input_vit_mc = default_lr_vit_mc
        lr_config_vit_mc = [lr_input_vit_mc]
        experiment_params["learning_rate"] = lr_config_vit_mc[0]

        l2_lambda_vit_mc = 0.0  # Default for ViT supervised
        experiment_params["l2_lambda"] = l2_lambda_vit_mc

        trainer = ModelTrainer(
            model,
            device,
            binary_classification=False,  # Multiclass
            learning_rate=lr_config_vit_mc,
            monitor_gradients=monitor_gradients_vit_mc,
            gradient_monitor_interval=(
                gradient_monitor_interval_vit_mc if monitor_gradients_vit_mc else 100
            ),
            finetune_bn=finetune_bn_vit_mc,  # Passed, though ModelTrainer's BN logic primarily targets ResNet
            lam=l2_lambda_vit_mc,
        )

        print(f"\n{get_swedish_waiting_message()}")

        start_time = time.time()
        model, history = trainer.train(
            train_loader, val_loader, num_epochs=num_epochs_vit_mc, print_graph=True
        )
        end_time = time.time()
        training_time = (
            end_time - start_time
        )  # Measured, but not passed to evaluation for ViT
        print(f"Training completed in {training_time:.2f} seconds.")

        save_choice = input("\nDo you want to save the model? (y/n): ").lower()
        if save_choice == "y":
            name_parts = [
                model_architecture,
                classification_type,
                f"{num_epochs_vit_mc}ep",
                f"lr{lr_config_vit_mc[0]}",
                training_type_str,  # "sup"
                f"aug{data_augmentation_vit_mc}",  # augFalse
                f"bn{finetune_bn_vit_mc}",  # bnFalse
            ]
            # L2reg not added to filename if 0.0
            if monitor_gradients_vit_mc:
                name_parts.append("gradmon")

            model_filename_base = "_".join(name_parts)
            model_save_path = os.path.join(
                "..",
                "models",
                model_architecture,
                classification_type,
                model_filename_base + ".pth",
            )
            experiment_params["model_path"] = model_save_path

            print(f"Attempting to save model to: {model_save_path}")
            trainer.save_model(full_save_path=model_save_path)

            evaluation_output_dir = os.path.join(
                "..",
                "evaluation",
                model_architecture,
                classification_type,
                model_filename_base,
            )
            print(
                f"Placeholder: Call evaluation for {model_save_path}, output to {evaluation_output_dir}"
            )
            # evaluation.perform_and_save_evaluation(model_path=model_save_path, evaluation_output_dir=evaluation_output_dir, experiment_params_dict=experiment_params)

        print("\nEvaluating ViT model on test set...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        print("\nFinal Test Results (ViT Multi-class):")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    elif choice == "2":
        run_experiment_vit_multiclass_semi()  # Calls the existing semi-supervised function
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
