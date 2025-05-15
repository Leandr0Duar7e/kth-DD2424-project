from models.resnet import get_ResNet50_model
from trainer import train_model_with_adam, evaluate_model
from dataset import get_oxford_pet_dataloaders
import torch
import torch.nn as nn

if __name__ == "__main__":

    print("\nList of available experiments on Oxford-IIIT Pet Dataset: ")
    print("1. ResNet50 binary classification with Adam optimizer (E.1)")
    print("2. Multi-class classification of dogs and cats with ResNet50 (E.2)")
    print("3. ")

    experiment_number = int(
        input("\nEnter the number of the experiment you want to run: ")
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\nGPU available, using GPU")
    except:
        print("\nNo GPU available, using CPU")
        device = torch.device("cpu")

    match experiment_number:

        case 1:
            print(
                "\nStart of experiment 1: fine-tuning ResNet50 binary classification with Adam optimizer, the backbone is frozen and only the last layer is trained"
            )

            print("\n Loading data...")
            train_loader, val_loader, test_loader, num_classes = (
                get_oxford_pet_dataloaders(
                    data_dir="../data/raw", batch_size=32, binary_classification=True
                )
            )

            print("Insert the number of layers to train (last layer excluded): ")
            num_layers = int(input())
            print("\n Loading model...")
            model = get_ResNet50_model(
                binary_classification=True,
                freeze_backbone=True,
                num_train_layers=num_layers,
            )

            model, history = train_model_with_adam(
                model,
                device,
                train_loader,
                val_loader,
                binary_classification=True,
                num_epochs=3,
            )

            # Print final test results
            criterion = nn.BCEWithLogitsLoss()
            test_loss, test_acc = evaluate_model(
                model, device, test_loader, criterion, binary_classification=True
            )
            print("\nFinal Test Results:")
            print(f"Test Loss: {test_loss:.4f}, " f"Test Acc: {test_acc:.4f}%")

        case 2:
            print(
                "\nStart of experiment 2: fine-tuning ResNet50 multi-class classification with Adam optimizer, how many layers to train?"
            )
            num_train_layers = int(input())

            print("\n Loading data...")
            train_loader, val_loader, test_loader, num_classes = (
                get_oxford_pet_dataloaders(
                    data_dir="../data/raw", batch_size=32, binary_classification=False
                )
            )

            print(" Loading model..")
            model = get_ResNet50_model(
                binary_classification=False,
                freeze_backbone=False,
                num_train_layers=num_train_layers,
            )

            model, history = train_model_with_adam(
                model,
                device,
                train_loader,
                val_loader,
                binary_classification=False,
                num_epochs=2,
            )

            # Print final test results
            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate_model(
                model, device, test_loader, criterion, binary_classification=False
            )
            print("\nFinal Test Results:")
            print(f"Test Loss: {test_loss:.4f}, " f"Test Acc: {test_acc:.2f}%")

            pass
