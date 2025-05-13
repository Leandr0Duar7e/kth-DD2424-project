from models.resnet import get_ResNet50_model
from training.trainer import train_model_with_adam, evaluate_model
from data.dataset import get_oxford_pet_dataloaders
import torch
if __name__ == "__main__":
    
    print("\nList of available experiments on Oxford-IIIT Pet Dataset: ")
    print("1. ResNet50 binary classification with Adam optimizer")
    print("2. ...")
    
    experiment_number = int(input("\nEnter the number of the experiment you want to run: "))
    
    # Get data loaders and number of classes
    train_loader, val_loader, test_loader, num_classes = get_oxford_pet_dataloaders(
        data_dir="../data/raw", 
        batch_size=32,
        binary_classification=True
    )
    
    match experiment_number:
        
        case 1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            
            
            model = get_ResNet50_model(binary_classification=True, freeze_backbone=True)
            model, history = train_model_with_adam(model, device, train_loader, val_loader, binary_classification=True, num_epochs=3)
            
            # Print final test results
            test_loss, test_acc = evaluate_model(model, device, test_loader, binary_classification=True)
            print("\nFinal Test Results:")
            print(f'Test Loss: {test_loss}, '
                f'Test Acc: {test_acc}%')
            
        case 2:
            pass
    