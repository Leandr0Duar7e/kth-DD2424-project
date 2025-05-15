import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(
        self, binary_classification=True, freeze_backbone=True, num_train_layers=None
    ):
        super(ResNet50, self).__init__()

        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Replace the final layer
        num_features = self.backbone.fc.in_features
        num_classes = (
            1 if binary_classification else 37
        )  # 37 pet breeds in Oxford dataset
        self.backbone.fc = nn.Linear(num_features, num_classes)

        if freeze_backbone:
            # Freeze all backbone layers initially
            for param in self.backbone.parameters():
                param.requires_grad = False

            if num_train_layers is not None:
                # Get all layers in the backbone
                layers = list(self.backbone.children())[:-1]

                num_layers_to_unfreeze = min(num_train_layers, len(layers))

                # Unfreeze the specified number of layers from the end
                for layer in layers[-num_layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # Final layer always trained
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def get_ResNet50_model(binary_classification, freeze_backbone, num_train_layers=None):
    """
    Get a ResNet50 model for pet classification
    Args:
        binary_classification (bool): If True, model outputs binary classification (dog/cat)
        freeze_backbone (bool): If True, backbone layers are frozen by default
        num_train_layers (int, optional): Number of layers from the end to unfreeze for training
    Returns:
        ResNet50: Configured ResNet50 model
    """
    return ResNet50(binary_classification, freeze_backbone, num_train_layers)
