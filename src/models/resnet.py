import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn


class ResNet50(nn.Module):
    """
    ResNet50 model for pet classification

    Attributes:
        backbone (torchvision.models.ResNet): The ResNet50 model

    Args:
        binary_classification (bool): If True, model outputs binary classification (dog/cat)
        freeze_backbone (bool): If True, backbone layers are frozen by default
        num_train_layers (int, optional): Number of layers from the end to unfreeze for training
    """

    def __init__(
        self, binary_classification=False, freeze_backbone=False, num_train_layers=None
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

            if num_train_layers is not None and num_train_layers > 0:
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
        """Forward pass through the network"""
        return self.backbone(x)
    
    def get_index_weighted_layers(self, finetune_bn=True):
        layers = list(self.backbone.children())

        if finetune_bn:
            param_layers = [i for i, layer in enumerate(layers) if any(True for _ in layer.parameters())]
        else:
            param_layers = []
            
            for i, layer in enumerate(layers):
                if not (isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d)) and any(True for _ in layer.parameters()):
                    param_layers.append(i)


        return param_layers


