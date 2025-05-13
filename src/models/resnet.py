import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, binary_classification=True, freeze_backbone=True):
        super(ResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        if freeze_backbone:
            # Freeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        num_classes = 1 if binary_classification else 37  # 37 pet breeds in Oxford dataset
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Final (and new) layer always trained
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return self.backbone(x)

def get_ResNet50_model(binary_classification, freeze_backbone):
    
    return ResNet50(binary_classification, freeze_backbone)

