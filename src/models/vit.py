import torch.nn as nn
from transformers import ViTForImageClassification


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for pet classification.
    Uses a pre-trained model from Hugging Face Transformers.

    Args:
        model_name_or_path (str): Identifier for the pre-trained ViT model.
        binary_classification (bool): If True, configures for binary (1 output),
                                     else for multi-class (37 outputs).
        freeze_backbone (bool): If True, freeze the weights of the ViT backbone.
        num_train_encoder_layers (int): Number of final encoder layers to unfreeze.
                                        Only active if freeze_backbone is True.
                                        0 means only classifier is trained.
    """

    def __init__(
        self,
        model_name_or_path="google/vit-base-patch16-224",
        binary_classification=True,
        freeze_backbone=False,
        num_train_encoder_layers=0,
    ):
        super(ViT, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.binary_classification = binary_classification
        self.num_labels = 1 if self.binary_classification else 37
        self.freeze_backbone = freeze_backbone
        self.num_train_encoder_layers = num_train_encoder_layers

        # Load the pre-trained ViT model with a new classifier head
        self.vit_model = ViTForImageClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )

        if self.freeze_backbone:
            # First, freeze all parameters in the ViT backbone (embeddings + all encoder layers)
            for param in self.vit_model.vit.parameters():
                param.requires_grad = False

            # Then, unfreeze the specified number of final encoder layers
            if self.num_train_encoder_layers > 0:
                # The encoder layers are in self.vit_model.vit.encoder.layer (a ModuleList)
                # Ensure num_train_encoder_layers is not more than available layers
                num_encoder_layers_total = len(self.vit_model.vit.encoder.layer)
                layers_to_unfreeze = min(
                    self.num_train_encoder_layers, num_encoder_layers_total
                )

                for i in range(layers_to_unfreeze):
                    # Unfreeze layers from the end: layer[-(i+1)]
                    for param in self.vit_model.vit.encoder.layer[
                        -(i + 1)
                    ].parameters():
                        param.requires_grad = True

        # Ensure the classifier parameters are always trainable (it's re-initialized)
        for param in self.vit_model.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        """
        Forward pass through the ViT model.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).
                                        Should be preprocessed by the ViT's image processor.
        Returns:
            torch.Tensor: Logits from the classifier.
        """
        outputs = self.vit_model(pixel_values=pixel_values)
        return outputs.logits


# Example of how to get the image processor (will be used in dataset.py)
# from transformers import AutoImageProcessor
# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
