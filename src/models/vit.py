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
    """

    def __init__(
        self,
        model_name_or_path="google/vit-base-patch16-224",
        binary_classification=True,
    ):
        super(ViT, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.binary_classification = binary_classification
        self.num_labels = 1 if self.binary_classification else 37

        # Load the pre-trained ViT model with a new classifier head
        self.vit_model = ViTForImageClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,  # Allows loading pre-trained weights for the body
            # while re-initializing the classifier head.
        )

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
