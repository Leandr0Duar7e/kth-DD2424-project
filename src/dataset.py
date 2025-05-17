from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
import os
from PIL import Image
import pandas as pd
import torch
from transformers import AutoImageProcessor


class OxfordPetDataset(Dataset):
    """
    Oxford Pet Dataset loader

    Args:
        root_dir: Path to the dataset directory containing 'images' and 'annotations' folders
        transform: Optional transform to be applied on a sample
        binary_classification: If True, convert labels to binary (dog=1, cat=0)
    """

    def __init__(self, root_dir, transform=None, binary_classification=True):
        self.root_dir = root_dir
        self.transform = transform if transform else self._get_transforms()
        self.binary_classification = binary_classification

        # Load the annotations file
        annotations_file = os.path.join(root_dir, "annotations", "list.txt")
        self.data = []

        with open(annotations_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:  # Image CLASS-ID SPECIES BREED-ID
                    image_name = parts[0]
                    class_id = int(parts[1])
                    species_id = int(parts[2])
                    breed_id = int(parts[3])

                    if binary_classification:
                        label = 1 if species_id == 2 else 0  #  (dog=1, cat=0)
                    else:
                        label = class_id - 1  # 0-36 classes
                    self.data.append((image_name, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root_dir, "images", img_name + ".jpg")

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # If image is corrupted, return a black image
            print(f"Corrupted image: {img_path}")
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        # Determine label dtype based on classification type and model target
        # BCEWithLogitsLoss (binary) expects float labels. CrossEntropyLoss (multi-class) expects long labels.
        label_dtype = torch.float32 if self.binary_classification else torch.long
        return image, torch.tensor(label, dtype=label_dtype)

    @staticmethod
    def _get_transforms(data_augmentation=False):
        """
        Get image transformations for the Oxford Pet Dataset
        Args:
            data_augmentation: If True, returns transforms with augmentation for training
        Returns:
            transforms.Compose: Image transformations for ResNet50
        """
        if data_augmentation:
            return transforms.Compose(
                [
                    transforms.Resize(256),  # Resize to slightly larger size
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip
                    transforms.RandomRotation(10),  # Random rotation up to 10 degrees
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # ImageNet normalization
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),  # ResNet50 expects 224x224 input
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # ImageNet normalization
                ]
            )

    @classmethod
    def get_dataloaders(
        cls,
        data_dir,
        batch_size=32,
        binary_classification=True,
        data_augmentation=False,
        model_type="resnet",
        vit_model_name="google/vit-base-patch16-224",
    ):
        """
        Get train, validation and test dataloaders for the Oxford Pet Dataset

        Args:
            data_dir: Path to the dataset directory
            batch_size: Batch size for the dataloaders
            binary_classification: If True, convert labels to binary (dog=1, cat=0)
            model_type (str): "resnet" or "vit". Determines preprocessing.
            vit_model_name (str): Hugging Face model name if model_type is "vit".

        Returns:
            train_loader: DataLoader for training data (80% of dataset)
            val_loader: DataLoader for validation data (10% of dataset)
            test_loader: DataLoader for test data (10% of dataset)
            num_classes: Number of classes (1 for binary, 37 for multi-class)
        """
        current_transform = None
        if model_type == "vit":
            if not vit_model_name:
                raise ValueError("vit_model_name must be provided for ViT model type")
            print(f"\nUsing ViT image processor for {vit_model_name}...")
            image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
            # The transform will take a PIL image and return processed pixel_values tensor
            # Squeeze(0) removes the batch dimension added by default by the processor when processing a single image.
            current_transform = lambda pil_img: image_processor(
                images=pil_img, return_tensors="pt"
            )["pixel_values"].squeeze(0)
        elif model_type == "resnet":
            print("\nUsing ResNet image transforms...")
            current_transform = cls._get_transforms()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        dataset = cls(
            root_dir=data_dir,
            transform=current_transform,  # Pass the selected transform/processor
            binary_classification=binary_classification,
        )

        dataset = OxfordPetDataset(
            root_dir=data_dir,
            transform=transform,
            binary_classification=binary_classification,
        )

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Split into train, validation and test sets (80-10-10 split)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        if data_augmentation:
            train_transform = cls.get_transforms(data_augmentation=True)
            train_dataset.transform = train_transform

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_classes = 1 if binary_classification else 37
        return train_loader, val_loader, test_loader, num_classes
