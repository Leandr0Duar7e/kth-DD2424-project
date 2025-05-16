from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
import os
from PIL import Image
import pandas as pd
import torch


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

        return image, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def _get_transforms():
        """
        Get image transformations for the Oxford Pet Dataset
        Returns:
            transforms.Compose: Image transformations for ResNet50
        """
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
    def get_dataloaders(cls, data_dir, batch_size=32, binary_classification=True):
        """
        Get train, validation and test dataloaders for the Oxford Pet Dataset

        Args:
            data_dir: Path to the dataset directory
            batch_size: Batch size for the dataloaders
            binary_classification: If True, convert labels to binary (dog=1, cat=0)

        Returns:
            train_loader: DataLoader for training data (80% of dataset)
            val_loader: DataLoader for validation data (10% of dataset)
            test_loader: DataLoader for test data (10% of dataset)
            num_classes: Number of classes (1 for binary, 37 for multi-class)
        """
        transform = cls._get_transforms()
        dataset = cls(
            root_dir=data_dir,
            transform=transform,
            binary_classification=binary_classification,
        )

        # Calculate split sizes
        total_size = len(dataset)
        print(f"Total size of dataset: {total_size}")
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Split into train, validation and test sets (80-10-10 split)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_classes = 1 if binary_classification else 37
        return train_loader, val_loader, test_loader, num_classes
