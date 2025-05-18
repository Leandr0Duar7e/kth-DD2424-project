from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
import os
from PIL import Image
import pandas as pd
import torch
import random
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
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
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
        random_seed=42,
    ):
        """
        Get train, validation and test dataloaders for the Oxford Pet Dataset

        Args:
            data_dir: Path to the dataset directory
            batch_size: Batch size for the dataloaders
            binary_classification: If True, convert labels to binary (dog=1, cat=0)
            data_augmentation: If True, applies augmentation to the training set (ResNet only)
            model_type (str): "resnet" or "vit". Determines preprocessing.
            vit_model_name (str): Hugging Face model name if model_type is "vit".
            random_seed (int): Seed for the random number generator for splitting.

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
            current_transform = lambda pil_img: image_processor(
                images=pil_img, return_tensors="pt"
            )["pixel_values"].squeeze(
                0
            )  # Squeeze(0) removes the batch dimension added by default by the processor when processing a single image.
        elif model_type == "resnet":
            print(
                "\nUsing ResNet image transforms (initially non-augmented for split)..."
            )
            # Base transform for the dataset to be split (val/test will use this)
            current_transform = cls._get_transforms(data_augmentation=False)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Main dataset object using the determined transform (non-augmented for ResNet here)
        # This dataset instance will be used by val_dataset and test_dataset subsets.
        dataset_for_splitting = cls(
            root_dir=data_dir,
            transform=current_transform,
            binary_classification=binary_classification,
        )

        # Calculate split sizes
        total_size = len(dataset_for_splitting)
        print(f"Total size of dataset: {total_size}")
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Split into train, validation and test sets (80-10-10 split)
        generator = torch.Generator().manual_seed(random_seed)
        train_subset, val_subset, test_subset = random_split(
            dataset_for_splitting,
            [train_size, val_size, test_size],
            generator=generator,
        )

        # Prepare the final training dataset for the DataLoader
        final_train_dataset = train_subset

        if data_augmentation and model_type == "resnet":
            print("\nApplying data augmentation to the ResNet training set...")
            # Create a new OxfordPetDataset instance with augmented transforms
            augmented_dataset_source = cls(
                root_dir=data_dir,
                transform=cls._get_transforms(data_augmentation=True),
                binary_classification=binary_classification,
            )
            # Create a new Subset using the indices from the original train_subset,
            # but pointing to the newly created augmented_dataset_source.
            final_train_dataset = torch.utils.data.Subset(
                augmented_dataset_source, train_subset.indices
            )

        # Create data loaders
        train_loader = DataLoader(
            final_train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        num_classes = 1 if binary_classification else 37
        return train_loader, val_loader, test_loader, num_classes

    @classmethod
    def get_semi_supervised_loaders(
        cls,
        data_dir,
        batch_size=32,
        label_fraction=0.1,
        binary_classification=True,
        data_augmentation=False,
        model_type="resnet",
        vit_model_name="google/vit-base-patch16-224",
    ):
        """
        Return (labeled_loader, unlabeled_loader, val_loader, test_loader) for semi-supervised training
        """

        # --- 1. Choose the base transform ---
        if model_type == "vit":
            print(f"\nUsing ViT image processor for {vit_model_name}...")
            image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
            base_transform = lambda pil_img: image_processor(
                images=pil_img, return_tensors="pt"
            )["pixel_values"].squeeze(0)
        else:
            print("\nUsing ResNet image transforms...")
            base_transform = cls._get_transforms(data_augmentation=False)

        # --- 2. Create a single dataset for splitting ---
        full_dataset = cls(
            root_dir=data_dir,
            transform=base_transform,
            binary_classification=binary_classification,
        )

        # --- 3. Shuffle and split indices ---
        total_size = len(full_dataset)
        indices = list(range(total_size))
        torch.manual_seed(42)
        torch.random.manual_seed(42)
        random.shuffle(indices)

        labeled_size = int(label_fraction * total_size)
        labeled_indices = indices[:labeled_size]
        unlabeled_indices = indices[labeled_size : int(0.8 * total_size)]
        val_indices = indices[int(0.8 * total_size) : int(0.9 * total_size)]
        test_indices = indices[int(0.9 * total_size) :]

        # --- 4. Apply optional augmentation to labeled set only (for ResNet only) ---
        if model_type == "resnet" and data_augmentation:
            print("\nApplying data augmentation to labeled ResNet training set...")
            augment_transform = cls._get_transforms(data_augmentation=True)
            augmented_dataset = cls(
                root_dir=data_dir,
                transform=augment_transform,
                binary_classification=binary_classification,
            )
            labeled_dataset = torch.utils.data.Subset(
                augmented_dataset, labeled_indices
            )
        else:
            labeled_dataset = torch.utils.data.Subset(full_dataset, labeled_indices)

        unlabeled_dataset = torch.utils.data.Subset(full_dataset, unlabeled_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        # --- 5. Create loaders ---
        labeled_loader = DataLoader(
            labeled_dataset, batch_size=batch_size, shuffle=True
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return labeled_loader, unlabeled_loader, val_loader, test_loader
