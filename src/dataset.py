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

    @staticmethod
    def _get_vit_augment_transforms():
        """
        Returns a Compose object for ViT data augmentation.
        These transforms are applied to PIL images before the AutoImageProcessor.
        """
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(degrees=10),
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
        apply_imbalance=False,
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
        image_processor = None  # Initialize image_processor to be accessible later

        if model_type == "vit":
            if not vit_model_name:
                raise ValueError("vit_model_name must be provided for ViT model type")
            print(f"\nUsing ViT image processor for {vit_model_name}...")
            image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
            # Base transform for ViT (val/test and initial train_subset) - no augmentation here
            current_transform = lambda pil_img: image_processor(
                images=pil_img, return_tensors="pt"
            )["pixel_values"].squeeze(0)
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

        # Only include 20% of images from each cat breed (classes 0–11) to simulate imbalance
        if apply_imbalance and not binary_classification:
            print(
                "Applying class imbalance: keeping 20% of cat breed samples in training..."
            )

            def create_imbalanced_subset(train_subset, full_dataset, keep_fraction=0.2):
                from collections import defaultdict
                import random

                class_to_indices = defaultdict(list)
                for idx in train_subset.indices:
                    _, label = full_dataset[idx]
                    class_to_indices[int(label)].append(idx)

                imbalanced_indices = []
                for class_id, indices in class_to_indices.items():
                    if class_id < 12:  # Classes 0–11 are cat breeds
                        n_keep = max(1, int(len(indices) * keep_fraction))
                        imbalanced_indices.extend(random.sample(indices, n_keep))
                    else:
                        imbalanced_indices.extend(indices)

                return torch.utils.data.Subset(full_dataset, imbalanced_indices)

            final_train_dataset = create_imbalanced_subset(
                train_subset, dataset_for_splitting
            )
        else:
            final_train_dataset = train_subset

        # Apply data augmentation if specified, after splitting
        if data_augmentation:
            if model_type == "resnet":
                print("\nApplying data augmentation to the ResNet training set...")
                augmented_dataset_source_resnet = cls(
                    root_dir=data_dir,
                    transform=cls._get_transforms(
                        data_augmentation=True
                    ),  # ResNet's own augment + ToTensor + Normalize
                    binary_classification=binary_classification,
                )
                final_train_dataset = torch.utils.data.Subset(
                    augmented_dataset_source_resnet, train_subset.indices
                )
            elif model_type == "vit":
                print("\nApplying data augmentation to the ViT training set...")
                vit_augment_chain = (
                    cls._get_vit_augment_transforms()
                )  # PIL -> PIL augmentations

                # Create a transform that applies PIL augmentations THEN the ViT processor
                augmented_transform_for_vit_source = lambda pil_img: image_processor(
                    images=vit_augment_chain(pil_img), return_tensors="pt"
                )["pixel_values"].squeeze(0)

                augmented_dataset_source_vit = cls(
                    root_dir=data_dir,
                    transform=augmented_transform_for_vit_source,
                    binary_classification=binary_classification,
                )
                final_train_dataset = torch.utils.data.Subset(
                    augmented_dataset_source_vit, train_subset.indices
                )

        # Create data loaders
        train_loader = DataLoader(
            final_train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Determine num_classes to return
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
        image_processor = None  # Initialize image_processor

        # --- 1. Choose the base transform (non-augmented) for the full_dataset ---
        if model_type == "vit":
            print(f"\nUsing ViT image processor for {vit_model_name}...")
            image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
            base_transform = lambda pil_img: image_processor(
                images=pil_img, return_tensors="pt"
            )["pixel_values"].squeeze(0)
        else:  # ResNet
            print(
                "\nUsing ResNet image transforms (non-augmented for initial full_dataset)..."
            )
            base_transform = cls._get_transforms(data_augmentation=False)

        # --- 2. Create a single dataset for splitting (using non-augmented transform) ---
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

        # --- 4. Apply optional augmentation to labeled set only ---
        if data_augmentation:
            if model_type == "resnet":
                print("\nApplying data augmentation to labeled ResNet training set...")
                augment_transform_resnet = cls._get_transforms(data_augmentation=True)
                augmented_dataset_resnet = cls(
                    root_dir=data_dir,
                    transform=augment_transform_resnet,
                    binary_classification=binary_classification,
                )
                labeled_dataset = torch.utils.data.Subset(
                    augmented_dataset_resnet, labeled_indices
                )
            elif model_type == "vit":
                print("\nApplying data augmentation to labeled ViT training set...")
                # Ensure image_processor is available (it should be from step 1 if model_type is vit)
                if image_processor is None:  # Should not happen if logic is correct
                    image_processor = AutoImageProcessor.from_pretrained(vit_model_name)

                vit_augment_chain_ssl = cls._get_vit_augment_transforms()
                augmented_transform_vit_ssl = lambda pil_img: image_processor(
                    images=vit_augment_chain_ssl(pil_img), return_tensors="pt"
                )["pixel_values"].squeeze(0)

                augmented_dataset_source_vit_ssl = cls(
                    root_dir=data_dir,
                    transform=augmented_transform_vit_ssl,
                    binary_classification=binary_classification,
                )
                labeled_dataset = torch.utils.data.Subset(
                    augmented_dataset_source_vit_ssl, labeled_indices
                )
            else:  # model_type not resnet or vit, or data_augmentation is False but somehow in this block
                labeled_dataset = torch.utils.data.Subset(full_dataset, labeled_indices)
        else:  # No data augmentation for the labeled set
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
