import os
import torch
import random
from tqdm import tqdm
import time
from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

# Constants
SWEDISH_MESSAGES = [
    "Time for a 'fika' break while you wait - a Swedish coffee tradition...",
    "Waiting time - perfect for enjoying a cinnamon bun, a Swedish specialty!",
    "Processing... taking longer than a Stockholm subway commute...",
    "Model training in progress - experience some Swedish 'lugn' (calmness)...",
    "Loading... at the relaxed pace of a Swedish summer morning...",
    "This wait is longer than KTH's winter break...",
    "Training - like walking through Stockholm's old town, take your time...",
    "Processing... slower than getting through IKEA on a Saturday...",
    "While waiting, imagine you're on a bench at KTH campus overlooking Stockholm...",
    "Computing... time to enjoy some theoretical Swedish efficiency...",
]


def set_seed(seed=42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Get the available device (CUDA GPU or CPU)

    Returns:
        device: PyTorch device
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        return device
    except:
        print("\nNo GPU available, using CPU")
        return torch.device("cpu")


def get_swedish_waiting_message():
    """
    Get a random Swedish-themed waiting message

    Returns:
        message: A random message about waiting with a Swedish theme
    """
    return random.choice(SWEDISH_MESSAGES)


def create_directories(base_path, directories):
    """
    Create all directories in a given base path

    Args:
        base_path: Base directory path
        directories: List of directory names to create
    """
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Directory created: {path}")


def display_progress_with_humor(iterable, desc="Processing", total=None):
    """
    Display a progress bar with occasional Swedish humor messages

    Args:
        iterable: Iterable to process
        desc: Description for the progress bar
        total: Total number of items

    Yields:
        Items from the iterable
    """
    pbar = tqdm(iterable, desc=desc, total=total)
    last_message_time = time.time()
    message_interval = 30  # Show message every 30 seconds

    for i, item in enumerate(pbar):
        # Occasionally show a Swedish-themed message
        current_time = time.time()
        if current_time - last_message_time > message_interval:
            pbar.set_description(get_swedish_waiting_message())
            last_message_time = current_time
            time.sleep(1)  # Show the message for a second
            pbar.set_description(desc)  # Restore the original description

        yield item


def compute_class_weights(dataset, num_classes):
    """
    Computes inverse-frequency class weights for use in CrossEntropyLoss.
    """
    label_counts = Counter()
    for _, label in dataset:
        label_counts[int(label)] += 1

    total = sum(label_counts.values())
    weights = [
        total / (num_classes * label_counts[i]) if i in label_counts else 0.0
        for i in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float)


def get_oversampled_loader(dataset, batch_size):
    """
    Returns a DataLoader using WeightedRandomSampler to balance class frequencies.
    """
    label_counts = Counter()
    for _, label in dataset:
        label_counts[int(label)] += 1

    weights = [1.0 / label_counts[int(label)] for _, label in dataset]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def is_vit_model(model):
    """
    Checks if a model is a Vision Transformer (ViT) model.

    Args:
        model: The model to check

    Returns:
        bool: True if the model is a Vision Transformer, False otherwise
    """
    # Check for ViT class name
    is_vit = model.__class__.__name__ == "ViT"

    # Check for ViT method to be sure
    has_vit_method = hasattr(model, "get_trainable_blocks_info")

    return is_vit and has_vit_method
