import torch
import time
import gradio as gr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import PIL.Image as Image
from io import BytesIO
import os
import argparse
import sys
from fastapi import FastAPI

# Local imports
from models.resnet import ResNet50
from trainer import ModelTrainer
from dataset import OxfordPetDataset
from utils import get_device, get_swedish_waiting_message, create_directories, set_seed

# Ensure reproducibility
set_seed(42)

# Create required directories
create_directories("../models/resnet", ["binary", "multiclass", "pretrained"])

# Global variables to store model, trainer and data
loaded_models = {"binary": None, "multiclass": None}
trainers = {"binary": None, "multiclass": None}
device = get_device()
dataloaders = {"binary": None, "multiclass": None}


def load_data(binary_classification=True):
    """Load dataset with appropriate classification type"""
    print(f"\nLoading Oxford-IIIT Pet Dataset (binary={binary_classification})...")

    # Check if already loaded
    key = "binary" if binary_classification else "multiclass"
    if dataloaders[key] is not None:
        return dataloaders[key]

    # Load data
    train_loader, val_loader, test_loader, num_classes = (
        OxfordPetDataset.get_dataloaders(
            data_dir="../data/raw",
            batch_size=32,
            binary_classification=binary_classification,
        )
    )

    dataloaders[key] = (train_loader, val_loader, test_loader, num_classes)
    return dataloaders[key]


def load_model(binary_classification=True, num_train_layers=0):
    """Load model based on classification type"""
    key = "binary" if binary_classification else "multiclass"

    # Return existing model if it matches the requirements
    if loaded_models[key] is not None:
        return loaded_models[key]

    freeze = True if binary_classification else False
    model = ResNet50(
        binary_classification=binary_classification,
        freeze_backbone=freeze,
        num_train_layers=num_train_layers,
    )

    loaded_models[key] = model
    return model


def initialize_trainer(model, binary_classification=True):
    """Initialize a trainer for the model"""
    key = "binary" if binary_classification else "multiclass"

    if trainers[key] is not None:
        return trainers[key]

    trainer = ModelTrainer(model, device, binary_classification=binary_classification)
    trainers[key] = trainer
    return trainer


def get_sample_images(binary_classification=True):
    """Get some sample images from the dataset for display"""
    key = "binary" if binary_classification else "multiclass"

    if dataloaders[key] is None:
        load_data(binary_classification)

    train_loader = dataloaders[key][0]

    # Get a batch of images
    images, labels = next(iter(train_loader))

    # Convert to numpy for visualization (only take first 5)
    sample_images = []
    for i in range(min(5, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        label_text = (
            "Dog"
            if labels[i] == 1
            else "Cat" if binary_classification else f"Breed {int(labels[i])}"
        )
        sample_images.append((img, label_text))

    return sample_images


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()

    # Save figure to BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    # Open image from buffer
    img = Image.open(buf)
    return np.array(img)


def run_experiment(task_type, num_layers, num_epochs):
    """Run classification experiment"""
    binary_classification = task_type == "binary"
    progress_text = ""

    # Update progress
    progress_text += f"Starting {'binary' if binary_classification else 'multi-class'} classification experiment...\n"
    progress_text += "Loading data...\n"
    yield progress_text, None

    # Load data
    train_loader, val_loader, test_loader, _ = load_data(
        binary_classification=binary_classification
    )

    # Update progress
    progress_text += f"Dataset loaded successfully with {len(train_loader.dataset)} training samples\n"
    progress_text += f"Using model with {num_layers} trainable layers\n"
    yield progress_text, None

    # Load model
    model = load_model(
        binary_classification=binary_classification, num_train_layers=num_layers
    )

    # Initialize trainer
    trainer = initialize_trainer(model, binary_classification=binary_classification)

    # Update progress with waiting message
    progress_text += f"{get_swedish_waiting_message()}\n"
    progress_text += f"Training for {num_epochs} epochs (this may take a while)...\n"
    yield progress_text, None

    # Train model
    model, history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # Evaluate on test set
    progress_text += "Evaluating model on test set...\n"
    yield progress_text, None

    test_loss, test_acc = trainer.evaluate(test_loader)

    # Update final results
    progress_text += f"Final Test Results:\n"
    progress_text += f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n"
    progress_text += "Experiment completed!"

    # Plot history
    history_plot = plot_training_history(history)

    yield progress_text, history_plot


def create_ui():
    """Create a simplified Gradio interface"""
    with gr.Blocks(
        title="KTH Pet Classification", theme=gr.themes.Soft(primary_hue="blue")
    ) as app:
        gr.Markdown(
            """
        # Pet Classification with ResNet50
        ### KTH Royal Institute of Technology - DD2424 Deep Learning
        
        This app runs experiments for classifying pets from the Oxford-IIIT Pet Dataset.
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_type = gr.Radio(
                    ["binary", "multiclass"],
                    label="Classification Task",
                    value="binary",
                    info="Binary: Dog vs Cat | Multiclass: 37 Breeds",
                )

                num_layers = gr.Slider(
                    0,
                    5,
                    value=0,
                    step=1,
                    label="Number of Layers to Train",
                    info="0 means only the last layer will be trained",
                )

                num_epochs = gr.Slider(
                    1,
                    5,
                    value=1,
                    step=1,
                    label="Number of Epochs",
                    info="More epochs = better results but longer training time",
                )

                run_button = gr.Button("Run Experiment", variant="primary")

            with gr.Column(scale=2):
                # Output area
                with gr.Row():
                    output_text = gr.Textbox(
                        label="Progress", lines=8, show_copy_button=True
                    )
                with gr.Row():
                    output_plot = gr.Image(
                        label="Training Results", show_download_button=True, height=350
                    )

        # Sample images section
        with gr.Accordion("Show Sample Images", open=False):
            view_samples = gr.Button("View Sample Images")
            gallery = gr.Gallery(
                label="Sample Images from Dataset", columns=5, height=200
            )

        # Set up event handlers
        run_button.click(
            fn=run_experiment,
            inputs=[task_type, num_layers, num_epochs],
            outputs=[output_text, output_plot],
        )

        # Get samples function
        def load_samples(task):
            binary = task == "binary"
            samples = get_sample_images(binary_classification=binary)
            return [img for img, _ in samples]

        view_samples.click(fn=load_samples, inputs=[task_type], outputs=[gallery])

        gr.Markdown(
            """
        ### Instructions
        1. Select the type of classification task (binary or multi-class)
        2. Choose how many layers to train (0 means only the last layer)
        3. Set the number of training epochs
        4. Click "Run Experiment" to start training
        5. View the results in the progress area and the plots
        """
        )

    return app


# Create the Gradio app
gradio_app = create_ui()

# Create a FastAPI app and mount the Gradio app
app = FastAPI(title="KTH Pet Classification API")
app.mount("/", gr.routes.App.create_app(gradio_app))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the KTH Pet Classification web app"
    )
    parser.add_argument(
        "--server",
        choices=["gradio", "uvicorn"],
        default="gradio",
        help="Choose server type: gradio (default) or uvicorn",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    args = parser.parse_args()

    # Ensure requirements are updated
    required_packages = ["gradio>=5.29.1", "uvicorn>=0.27.0", "fastapi>=0.110.0"]
    if os.path.exists("../requirements.txt"):
        with open("../requirements.txt", "r") as f:
            requirements = f.read()

        with open("../requirements.txt", "a") as f:
            for package in required_packages:
                package_name = package.split(">=")[0]
                if package_name not in requirements:
                    f.write(f"{package}\n")

    if args.server == "uvicorn":
        print("Starting web application with Uvicorn server...")
        print(f"To access the app, navigate to: http://127.0.0.1:{args.port}")
        print("To stop the server, press CTRL+C")

        # We need to install uvicorn if it's not already installed
        try:
            import uvicorn
        except ImportError:
            print("Uvicorn not found. Installing...")
            os.system("pip install uvicorn")
            import uvicorn

        # Run with uvicorn
        uvicorn.run("app:app", host="127.0.0.1", port=args.port, reload=True)
    else:
        # Default Gradio server
        print("Starting web application with Gradio server...")
        gradio_app.launch(server_name="127.0.0.1", server_port=args.port, share=False)
