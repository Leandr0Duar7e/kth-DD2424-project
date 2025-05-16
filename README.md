# Comparing ResNet and Vision Transformers: Supervised vs Semi-Supervised Learning for Pet Breed Classification

This project explores and compares the performance of ResNet and Vision Transformer (ViT) architectures for pet breed classification using the Oxford-IIIT Pet Dataset. It examines both fully supervised and semi-supervised learning approaches.

## Project Structure

The current implementation includes:

```
kth-DD2424-project/
├── data/                           # Data directory
│   └── raw/                        # Original Oxford-IIIT Pet Dataset
│       ├── images/                 # Original images
│       └── annotations/            # Dataset annotations and labels
├── models/                         # Saved model checkpoints
│   └── resnet/                     # ResNet models
│       ├── binary/                 # Fine-tuned models for binary classification
│       ├── multiclass/             # Fine-tuned models for multiclass classification
│       └── pretrained/             # Pretrained ResNet models
├── src/                            # Source code
│   ├── models/                     # Model definitions
│   │   └── resnet.py               # ResNet50 model implementation
│   ├── dataset.py                  # Dataset loading and preprocessing
│   ├── trainer.py                  # Model training and evaluation
│   ├── utils.py                    # Utility functions
│   ├── main.py                     # Command-line interface
│   ├── app.py                      # Web interface (in progress)
│   └── evaluation.py               # Evaluation code (placeholder)
└── requirements.txt                # Project dependencies
```

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the Oxford-IIIT Pet Dataset and place it in `data/raw/`
   - The dataset should contain an `images` folder with all pet images
   - It should also have an `annotations` folder with the label information

## Running the Project

The project can be run in two ways: using the command-line interface or the web interface.

### Command-Line Interface

The `main.py` script provides a command-line interface for running experiments:

```bash
cd src
python main.py
   ```

This will present you with options:
1. **Binary Classification (Dogs vs Cats)**: Fine-tunes a ResNet50 model for binary classification with Adam optimizer.
2. **Multi-Class Classification (37 Breeds)**: Fine-tunes a ResNet50 model for multi-class breed classification.

For each experiment, you can specify:
- The number of layers to train in addition to the final layer
- The number of epochs to train

The model will be trained on the Oxford-IIIT Pet Dataset, and the results will be displayed at the end.

### Example Usage:

```bash
# Run the binary classification experiment
python main.py
# Then choose option 1 and follow the prompts
```

## Implemented Features

The current implementation includes:

1. **ResNet50 Model**:
   - Pre-trained ResNet50 backbone
   - Option to freeze/unfreeze specific layers
   - Support for both binary and multi-class classification

2. **Data Processing**:
   - Oxford-IIIT Pet Dataset loading and preprocessing
   - Image transformations for ResNet50
   - Train/validation/test splitting

3. **Training & Evaluation**:
   - Model training with Adam optimizer
   - Progress tracking and metrics (loss, accuracy)
   - Model evaluation on test set
   - Option to save trained models

4. **User Interface**:
   - Command-line interface for running experiments

## Running on Google Cloud VM

This section outlines how to set up and run the project on a shared Google Cloud VM.

**Phase 1: Your Local Machine Setup (One-Time)**

1.  **Install Google Cloud SDK (`gcloud` CLI)**:
    *   If you don't have it installed, download and install it from the official Google Cloud documentation: [Install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
    *   Follow the instructions for your operating system.

2.  **Initialize and Configure `gcloud`**:
    *   Open a new terminal on your local machine.
    *   Run the command:
        ```bash
        gcloud init
        ```
    *   **Login**: Your web browser will open, prompting you to log in. Use the Google account that has been granted access to the GCP project.
    *   **Choose Project**: When prompted to pick a cloud project, select the appropriate project ID (e.g., `proud-voice-459917-p7`).
    *   **Default Region and Zone**:
        *   The tool will ask: `Do you want to configure a default Compute Region and Zone? (Y/n)?` Type `Y` and press Enter.
        *   When prompted for the default region, enter the VM's region (e.g., `asia-east1`).
        *   When prompted for the default zone, enter the VM's zone (e.g., `asia-east1-c`).

**Phase 2: Getting Your Code to the VM (Do this when you update code)**

1.  **Archive Your Project Locally**:
    *   Open your local terminal and navigate to the directory that *contains* your project folder (e.g., if your project is `~/Projects/kth-DD2424-project`, then `cd ~/Projects/`).
    *   Run this command to create a compressed archive. This excludes the `.git` history and `.DS_Store` files, making the archive smaller and the transfer faster:
        ```bash
        tar --exclude='.git' --exclude='.DS_Store' -czvf kth-DD2424-project.tar.gz kth-DD2424-project
        ```
        (This creates an archive named `kth-DD2424-project.tar.gz` in the current directory).

2.  **Copy the Archive to the VM**:
    *   From the same local directory where you created the archive, run:
        ```bash
        gcloud compute scp kth-DD2424-project.tar.gz YOUR_VM_NAME:~/
        ```
        (Replace `YOUR_VM_NAME` with the actual name of your VM, e.g., `deeplearning-1-vm`).

**Phase 3: Setup & Run on the VM**

1.  **Connect to the VM via SSH**:
    *   In your local terminal, run:
        ```bash
        gcloud compute ssh YOUR_VM_NAME
        ```
    *   If it's your first time connecting from your computer, `gcloud` might generate SSH keys. Follow the prompts (you can typically accept defaults and opt for no passphrase for simplicity).

2.  **Prepare Project on VM (First time, or if you re-copied the .tar.gz)**:
    *   You'll be in your home directory (`~`) on the VM.
    *   If you have an old version of the project directory on the VM, remove it first:
        ```bash
        rm -rf ~/kth-DD2424-project
        ```
    *   Extract your newly copied archive:
        ```bash
        tar -xzvf ~/kth-DD2424-project.tar.gz
        ```
    *   (Optional) Delete the archive file from the VM to save space:
        ```bash
        rm ~/kth-DD2424-project.tar.gz
        ```

3.  **Navigate to Project & Setup Python Environment**:
    *   Change into your project directory on the VM:
        ```bash
        cd ~/kth-DD2424-project
        ```
    *   Create a Python virtual environment:
        ```bash
        python3 -m venv .venv
        ```
    *   Activate the virtual environment:
        ```bash
        source .venv/bin/activate
        ```
        (Your terminal prompt should now start with `(.venv)`).
    *   Install project dependencies:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run Your Code**:
    *   Ensure the virtual environment is active (see `(.venv)` in your prompt).
    *   Navigate to your source code directory (e.g., `cd src` if your main script is there).
    *   Execute your script (example):
        ```bash
        python main.py --model_type resnet --epochs 10
        ```
        (Adjust the script name and arguments based on your `main.py`).

**Important Notes for VM Usage:**
*   **Updating Code on VM**: To update the code, repeat Phase 2 (archive locally, copy archive) and then on the VM, delete the old project folder (if it exists) and re-extract the new archive (Phase 3, Step 2). You typically don't need to recreate the `.venv` or reinstall *all* requirements unless your `requirements.txt` file has changed.
*   **Exiting the VM**: Type `exit` in the VM's terminal to return to your local machine.
*   **Stopping the VM**: Coordinate with your team to **stop the VM** from the Google Cloud Console (Compute Engine > VM Instances page) when no one is actively using it. This helps manage costs, especially for VMs with GPUs.

## Team

- Francesco Olivieri
- Inês Mesquita
- Leandro Duarte

KTH Royal Institute of Technology, DD2424 Deep Learning in Data Science
