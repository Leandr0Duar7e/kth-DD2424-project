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
   - Web interface for easier interaction (in progress)

## Web Application (Experimental)

A web application has been developed to provide an easy-to-use interface for the pet classification experiments. This is still a work in progress and might not function properly yet.

To run the web application:

```bash
cd src
uvicorn app:app --reload
```

The application will be available at: http://127.0.0.1:8000

**Note**: The web application requires additional dependencies like `gradio`, `fastapi`, and `uvicorn`. These are included in the `requirements.txt` file.

## Team

- Francesco Olivieri
- Inês Mesquita
- Leandro Duarte

KTH Royal Institute of Technology, DD2424 Deep Learning in Data Science
