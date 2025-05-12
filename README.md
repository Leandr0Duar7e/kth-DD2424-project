# Comparing ResNet and Vision Transformers: Supervised vs Semi-Supervised Learning for Pet Breed Classification

This project explores and compares the performance of ResNet and Vision Transformer (ViT) architectures for pet breed classification using the Oxford-IIIT Pet Dataset. It examines both fully supervised and semi-supervised learning approaches.

## Project Structure

```
kth-DD2424-project/
├── data/                           # Data directory
│   ├── raw/                        # Original immutable data
│   │   └── oxford_pets/            # Original Oxford-IIIT Pet Dataset
│   │       ├── images/             # Original images
│   │       └── annotations/        # Dataset annotations and labels
│   ├── interim/                    # Intermediate processed data
│   │   ├── binary/                 # Data processed for binary classification (cats vs dogs)
│   │   └── multiclass/             # Data processed for breed classification (37 classes)
│   └── processed/                  # Final processed datasets ready for model training
│       ├── full/                   # 100% of labeled data
│       ├── imbalanced/             # Imbalanced dataset with reduced examples for some classes
│       ├── semi_1/                 # 1% labeled data setup for semi-supervised learning
│       ├── semi_10/                # 10% labeled data setup for semi-supervised learning
│       └── semi_50/                # 50% labeled data setup for semi-supervised learning
├── models/                         # Saved model checkpoints
│   ├── resnet/                     # ResNet models
│   │   ├── binary/                 # Fine-tuned models for binary classification
│   │   ├── multiclass/             # Fine-tuned models for multiclass classification
│   │   └── pretrained/             # Pretrained ResNet models
│   ├── vit/                        # Vision Transformer models
│   │   ├── binary/                 # Fine-tuned ViT models for binary classification
│   │   ├── multiclass/             # Fine-tuned ViT models for multiclass classification
│   │   └── pretrained/             # Pretrained ViT models
│   └── experiments/                # Models from specific experiments
│       ├── exp1_baseline/          # Baseline experiment results
│       ├── exp2_finetune_layers/   # Results from fine-tuning different layers
│       └── exp3_gradual_unfreeze/  # Results from gradual unfreezing approach
├── src/                            # Source code
│   ├── data/                       # Data processing code
│   ├── models/                     # Model definitions
│   ├── training/                   # Training code
│   ├── evaluation/                 # Evaluation code
│   └── utils/                      # Utility functions
├── scripts/                        # Scripts for different stages
├── results/                        # Experiment results, metrics, etc.
├── figures/                        # Generated figures for analysis
├── report/                         # Report and presentation materials
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the Oxford-IIIT Pet Dataset and place it in `data/raw/oxford_pets/`

## Running the Project

The project is divided into several stages:

1. **Data Preparation**:
   ```
   python scripts/1_prepare_data.py
   ```

2. **Binary Classification** (Dogs vs Cats):
   ```
   python scripts/2_binary_classification.py
   ```

3. **Multi-Class Classification** (37 Breeds):
   ```
   python scripts/3_multiclass_classification.py
   ```

4. **Semi-Supervised Learning**:
   ```
   python scripts/4_semi_supervised_learning.py
   ```

5. **Generate Figures** for the report:
   ```
   python scripts/5_generate_figures.py
   ```

## Team

- Francesco Olivieri
- Inês Mesquita
- Leandro Duarte

KTH Royal Institute of Technology, DD2424 Deep Learning in Data Science
