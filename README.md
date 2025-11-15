# vision-transformer-pytorch
A PyTorch implementation of the Vision Transformer (ViT) architecture for image classification, trained on the MNIST dataset. This project demonstrates a complete, modular implementation of the Vision Transformer from scratch with comprehensive error handling, logging, and documentation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Vision Transformer (ViT) is a transformer-based architecture for image classification that treats image patches as sequences, similar to how transformers process text. This implementation includes:

- **Patch Embedding**: Converts images into patch embeddings
- **Positional Encoding**: Learnable positional embeddings for patch sequences
- **Transformer Encoder**: Multi-head self-attention and feed-forward layers
- **Classification Head**: MLP head for final predictions

## ✨ Features

- **Modular Design**: Clean separation of concerns with dedicated modules for models, data, training, and evaluation
- **Comprehensive Logging**: Detailed logging at each major processing step
- **Error Handling**: Robust error checking and validation throughout the codebase
- **Documentation**: Extensive comments and docstrings
- **GPU Support**: Automatic CUDA detection and usage
- **Visualization**: Built-in prediction visualization utilities
- **Model Checkpointing**: Save and load trained models

## 📁 Project Structure

```
VisionTransformer_ViT/
├── config.yml                # Configuration file (YAML format)
├── config.py                 # Configuration loader module
├── main.py                   # Main entry point
├── train.py                  # Training module
├── evaluate.py               # Evaluation module
├── visualize.py              # Visualization utilities
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── models/                   # Model components
│   ├── __init__.py
│   ├── patch_embedding.py   # Patch embedding layer
│   ├── transformer_encoder.py # Transformer encoder block
│   ├── mlp_head.py          # Classification head
│   └── vision_transformer.py # Complete ViT model
└── data/                     # Data utilities
    ├── __init__.py
    └── dataset.py           # Dataset loading and preprocessing
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Steps

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd VisionTransformer_ViT
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Basic Training

Run the complete training pipeline:

```bash
python main.py
```

This will:
1. Download the MNIST dataset (if not already present)
2. Initialize the Vision Transformer model
3. Train the model for the specified number of epochs
4. Evaluate on the validation set
5. Generate prediction visualizations

### Training Configuration

Modify `config.yml` to adjust hyperparameters:

```yaml
# Model architecture
model:
  patch_size: 7
  embed_dim: 64
  attention_heads: 4
  transformer_blocks: 4
  mlp_hidden_nodes: 128

# Training
training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 5
```

The configuration is loaded from `config.yml` at runtime. The `config.py` module handles loading and validation of the YAML configuration file.

### Using Individual Modules

You can also use individual modules programmatically:

```python
from models import VisionTransformer
from data import load_mnist_dataset, create_dataloaders
import train
import evaluate

# Load data
train_dataset = load_mnist_dataset(data_root="./data", train=True)
val_dataset = load_mnist_dataset(data_root="./data", train=False)
train_loader, val_loader = create_dataloaders(
    train_dataset, val_dataset, batch_size=64
)

# Create model
model = VisionTransformer(
    num_classes=10,
    num_channels=1,
    img_size=28,
    patch_size=7,
    embed_dim=64,
    attention_heads=4,
    transformer_blocks=4,
    mlp_hidden_nodes=128
)

# Train and evaluate
# ... (see main.py for complete example)
```

## 🏗️ Architecture

### Vision Transformer Components

1. **Patch Embedding** (`PatchEmbedding`):
   - Converts input images into non-overlapping patches
   - Projects patches into embedding space using a convolutional layer
   - Output shape: `(batch_size, num_patches, embed_dim)`

2. **CLS Token**:
   - Learnable classification token prepended to patch sequence
   - Used for final classification after transformer processing

3. **Positional Embedding**:
   - Learnable positional embeddings added to patch embeddings
   - Captures spatial relationships between patches

4. **Transformer Encoder** (`TransformerEncoder`):
   - Multi-head self-attention mechanism
   - Feed-forward MLP with GELU activation
   - Residual connections and layer normalization (pre-norm architecture)

5. **MLP Head** (`MLPHead`):
   - Layer normalization followed by linear layer
   - Produces class logits from CLS token

### Model Architecture Summary

```
Input Image (28x28x1)
    ↓
Patch Embedding (4x4 patches → 64-dim embeddings)
    ↓
Add CLS Token + Positional Embeddings
    ↓
Transformer Encoder Blocks (x4)
    ↓
Extract CLS Token
    ↓
MLP Head
    ↓
Class Predictions (10 classes)
```

## ⚙️ Configuration

Configuration is managed through `config.yml` (YAML format) for easy editing. The `config.py` module loads and validates the configuration at runtime.

Key configuration parameters in `config.yml`:

| Parameter | YAML Path | Default | Description |
|-----------|-----------|---------|-------------|
| `NUM_CLASSES` | `dataset.num_classes` | 10 | Number of output classes (MNIST digits) |
| `NUM_CHANNELS` | `dataset.num_channels` | 1 | Input channels (grayscale) |
| `IMG_SIZE` | `dataset.img_size` | 28 | Input image size |
| `PATCH_SIZE` | `model.patch_size` | 7 | Size of each patch |
| `EMBED_DIM` | `model.embed_dim` | 64 | Embedding dimension |
| `ATTENTION_HEADS` | `model.attention_heads` | 4 | Number of attention heads |
| `TRANSFORMER_BLOCKS` | `model.transformer_blocks` | 4 | Number of transformer blocks |
| `MLP_HIDDEN_NODES` | `model.mlp_hidden_nodes` | 128 | MLP hidden layer size |
| `BATCH_SIZE` | `training.batch_size` | 64 | Training batch size |
| `LEARNING_RATE` | `training.learning_rate` | 0.001 | Learning rate |
| `EPOCHS` | `training.epochs` | 5 | Number of training epochs |

## 📊 Results

The model achieves high accuracy on the MNIST dataset. Typical results:

- **Training Accuracy**: ~97-98% after 5 epochs
- **Validation Accuracy**: ~97-98%

Results may vary based on hyperparameters and random initialization.

## 📝 Logging

The implementation includes comprehensive logging:

- **Training progress**: Loss and accuracy at regular intervals
- **Model information**: Parameter counts, architecture details
- **Error messages**: Detailed error information for debugging
- **Evaluation metrics**: Validation loss and accuracy

Logs are written to both console and `training.log` file.

## 🔧 Error Handling

The codebase includes extensive error checking:

- Input validation for all functions
- Shape checking for tensors
- Device compatibility checks
- Dataset loading error handling
- Model initialization validation

## 📦 Output Files

After training, the following files are generated:

- `checkpoints/vit_model.pth`: Saved model checkpoint
- `results/predictions.png`: Visualization of predictions
- `training.log`: Training log file

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Original Vision Transformer paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- PyTorch community for excellent documentation and examples

## 📚 References

- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv preprint arXiv:2010.11929.

---

**Note**: This is an educational implementation. For production use, consider additional optimizations, data augmentation, and hyperparameter tuning.
