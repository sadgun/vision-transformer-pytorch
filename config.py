"""
Configuration file for Vision Transformer (ViT) implementation.

This module loads configuration from a YAML file and provides access to
all hyperparameters, model configuration, and training settings.
"""

import yaml
import logging
from pathlib import Path

# Set up logger (handle case where logging might not be configured yet)
try:
    logger = logging.getLogger(__name__)
except Exception:
    # Fallback if logging not configured
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Please create a config.yml file in the project root."
        )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise


# Load configuration
_config = load_config()

# Extract configuration values with error checking
try:
    # Dataset configuration
    NUM_CLASSES = _config['dataset']['num_classes']
    NUM_CHANNELS = _config['dataset']['num_channels']
    IMG_SIZE = _config['dataset']['img_size']
    DATA_ROOT = _config['dataset']['data_root']
    
    # Model architecture hyperparameters
    PATCH_SIZE = _config['model']['patch_size']
    EMBED_DIM = _config['model']['embed_dim']
    ATTENTION_HEADS = _config['model']['attention_heads']
    TRANSFORMER_BLOCKS = _config['model']['transformer_blocks']
    MLP_HIDDEN_NODES = _config['model']['mlp_hidden_nodes']
    
    # Training hyperparameters
    BATCH_SIZE = _config['training']['batch_size']
    LEARNING_RATE = _config['training']['learning_rate']
    EPOCHS = _config['training']['epochs']
    LOG_INTERVAL = _config['training']['log_interval']
    SHUFFLE_TRAIN = _config['training']['shuffle_train']
    SHUFFLE_VAL = _config['training']['shuffle_val']
    
    # Device configuration
    USE_CUDA = _config['device']['use_cuda']
    
    # Calculate derived parameters
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # Total number of patches per image
    
    # Validate configuration values
    if NUM_CLASSES <= 0:
        raise ValueError(f"num_classes must be positive, got {NUM_CLASSES}")
    if NUM_CHANNELS <= 0:
        raise ValueError(f"num_channels must be positive, got {NUM_CHANNELS}")
    if IMG_SIZE <= 0:
        raise ValueError(f"img_size must be positive, got {IMG_SIZE}")
    if IMG_SIZE % PATCH_SIZE != 0:
        raise ValueError(
            f"img_size ({IMG_SIZE}) must be divisible by patch_size ({PATCH_SIZE})"
        )
    if PATCH_SIZE <= 0:
        raise ValueError(f"patch_size must be positive, got {PATCH_SIZE}")
    if EMBED_DIM <= 0:
        raise ValueError(f"embed_dim must be positive, got {EMBED_DIM}")
    if ATTENTION_HEADS <= 0:
        raise ValueError(f"attention_heads must be positive, got {ATTENTION_HEADS}")
    if EMBED_DIM % ATTENTION_HEADS != 0:
        raise ValueError(
            f"embed_dim ({EMBED_DIM}) must be divisible by attention_heads ({ATTENTION_HEADS})"
        )
    if TRANSFORMER_BLOCKS <= 0:
        raise ValueError(f"transformer_blocks must be positive, got {TRANSFORMER_BLOCKS}")
    if MLP_HIDDEN_NODES <= 0:
        raise ValueError(f"mlp_hidden_nodes must be positive, got {MLP_HIDDEN_NODES}")
    if BATCH_SIZE <= 0:
        raise ValueError(f"batch_size must be positive, got {BATCH_SIZE}")
    if LEARNING_RATE <= 0:
        raise ValueError(f"learning_rate must be positive, got {LEARNING_RATE}")
    if EPOCHS <= 0:
        raise ValueError(f"epochs must be positive, got {EPOCHS}")
    if LOG_INTERVAL <= 0:
        raise ValueError(f"log_interval must be positive, got {LOG_INTERVAL}")
    
    logger.info(
        f"Configuration validated: "
        f"num_classes={NUM_CLASSES}, img_size={IMG_SIZE}, "
        f"patch_size={PATCH_SIZE}, embed_dim={EMBED_DIM}, "
        f"epochs={EPOCHS}, batch_size={BATCH_SIZE}"
    )

except KeyError as e:
    logger.error(f"Missing required configuration key: {e}")
    raise ValueError(f"Missing required configuration key: {e}") from e
except Exception as e:
    logger.error(f"Error processing configuration: {e}")
    raise
