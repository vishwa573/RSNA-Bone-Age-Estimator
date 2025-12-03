import torch


class Config:
    """Configuration for Bone Age Prediction project."""

    # Data and paths
    DATA_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"
    OUTPUT_DIR: str = "output"
    MODEL_DIR: str = "output/models"
    PLOTS_DIR: str = "output/plots"
    CSV_FILENAME: str = "boneage-training-dataset.csv"

    # Training hyperparameters
    IMG_SIZE: int = 256
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_WORKERS: int = 4
    SEED: int = 42

    # Device configuration
    if torch.cuda.is_available():
        DEVICE: str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE: str = "mps"
    else:
        DEVICE: str = "cpu"
