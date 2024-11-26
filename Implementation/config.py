# config.py

import numpy as np
import random
import tensorflow as tf

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Evaluation metric: 'PSNR' or 'SynFlow'
EVALUATION_METRIC = 'PSNR'  # You can change to 'SynFlow' as needed

# Local Search Configuration
LOCAL_SEARCH_CONFIG = {
    'MAX_EVALUATIONS': 25000,      # Total evaluations per local search
    'TABU_TENURE': 5,              # Tenure for Tabu Search
    'INITIAL_TEMP': 100,           # Initial temperature for Simulated Annealing
    'COOLING_RATE': 0.95,          # Cooling rate for Simulated Annealing
}

# Dataset Configuration
# Replace with your actual datasets
# Example placeholders (users should replace these with actual data)
# For PSNR evaluation, datasets are required for training and validation
train_images = np.random.rand(100, 64, 64, 3)  # Placeholder: Replace with actual training images
train_labels = np.random.rand(100, 64, 64, 3)  # Placeholder: Replace with actual training labels
val_images = np.random.rand(20, 64, 64, 3)     # Placeholder: Replace with actual validation images
val_labels = np.random.rand(20, 64, 64, 3)     # Placeholder: Replace with actual validation labels

DATASET_TRAIN = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
DATASET_VAL = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
EPOCHS = 5  # Number of epochs for training when using PSNR

# Device Configuration
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
