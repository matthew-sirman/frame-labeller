"""
Configuration.

Contains global variables defining the configuration of the predictor.
"""

# General
USE_CUDA_IF_AVAILABLE = True
RANDOM_SEED = 100

# Dataset sources
EDS_SOURCES_DIR = "../DeepBank"
PTB_SOURCES_DIR = "../PTB"
PTB_PARSE_LOAD_FILE = "sentence_to_ptb.json"
PTB_PARSE_SAVE_FILE = "sentence_to_ptb.json"
SEMLINK_DATA_FILE = "../1.2.2c.okay.txt"

# Dataset parameters
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.1
TEST_SPLIT = 0.1
SHUFFLE_BEFORE_SPLITTING = True

# Model parameters
MODEL_EMBEDDING_SIZE = 50
MODEL_ATTENTION_HEADS = 2
MODEL_HIDDEN_NODE_FEATURE_SIZE = 50

# Training parameters
LEARNING_RATE = 0.0005
TRAIN_EPOCHS = 0
TRAIN_LOG_PER_N_STEPS = 2000
VALIDATE_AFTER_EPOCHS = 5

SAVE_AFTER_STEPS = 5000

SAVE_MODEL = None # "../trained-models/frame-labeller-v4"
LOAD_MODEL = "../trained-models/frame-labeller-v4"
MODEL_NAME = "model"
