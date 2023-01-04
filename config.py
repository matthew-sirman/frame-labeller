"""
Configuration.

Contains global variables defining the configuration of the predictor.
"""

# Dataset sources
EDS_SOURCES_DIR = "../DeepBank"
SEMLINK_DATA_FILE = "../1.2.2c.okay.txt"

# Dataset parameters
TRAIN_SPLIT = 0.7
DEV_SPLIT = 0.1
TEST_SPLIT = 0.2
SHUFFLE_BEFORE_SPLITTING = True

# Model parameters
MODEL_EMBEDDING_SIZE = 100
MODEL_ATTENTION_HEADS = 2
MODEL_HIDDEN_NODE_FEATURE_SIZE = 200
MODEL_OUTPUT_EMBEDDING_SIZE = 100
MODEL_SUBGRAPH_HOPS = 3

# Training parameters
LEARNING_RATE = 0.0005
TRAIN_EPOCHS = 10
TRAIN_LOG_PER_N_STEPS = 2000
VALIDATE_AFTER_EPOCHS = 5

SAVE_AFTER_STEPS = 2000

SAVE_MODEL = "../trained-models/frame-labeller-v1"
LOAD_MODEL = None
MODEL_NAME = "model"

# Constants
IGNORE_ATTRS = {"L-INDEX", "R-INDEX", "L-HNDL", "R-HNDL", "ARG"}
