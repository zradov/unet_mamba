import os
import torch

# base path of the dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
# define the path to the images and masks dataset
IMAGES_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASKS_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
# define the train dataset split threshold in percentage of all the data.
TRAIN_PCT = 0.75
TEST_PCT = 0.05
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
# set the random seed for reproducibility
RANDOM_SEED = 42

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# define the input image dimensions
INPUT_IMAGE_SIZE = (256, 256)
# the number of channels in the residual connections from the encoder.
RESIDUAL_CONNS_IN_CHANNELS = [4096, 1024, 256]
# the number of input channels for each decoder block.
DECODER_IN_CHANNELS = [4096, 1024, 256]
# the sequence length for the state space model in the decoder
DECODER_SEQUENCE_LENGTH = 8
    
# Training batch size
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 32))
# Validation batch size
VAL_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 16))
# Testing batch size
TEST_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 8))
# define threshold to filter weak predictions
PRED_THRESHOLD = 0.5
# Model checkpoints path
CHECKPOINTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
# define the path to the base output directory
BASE_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_FILE_NAME_FORMAT = "best_model-{epoch:02d}-{val_loss:.2f}"
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")

# Hyperparameters
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 10))

# Logging configuration
DEBUG_MESSAGE_FORMAT = "%(levelname)s:%(name)s:%(funcName)s:%(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
