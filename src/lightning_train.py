import os
import time
import config
import shutil
import logging
import matplotlib
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from mamba_unet_model import LitMambaUnet
from dataset import get_data, get_data_loaders
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
if not os.getenv("COLAB_RELEASE_TAG"):
    matplotlib.use('Qt5Agg')


def plot_history(train_loss, val_loss):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_loss, color="blue", label="Training loss")
    plt.plot(val_loss, color="red", label="Validation loss")
    plt.title("Training loss on dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)
    plt.show()


if __name__ == "__main__":
    if os.path.exists(config.BASE_OUTPUT):
        shutil.rmtree(config.BASE_OUTPUT)
    os.mkdir(config.BASE_OUTPUT)
    start_time = time.time()
    input_shape = (3, *config.INPUT_IMAGE_SIZE)
    model = LitMambaUnet(input_shape, 
                         config.DECODER_IN_CHANNELS, 
                         config.DECODER_SEQUENCE_LENGTH, 
                         config.RESIDUAL_CONNS_IN_CHANNELS)
    early_stopping = EarlyStopping(monitor="epoch_avg_val_loss", patience=5, mode="min")
    checkpoint = ModelCheckpoint(config.BASE_OUTPUT, 
                                 filename=config.MODEL_FILE_NAME_FORMAT,
                                 save_top_k=1,
                                 mode="min")
    train_ds, val_ds, test_ds = get_data(config)
    train_dl, val_dl, test_dl = get_data_loaders(config, train_ds, val_ds, test_ds)
    trainer = Trainer(callbacks=[early_stopping, checkpoint], max_epochs=config.NUM_EPOCHS)
    trainer.fit(model, train_dl, val_dl)
    print(f"Training elapsed time: {time.time()-start_time}s")
    plot_history(model.epoch_train_loss, model.epoch_val_loss)
    trainer.test(model, dataloaders=test_dl, ckpt_path="best")
    