import os
import torch
import config
import matplotlib
import matplotlib.pyplot as plt
from mamba_unet_model import LitMambaUnet
from dataset import get_data, get_data_loaders

#logger = logging.getLogger(__name__)
if not os.getenv("COLAB_RELEASE_TAG"):
    matplotlib.use('Qt5Agg')


def hide_ticks(ax, index):
    ax[index][0].set_xticks([])
    ax[index][0].set_yticks([])
    ax[index][1].set_xticks([])
    ax[index][1].set_yticks([])
    ax[index][2].set_xticks([])
    ax[index][2].set_yticks([])
        

def plot_predictions(images, y_pred, y_true):
    fig, ax = plt.subplots(y_pred.shape[0], 3, figsize=(2, 5))

    for index, (img, pred_mask, true_mask) in enumerate(zip(images, y_pred, y_true)):
        hide_ticks(ax, index)
        
        ax[index][0].imshow(img.permute(1, 2, 0))
        ax[index][1].imshow(pred_mask.permute(1, 2, 0))
        ax[index][2].imshow(true_mask.permute(1, 2, 0))

    plt.show()


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_data(config)
    batch_size = 4
    _, _, test_loader = get_data_loaders(config, train_ds, val_ds, test_ds)
    input_shape = (3, *config.INPUT_IMAGE_SIZE)
    model = LitMambaUnet.load_from_checkpoint(
        os.path.join(config.CHECKPOINTS_PATH, "best.ckpt"),
        input_shape=input_shape, 
        dec_in_channels=config.DECODER_IN_CHANNELS, 
        dec_seq_len=config.DECODER_SEQUENCE_LENGTH,
        res_conns_channels=config.RESIDUAL_CONNS_IN_CHANNELS)
    model.eval()
    with torch.no_grad():
        test_images, test_masks = next(iter(test_loader))
        # the model predictions contain raw logits values that need
        # to be passed through a sigmoid function to turn them into
        # valid probabilities between 0 and 1.
        probs = torch.sigmoid(model(test_images).squeeze(0))
        pred_masks = (probs > config.PRED_THRESHOLD).float() 
        print(pred_masks.shape)
        plot_predictions(test_images, pred_masks, test_masks)

    