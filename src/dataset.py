import cv2
import torch
import logging
from imutils import paths
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_data_loaders(config, train_ds, val_ds, test_ds):
    """
    Create the training, validation, and testing data loaders.

    Args:
        config: Configuration object containing parameters.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        test_ds: Testing dataset.

    Returns:
        the training, validation and the test DataLoader objects.
    """

    train_loader = DataLoader(train_ds, 
                        shuffle=True, 
                        batch_size=config.TRAIN_BATCH_SIZE,
                        pin_memory=config.PIN_MEMORY,
                        num_workers=0)
    val_loader = DataLoader(val_ds,
                            shuffle=False,
                            batch_size=config.VAL_BATCH_SIZE,
                            pin_memory=config.PIN_MEMORY,
                            num_workers=0)
    test_loader = DataLoader(test_ds,
                            shuffle=False,
                            batch_size=config.TEST_BATCH_SIZE,
                            pin_memory=config.PIN_MEMORY,
                            num_workers=0)
    
    
    return train_loader, val_loader, test_loader


def get_images_and_masks(config):
    """
    Load the dataset, perform the train/val/test split, and create the corresponding Dataset objects.

    Args:
        config: Configuration object containing parameters.

    Returns:
        the training, validation and the test Dataset objects.
    """

    img_paths = sorted(paths.list_images(config.IMAGES_DATASET_PATH))
    mask_paths = sorted(paths.list_images(config.MASKS_DATASET_PATH))
    train_val_split = train_test_split(img_paths, 
                             mask_paths, 
                             train_size=config.TRAIN_PCT, 
                             random_state=config.RANDOM_SEED)
    train_imgs_paths, val_imgs_paths = train_val_split[:2]
    train_masks_paths, val_masks_paths = train_val_split[2:]
    test_imgs_count = int(len(img_paths) * config.TEST_PCT)
    val_test_split = train_test_split(val_imgs_paths, 
                                      val_masks_paths, 
                                      test_size=test_imgs_count,
                                      random_state=config.RANDOM_SEED)
    val_imgs_paths, test_img_paths = val_test_split[:2]
    val_masks_paths, test_masks_paths = val_test_split[2:]

    return train_imgs_paths, train_masks_paths, val_imgs_paths, val_masks_paths, test_img_paths, test_masks_paths


def get_data(config):
    """
    Load the dataset, perform the train/val/test split, and create the corresponding Dataset objects.

    Args:
        config: Configuration object containing parameters.

    Returns:
        the training, validation and the test Dataset objects.
    """

    train_imgs_paths, train_masks_paths, val_imgs_paths, val_masks_paths, test_img_paths, test_masks_paths = get_images_and_masks(config)

    transformations = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=config.INPUT_IMAGE_SIZE, scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ])
    test_transformations = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_ds = SegmentationDataset(image_paths=train_imgs_paths, 
                                   mask_paths=train_masks_paths,
                                   transforms=transformations)
    val_ds = SegmentationDataset(image_paths=val_imgs_paths, 
                                 mask_paths=val_masks_paths,
                                 transforms=transformations)
    test_ds = SegmentationDataset(image_paths=test_img_paths,
                                  mask_paths=test_masks_paths,
                                  transforms=test_transformations)

    logger.info(f"Found {len(train_ds)} examples in the training set ...")
    logger.info(f"Found {len(val_ds)} examples in the validation set ...")
    logger.info

    return train_ds, val_ds, test_ds
    

class SegmentationDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transforms=None, resize_to=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.resize_to = resize_to

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_AREA)
        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = cv2.resize(mask, self.resize_to, interpolation=cv2.INTER_AREA)

        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask
