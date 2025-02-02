import os
import zipfile
import platform
import warnings
from glob import glob
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import libraries
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import gc

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score
from torchinfo import summary

# Set float precision and CUDA configurations
torch.set_float32_matmul_precision('high')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "true"  # Corrected here


import wandb

# Initialize Weights and Biases
wandb.login()

# Configuration classes
@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 4
    IMAGE_SIZE: tuple[int, int] = (288, 288)  # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)
    BACKGROUND_CLS_ID: int = 0
    URL: str = r"https://www.dropbox.com/scl/fi/r0685arupp33sy31qhros/dataset_UWM_GI_Tract_train_valid.zip?rlkey=w4ga9ysfiuz8vqbbywk0rdnjw&dl=1"
    DATASET_PATH: str = os.path.join(os.getcwd(), "dataset_UWM_GI_Tract_train_valid")

@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "images", r"*.png")
    DATA_TRAIN_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "masks", r"*.png")
    DATA_VALID_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "images", r"*.png")
    DATA_VALID_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "masks", r"*.png")

@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 10
    NUM_EPOCHS: int = 100
    INIT_LR: float = 3e-4
    NUM_WORKERS: int = 0 if platform.system() == "Windows" else os.cpu_count()
    OPTIMIZER_NAME: str = "AdamW"
    WEIGHT_DECAY: float = 1e-4
    USE_SCHEDULER: bool = True  
    SCHEDULER: str = "MultiStepLR"
    MODEL_NAME: str = "nvidia/segformer-b4-finetuned-ade-512-512"

@dataclass
class InferenceConfig:
    BATCH_SIZE: int = 5
    NUM_BATCHES: int = 2

# Color mapping for segmentation
id2color = {
    0: (0, 0, 0),  
    1: (0, 0, 255),  
    2: (0, 255, 0),  
    3: (255, 0, 0),  
}
DatasetConfig.NUM_CLASSES = len(id2color)
rev_id2color = {value: key for key, value in id2color.items()}

# Dataset class
class MedicalDataset(Dataset):
    def __init__(self, *, image_paths, mask_paths, img_size, ds_mean, ds_std, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train
        self.img_size = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.transforms = self.setup_transforms(mean=self.ds_mean, std=self.ds_std)

    def __len__(self):
        return len(self.image_paths)

    def setup_transforms(self, *, mean, std):
        transforms = []
        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.12, rotate_limit=0.15, shift_limit=0.12, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=self.img_size[1] // 20, 
                                max_width=self.img_size[0] // 20, min_holes=5, 
                                fill_value=0, mask_fill_value=0, p=0.5)
            ])
        transforms.extend([
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(always_apply=True),  # (H, W, C) --> (C, H, W)
        ])
        return A.Compose(transforms)

    def load_file(self, file_path, depth=0):
        file = cv2.imread(file_path, depth)
        if depth == cv2.IMREAD_COLOR:
            file = file[:, :, ::-1]
        return cv2.resize(file, (self.img_size), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, index):
        image = self.load_file(self.image_paths[index], depth=cv2.IMREAD_COLOR)
        mask = self.load_file(self.mask_paths[index], depth=cv2.IMREAD_GRAYSCALE)
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"].to(torch.long)
        return image, mask

# Data module class
class MedicalSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, num_classes=10, img_size=(384, 384), ds_mean=(0.485, 0.456, 0.406), 
                 ds_std=(0.229, 0.224, 0.225), batch_size=20, num_workers=0, pin_memory=False, 
                 shuffle_validation=False):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_validation = shuffle_validation

    def prepare_data(self):
        dataset_zip_path = f"{DatasetConfig.DATASET_PATH}.zip"
        if not os.path.exists(DatasetConfig.DATASET_PATH):
            print("Downloading and extracting assets...", end="")
            file = requests.get(DatasetConfig.URL)
            open(dataset_zip_path, "wb").write(file.content)

            try:
                with zipfile.ZipFile(dataset_zip_path) as z:
                    z.extractall(os.path.split(dataset_zip_path)[0]) 
                    print("Done")
            except:
                print("Invalid file")
            os.remove(dataset_zip_path) 

    def setup(self, *args, **kwargs):
        train_imgs = sorted(glob(f"{Paths.DATA_TRAIN_IMAGES}"))
        train_msks = sorted(glob(f"{Paths.DATA_TRAIN_LABELS}"))
        valid_imgs = sorted(glob(f"{Paths.DATA_VALID_IMAGES}"))
        valid_msks = sorted(glob(f"{Paths.DATA_VALID_LABELS}"))

        self.train_ds = MedicalDataset(image_paths=train_imgs, mask_paths=train_msks, 
                                       img_size=self.img_size, is_train=True, 
                                       ds_mean=self.ds_mean, ds_std=self.ds_std)
        self.valid_ds = MedicalDataset(image_paths=valid_imgs, mask_paths=valid_msks, 
                                       img_size=self.img_size, is_train=False, 
                                       ds_mean=self.ds_mean, ds_std=self.ds_std)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, pin_memory=self.pin_memory,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size, pin_memory=self.pin_memory,
            num_workers=self.num_workers, shuffle=self.shuffle_validation
        )

# Utility functions for visualization
def num_to_rgb(num_arr, color_map=id2color):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
    for k in color_map.keys():
        output[single_layer == k] = color_map[k]
    return np.float32(output) / 255.0

def image_overlay(image, segmented_image):
    alpha = 1.0  
    beta = 0.7  
    gamma = 0.0 
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image, alpha, segmented_image, beta, gamma)

# Model class
class MedicalSegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(TrainingConfig.MODEL_NAME, 
                                                                     num_labels=DatasetConfig.NUM_CLASSES)
        self.loss_fn = self.loss_fn_dice
        self.train_f1 = MulticlassF1Score(num_classes=DatasetConfig.NUM_CLASSES, average='macro')
        self.valid_f1 = MulticlassF1Score(num_classes=DatasetConfig.NUM_CLASSES, average='macro')

    def forward(self, x):
        return self.model(x)

    def loss_fn_dice(self, outputs, masks):
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.view(-1, outputs.size(1))
        masks = masks.view(-1)
        intersection = (outputs * masks).sum()
        return 1 - (2.0 * intersection + 1) / (outputs.sum() + masks.sum() + 1)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.train_f1(outputs, masks)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.valid_f1(outputs, masks)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('valid_f1', self.valid_f1.compute())
        self.valid_f1.reset()

    def configure_optimizers(self):
        optimizer = getattr(optim, TrainingConfig.OPTIMIZER_NAME)(self.parameters(), lr=TrainingConfig.INIT_LR, 
                                                                   weight_decay=TrainingConfig.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1) 
        return [optimizer], [scheduler]

# Inference function
def inference(model, data_loader, device, num_batches=5, visualize=True):
    model.eval()
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            predicted_masks = model(images).argmax(dim=1)

            for i in range(len(images)):
                if i >= num_batches:
                    break
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                true_mask = masks[i].cpu().numpy()
                predicted_mask = predicted_masks[i].cpu().numpy()

                if visualize:
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(image)
                    plt.title("Input Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(num_to_rgb(true_mask), alpha=0.5)
                    plt.title("True Mask")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(num_to_rgb(predicted_mask), alpha=0.5)
                    plt.title("Predicted Mask")
                    plt.axis("off")

                    plt.show()

# Training setup
if __name__ == "__main__":
    # Initialize GPU
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    # Prepare data and model
    data_module = MedicalSegmentationDataModule(
        num_classes=DatasetConfig.NUM_CLASSES,
        img_size=DatasetConfig.IMAGE_SIZE,
        ds_mean=DatasetConfig.MEAN,
        ds_std=DatasetConfig.STD,
        batch_size=TrainingConfig.BATCH_SIZE,
        num_workers=TrainingConfig.NUM_WORKERS
    )
    data_module.prepare_data()
    data_module.setup()

    model = MedicalSegmentationModel()

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=TrainingConfig.NUM_EPOCHS,
        logger=WandbLogger(project='medical_segmentation', name='Segformer'),
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(monitor='valid_loss', mode='min', save_top_k=1)
        ]
    )
    
    # Train the model
    trainer.fit(model, data_module)

    # Inference
    val_loader = data_module.val_dataloader()
    inference(model, val_loader, device='cuda', num_batches=InferenceConfig.NUM_BATCHES)
