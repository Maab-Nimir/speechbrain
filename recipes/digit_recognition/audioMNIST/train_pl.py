import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy, MetricCollection

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

import wandb

from prepare_audioMNIST import AudioMNISTDataset
from custom_model import Resnet

seed = 42
pl.seed_everything(seed)


# define the model
class ResNetModule(pl.LightningModule):
    def __init__(self, backbone="resnet18", input_channels=4, num_classes=10):
        super().__init__()
        self.model = Resnet(
            backbone=backbone, input_channels=input_channels, num_classes=num_classes
        )
        self.num_classes = num_classes

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        # self.max_accuracy = 0.0
        # self.define_metrics()

        # to save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        preds, y, loss, acc = self.get_preds_loss_accuracy(batch)

        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("train_acc_epoch", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, y, loss, acc = self.get_preds_loss_accuracy(batch)
        self.log("valid_loss_epoch", loss, on_step=False, on_epoch=True)
        # self.log("valid_acc", acc, on_step=False, on_epoch=True)
        self.log("valid_acc_step", acc, on_step=True, on_epoch=False)
        self.accuracy.update(preds, y)

        # return acc

    def on_validation_epoch_end(self):
        # self.accuracy.compute()
        self.log("valid_acc_epoch", self.accuracy.compute())
        self.accuracy.reset()

    # def validation_epoch_end(self, valid_outputs):
    #     for acc in valid_outputs:
    #         if acc > self.max_accuracy:
    #             self.max_accuracy = acc

    def get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid steps are similar"""
        x, y = batch
        y = y.type(torch.LongTensor).to(x.device)
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = self.accuracy(preds, y)
        return preds, y, loss, acc


audiomnist_classes = 10
backbone = "resnet18"
input_channels = 4
model = ResNetModule(
    backbone=backbone, input_channels=input_channels, num_classes=audiomnist_classes
)


# define the data module
class AudioMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=16,
        num_workers=4,
    ):
        super().__init__()
        """
        cifar10 dataset contains 60,000 32x32 colored images.
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        dataset = AudioMNISTDataset()

        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        valid_size = int(dataset_size * 0.1)
        test_size = int(dataset_size * 0.1)

        self.train_data, self.valid_data, self.test_data = random_split(
            dataset,
            [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

        print("train data size: ", len(self.train_data))
        print("valid data size: ", len(self.valid_data))
        print("test data size: ", len(self.test_data))

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return train_dataloader

    def val_dataloader(self):
        valid_dataloader = DataLoader(
            self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return valid_dataloader


audio_mnist = AudioMNISTDataModule()

# set the wandb logger
wandb_logger = WandbLogger(
    entity="dl-eoct",
    project="sb-warmupproject",
    name="vanilla-resnet18",
)
checkpoint_callback = ModelCheckpoint(
    dirpath="/home/mila/m/maab.elrashid/scratch/warmup_project",
    filename="resnet18" + "{epoch}-{valid_acc_epoch:.2f}",
    save_top_k=1,
    monitor="valid_acc_epoch",
)

trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=2000,
    callbacks=[
        EarlyStopping(monitor="valid_acc_epoch", mode="max", patience=20),
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
    ],
    accelerator="gpu",
)

trainer.fit(model, audio_mnist)

# print(f"Highest valid accuracy: {model.max_accuracy:.4f}")

wandb.finish()
