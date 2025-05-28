import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.val_utils import compute_psnr_ssim
from pytorch_msssim import ssim


class TrainPSNRCallback(pl.Callback):
    def __init__(self, trainset, batch_size=8, num_workers=4):
        super().__init__()
        self.trainset = trainset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        dataloader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        psnr_total, n_count = 0.0, 0
        max_images = 100  # 最多計算 100 張
        image_count = 0

        with torch.no_grad():
            for [clean_name, de_id], degrad_patch, clean_patch in dataloader:
                # 確認剩下的圖片數量
                batch_size = degrad_patch.size(0)
                if image_count >= max_images:
                    break
                if image_count + batch_size > max_images:
                    # 只取需要的前幾張
                    take = max_images - image_count
                    degrad_patch = degrad_patch[:take]
                    clean_patch = clean_patch[:take]
                    # 你如果有 clean_name/de_id 也要一併處理
                restored = pl_module(degrad_patch.to(pl_module.device))
                batch_psnr, _, batch_n = compute_psnr_ssim(
                    restored, clean_patch.to(pl_module.device)
                )
                psnr_total += batch_psnr * batch_n
                n_count += batch_n
                image_count += batch_n  # 或 batch_size

        avg_psnr = psnr_total / n_count if n_count else 0.0

        print(
            f"[Epoch {trainer.current_epoch}] Logging train_psnr={avg_psnr} (only first 100 images)"
        )

        if trainer.is_global_zero and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {"train_psnr": avg_psnr, "epoch": trainer.current_epoch}
            )

        pl_module.train()


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.pixel_loss_fn = nn.L1Loss()
        self.alpha = 0.5
        self.beta = 0.5

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # L1 Loss
        pixel_loss = self.pixel_loss_fn(restored, clean_patch)
        # SSIM Loss (1 - ssim, 要maximize ssim)
        ssim_val = ssim(restored, clean_patch, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val

        # 混合 Loss
        total_loss = self.alpha * pixel_loss + self.beta * ssim_loss

        self.log("train_loss", total_loss)
        self.log("pixel_loss", pixel_loss)
        self.log("ssim_loss", ssim_loss)

        return total_loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150
        )

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train_mix")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1
    )
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    model = PromptIRModel()
    psnr_callback = TrainPSNRCallback(
        trainset, batch_size=opt.batch_size, num_workers=opt.num_workers
    )

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback, psnr_callback],
    )
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == "__main__":
    main()
