import os
import sys
import numpy as np
import argparse
import time
import timm
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
import audiomentations as aaug

from models import PANNsEfficientNet, PANNsResNet50Att
from pytorch_utils import (move_data_to_device, do_mixup)
from evaluate import Evaluator
import config
from losses import ImprovedFocalLoss, ImprovedPANNsLoss

from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from config import config

class LightModel(LightningModule):
    def __init__(self, model,fold_id = 0, data_dir=data_dir,
                 batch_size=batch_size, num_workers=4, DEVICE=DEVICE, mixup = False):
        super().__init__()
        self.model = model
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.DEVICE = DEVICE
        self.loss_fn = ImprovedPANNsLoss()
        #self.loss_fn = ImprovedFocalLoss()
        self.fold_id = fold_id
        self.mixup = mixup
        #self.train_df, self.val_df = train_df, valid_df

    def forward(self, batch):
        return self.model(batch)
   
    def training_step(self, batch, batch_idx):
        y, target = batch
        
        if self.mixup:
            mixup_augmenter = Mixup(1.)
            mix_y = mixup_augmenter.get_lambda(batch_size=self.batch_size, device = DEVICE)
            mix_target = do_mixup(target,mixup_augmenter.get_lambda(batch_size=self.batch_size, device = DEVICE))
            output = model((y,mix_y))
            bceLoss = self.loss_fn(output, mix_target)
        else:
            output = model((y,None))
            bceLoss = self.loss_fn(output, target)
        loss = bceLoss
        return loss
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"train_loss": avg_loss}
    
    def validation_step(self, batch, batch_idx):
        y, target = batch
        output = model((y,None))
        bceLoss = self.loss_fn(output, target)
        loss = bceLoss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        p = torch.sigmoid(output['clipwise_output'])
        score_class, weight = lwlrap(target.cpu().numpy(), p.cpu().numpy())
        score = (score_class * weight).sum()
        self.log("Lwlrap_epoch", score, on_epoch=True, prog_bar=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad = True, weight_decay = 1e-2)        
        #optimizer = AdamP(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,)
        lr_scheduler = {"scheduler": scheduler }
        return [optimizer], [lr_scheduler]
    
    
    def prepare_data(self):
        
        image_source = train_tp_df[['recording_id', 'species_id']].drop_duplicates()

        # get lists for image_ids and sources
        image_ids = image_source['recording_id'].to_numpy()
        sources = image_source['species_id'].to_numpy()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        split = skf.split(image_ids, sources) # second arguement is what we are stratifying by

        select = self.fold_id #Use fold
        train_ix, val_ix = next(islice(split, select, select+1))

            # translate indices to ids
        train_ids = image_ids[train_ix]
        val_ids = image_ids[val_ix]

        # create corresponding dfs
        self.train_df = train_tp_df[train_tp_df['recording_id'].isin(train_ids)]
        self.val_df = train_tp_df[train_tp_df['recording_id'].isin(val_ids)]
                                                
    def train_dataloader(self):
        sampler = ImbalancedDatasetSampler(MelSpecDataset(data_dir = self.data_dir,
                                                          df = self.train_df,
                                                          is_train=True),
                                           MelSpecDataset.get_label)
        train_aug = aaug.Compose([
            aaug.AddGaussianNoise(p=0.5),
            aaug.AddGaussianSNR(p=0.5),
            aaug.Gain(p=0.5),
            aaug.AddBackgroundNoise(sounds_path = noise_dir, p=0.5)
            #aaug.Normalize(p=0.3)
            #aaug.Shift(p=0.2)
            #aaug.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            #aaug.TimeStretch(p = 0.3)
       ])

        return DataLoader(
            MelSpecDataset(
                data_dir=self.data_dir,
                df=self.train_df, 
                is_train=True,
                waveform_transforms=train_aug,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            sampler = sampler
        )

    def val_dataloader(self):
        return DataLoader(
            MelSpecDataset(
                data_dir=self.data_dir,
                df=self.val_df, 
                is_train=True
            ), 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
          
args = {
    'sample_rate': config.sample_rate,
    'window_size': config.window_size,
    'hop_size': config.hop_size,
    'mel_bins': config.mel_bins,
    'fmin': config.fmin, 
    'fmax': config.fmax,
    'classes_num': config.classes_num
}

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    model = PANNsEfficientNet(args)

    model = LightModel(models, fold_id = 0, mixup = False,batch_size = 10)

    checkpoint_callback = ModelCheckpoint(
        filepath=OUTPUT_DIR,
        monitor='Lwlrap_epoch',
        mode='max',
        prefix=f'PANNsEfficientNetb4_fold{k}_',
        save_last = True,
        save_top_k = 1,
        save_weights_only=False,
        period = 1,
        verbose = False
    )

    logger = TensorBoardLogger(
        save_dir=OUTPUT_DIR,
        name=f'lightning_logs{k}'
    )
    trainer = Trainer(
        max_epochs=EPOCH,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=int(torch.cuda.is_available()),      
    )   

    trainer.fit(model) 