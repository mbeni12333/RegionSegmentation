import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
import os
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from torchvision import io


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


from UNeXt.archs import UNext
from UNeXt.losses import BCEDiceLoss
from pycocotools.coco import COCO




class SegModel(pl.LightningModule):
    def __init__(self, backbone, transforms=None):
        super(SegModel, self).__init__()

        self.net = backbone
        self.transforms = transforms
        self.learning_rate = 1e-4
        self.loss = BCEDiceLoss()
        self.name = 'UNext'
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.float()
        out = self.forward(img)
        # loss = F.cross_entropy(out, mask, ignore_index = 250)
        loss = self.loss(out, mask)
        self.log('train_loss', loss)
        return {'loss' : loss}
        
    def validation_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.float()
        out = self.forward(img)
        # loss = F.cross_entropy(out, mask, ignore_index = 250)
        loss = self.loss(out, mask)
        self.log('val_loss', loss)
        return {'val_loss' : loss}
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]


