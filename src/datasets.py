import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
import os
import sys

from albumentations.augmentations.crops.transforms import RandomCrop


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from torchvision import io


from UNeXt.archs import UNext
from UNeXt.losses import BCEDiceLoss
from pycocotools.coco import COCO


class LungTumorDataset(Dataset):
    def __init__(self, root_dir, ann_file, transforms=None, imageSize=512):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            ann_file (string): Path to the json file with annotations.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat2id = {cat['name']: cat['id'] for cat in self.cats}
        # self.color2id = {v: k for k, v in self.id2color.items()}
        self.id2cat = {cat['id']: cat['name'] for cat in self.cats}
        self.name2color = {'CD3': np.array([255, 255,   0]),
                           'Nkp46': np.array([255,   0,   0]),
                           'Tryptase': np.array([  0, 255, 230]),
                           'Necrotic': np.array([255,   0, 255]),
                           'Fibrotic': np.array([218, 215, 215]),
                           'Blood vessel': np.array([  0,   0, 255]),
                           'white': np.array([255, 253, 224]),
                           'hole': np.array([20, 20, 20]),
                           'Tumor Islet': np.array([255, 128,   0]),
                           'Tumor': np.array([255, 255, 184]),
                           'Ash': np.array([100, 100, 100]),
                           'mucin': np.array([214, 237, 255])}
        self.imageSize = imageSize
        self.transforms = RandomCrop(imageSize, imageSize, True)



    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            Returns:
            sample (dict): Sample
        """
        img_id = self.ids[idx]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        img_path = os.path.join(self.root_dir, self.coco.imgs[img_id]['file_name'])
        img = io.read_image(img_path, io.ImageReadMode.RGB)/255
        imgShape = img.shape



        groundTruth = np.zeros((len(self.cats), img.shape[1], img.shape[2]), dtype=np.bool_)

        for annotation in annotations:
            catId = annotation['category_id']
            # print(groundTruth[catId].shape)
            if(len(annotation["segmentation"][0]) > 4):
                groundTruth[catId] = np.logical_or(groundTruth[catId], self.coco.annToMask(annotation))


        groundTruth = torch.from_numpy(groundTruth).float()

        # add padding to the image and masks to make them the same size
        padx = self.imageSize - imgShape[2]
        pady = self.imageSize - imgShape[1]

        if padx > 0 or pady > 0:
            finalImage = torch.ones((3, self.imageSize, self.imageSize)).float()
            finalGroundTruth = torch.zeros((len(self.cats), self.imageSize, self.imageSize)).float()

            finalImage[:img.shape[0], :img.shape[1], :img.shape[2]] = img
            finalGroundTruth[:, :img.shape[1], :img.shape[2]] = groundTruth

            if self.transforms is not None:
                augmented = self.transforms(image=finalImage.permute(1, 2, 0), mask=finalGroundTruth.permute(1, 2, 0))
                finalImage, finalGroundTruth = augmented["image"].permute(2, 0, 1), augmented["mask"].permute(2, 0, 1)
            return finalImage, finalGroundTruth

        if self.transforms is not None:
            augmented = self.transforms(image=img.permute(1, 2, 0), mask=groundTruth.permute(1, 2, 0))
            img, groundTruth = augmented["image"].permute(2, 0, 1), augmented["mask"].permute(2, 0, 1)

        return img, groundTruth

    def get_cat_id(self, cat_name):
        return self.cat2id[cat_name]

    def get_cat_name(self, cat_id):
        return self.id2cat[cat_id]


    def exportXml(self, img_id, mask):
        """
        export the mask to icy xml format
        Args:
            img_id (int): Index
            mask (np.array): Mask
            output_dir (string): Directory to save the xml file.
        """
        img_path = os.path.join(self.root_dir, self.coco.imgs[img_id]['file_name'])

        xml_filename = os.path.split(img_path)[1].split('.')[0] + '.xml'
        




    
    def draw_mask(self, img, mask, maskGt=None):
        """
        Draw mask on image
        """
        img = img.numpy()
        mask = mask.numpy().astype(np.uint8)


        fig = plt.figure(figsize=(10, 10))
        plt.imshow(img.transpose(1, 2, 0))

        if maskGt is  None:
            fig, axes = plt.subplots(2, 6, figsize=(15, 15))
            axes = axes.flatten()
            for i in range(len(axes)):
                axes[i].set_title(self.get_cat_name(i))
                axes[i].imshow(mask[i, :, :], cmap='gray')
                axes[i].axis('off')

        else:
            
            fig, axes = plt.subplots(2, 6, figsize=(32, 15))
            axes = axes.flatten()
            for i in range(len(axes)):
                # show intersection with ground truth as white
                # false positif as red
                # false negative as blue

                axes[i].set_title(self.get_cat_name(i))
                fp = np.logical_and(mask[i, :, :], np.logical_not(maskGt[i, :, :]))[:, :, None]
                fn = np.logical_and(np.logical_not(mask[i, :, :]), maskGt[i, :, :])[:, :, None]
                tp = np.logical_and(mask[i, :, :], maskGt[i, :, :])[:, :, None]

                # axes[i].imshow(tp*np.array([0, 255, 0]) + fp*np.array([255, 0, 0]) + fn*np.array([0, 0, 255]))
                # show fp fn tp
                axes[i].imshow(tp*np.array([0, 255, 0]) + fp*np.array([255, 0, 0]))

                

                axes[i].axis('off')

        plt.show()
        # else:
        #     for i in range(len(self.cats)):
        #         # overlay the ith channel of the mask using the ith color
        #         color = self.name2color[self.id2cat[i]]
        #         plt.imshow(mask[i, :, :][:, :, None]*color[None, None, :], alpha=0.2)
        #     # plt.show()