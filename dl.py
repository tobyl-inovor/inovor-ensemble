'''
Dataloader for binary image classification dataset
'''
import os
import torch
import math
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BinaryDataset(Dataset):
    """
    Dataset builder for binary image classification dataset. 
    
    dir:    path to directory containing images
    df:     dataframe of csv (contains image id's and masks)
    
    """

    def __init__(self, dir, df, transform=None):

        self.dir = dir
        self.df = df[df.ImageId != "63849d9ce.jpg"] # remove corrupt image from dataframe - specific to airbus dataset
        self.transform = transform

        # extract binary labels
        self.labels = self.df["Class"].to_numpy()
            
    def __len__(self):
        return len(self.df["ImageId"])

    def __getitem__(self, idx):

        # load in image
        img = Image.open(self.dir + "/" + self.df["ImageId"].iloc[idx], mode='r')

        # apply image transformation
        if self.transform is not None:
            img = self.transform(img)

        # return image and label
        return img, torch.tensor(self.labels[idx]).long()


# preprocessing specific to airbus ship detection dataset
def classif_imgs_from_masks(masks_orig, dir):
    masks = masks_orig
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['Class'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)

    # remove too small files
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                                   os.stat(
                                                                       os.path.join(dir,
                                                                                    c_img_id)).st_size / 1024)
    size_thresh = 50 # kb
    corrupt = 0
    for i in range(len(unique_img_ids)):
        if unique_img_ids['file_size_kb'][i] <= size_thresh:
            corrupt += 1

    unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > size_thresh]

    masks.drop(['ships'], axis=1, inplace=True)

    return masks, unique_img_ids, corrupt