import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data.dataset import Dataset


def extract_random_array(img, size):
    start = np.zeros((3,))
    end = np.zeros((3,))
    for i in range(3):
        start[i] = int(np.random.randint(img.shape[i] - size[i]))
        end[i] = start[i] + size[i]

    start = start.astype(np.int32)
    end = end.astype(np.int32)
    crop = img[start[0]:end[0],
               start[1]:end[1],
               start[2]:end[2]]

    return crop

def center_crop(img, size):
    start = (np.array(img.shape[:3]) - np.array(size)) // 2
    end = np.zeros((3,))
    for i in range(3):
        end[i] = start[i] + size[i]

    start = start.astype(np.int32)
    end = end.astype(np.int32)
    crop = img[start[0]:end[0],
               start[1]:end[1],
               start[2]:end[2]]

    return crop

def flip(img):
    if np.random.random(1) > 0.5:
        img = np.array(img[:, :, ::-1])
    return img

def whiten_with_mask(img, mask):
    img -= img[mask].mean()
    img /= img[mask].std()

    return img


class CamCANDataset(Dataset):
    age_mean = 0.  # 45.25667351129839
    age_std = 1.  # 16.922015662026563

    attribute_dict = {
        'sex': lambda x: np.int64(x['gender_code'] - 1),
        'age': lambda x: np.float32(
            (x['age'] - CamCANDataset.age_mean) / CamCANDataset.age_std) # whiten age
    }

    def __init__(self, csv_path, base_path='/vol/biomedic2/bglocker/CamCAN/T1w/',
                 attribute='sex', augment=False, center_crop=False):
        self.base_path = base_path
        self.attribute = attribute
        self.augment = augment
        self.csv_path = csv_path
        self.csv = pd.read_csv(self.csv_path)
        self.crop_size = [128, 160, 160]
        self.center_crop = center_crop

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.loc[index]
        att = self.attribute_dict[self.attribute](row)

        t1p = os.path.join(self.base_path,
            'sub-{}_T1w_stripped.nii.gz'.format(row['Observations']))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1p)))

        bmp = os.path.join(self.base_path,
            'sub-{}_T1w_brain_mask.nii.gz'.format(row['Observations']))
        bm = sitk.GetArrayFromImage(sitk.ReadImage(str(bmp)))

        img = t1.astype(np.float32)

        bm = bm.astype(np.bool)

        img = whiten_with_mask(img, bm)

        if self.center_crop:
            img = center_crop(img, self.crop_size)
        else:
            img = extract_random_array(img, self.crop_size)

        if self.augment:
            img = flip(img)

        img = img[np.newaxis]

        return img, att, row['Observations']
