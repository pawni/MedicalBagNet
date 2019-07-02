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

def flip(img):
    if np.random.random(1) > 0.5:
        img = np.array(img[:, :, ::-1])
    return img

def whiten(img):
    img -= img.mean()
    img /= img.std()

    return img


class IxiDataset(Dataset):
    age_mean = 0.  # 45.25667351129839
    age_std = 1.  # 16.922015662026563

    attribute_dict = {
        'sex': lambda x: np.int64(x['SEX_ID (1=m, 2=f)'] - 1),
        'age': lambda x: np.float32(
            (x['AGE'] - IxiDataset.age_mean) / IxiDataset.age_std) # whiten age
    }

    def __init__(self, base_path, csv_path, scale='2mm', attribute='sex',
                 augment=False):
        self.base_path = base_path
        self.scale = scale
        self.attribute = attribute
        self.augment = augment
        self.csv_path = os.path.join(self.base_path, csv_path)
        self.csv = pd.read_csv(self.csv_path)
        self.crop_size = [64, 96, 96] if scale == '2mm' else [128, 192, 192]

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.loc[index]
        att = self.attribute_dict[self.attribute](row)

        t1p = os.path.join(self.base_path, self.scale, row['IXI_ID'],
                           'T1_{}.nii.gz'.format(self.scale))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1p)))

        img = t1.astype(np.float32)

        img = extract_random_array(img, self.crop_size)

        img = whiten(img)

        if self.augment:
            img = flip(img)

        img = img[np.newaxis]

        return img, att, row['IXI_ID']
