import copy
import nibabel as nib
import os
import SimpleITK as sitk
import torch
import sys
sys.path.append(".")
from torch.utils.data import Dataset

import numpy as np
np.random.seed(0)
import random
def random_mirror_flip(imgs_array, prob=0.5):
    """
    Perform flip along each axis with the given probability; Do it for all voxels；
    labels should also be flipped along the same axis.
    :param imgs_array:
    :param prob:
    :return:
    """
    for axis in range(1, len(imgs_array.shape)):
        random_num = np.random.random()
        if random_num >= prob:
            if axis == 1:
                imgs_array = imgs_array[:, ::-1, :]
            if axis == 2:
                imgs_array = imgs_array[:, :, ::-1]
    return imgs_array
def preprocess_img(img):
    c = img.shape[0]
    for i in range(0, c, 4):
        a = np.max(img[i, i+4])
        img[i, i+4] = img[i, i+4] * 255. / a
    return img
def random_crop(img, crop_size):
    c, h, w = img.shape
    min_size = min(h, w)
    assert min_size > crop_size[0]
    h_margin = (h - crop_size[0]) // 2
    w_margin = (w - crop_size[1]) // 2
    random_h = int(np.random.uniform(0, h_margin, 1))
    random_w = int(np.random.uniform(0, w_margin, 1))
    random_h = int(h_margin // 2)
    random_w = int(w_margin // 2)
    new_img = img[:, h_margin-random_h:h_margin+crop_size[0]-random_h, w_margin-random_w:w_margin+crop_size[1]-random_w]
    return new_img
def preprocess_label(img, single_label=None):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """

    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET) - orange
    ed = img == 2  # Peritumoral Edema (ED) - yellow
    et = img == 4  # GD-enhancing Tumor (ET) - blue
    bg = (img!=1)*(img!=2)*(img!=4)
    # print("ed",et.shape)
    if not single_label:
        # return np.array([ncr, ed, et], dtype=np.uint8)
        return np.array([ed, bg, ncr, et], dtype=np.uint8)
    elif single_label == "WT":
        img[ed] = 1.
        img[et] = 1.
        img[ncr] = 1.
        img[bg] = 0.
    elif single_label == "TC":
        img[ncr] = 0.
        img[bg] = 0.
        img[ed] = 1.
        img[et] = 1.
    elif single_label == "ET":
        img[ncr] = 0.
        img[ed] = 0.
        img[bg] = 0.
        img[et] = 1.
    else:
        raise RuntimeError("the 'single_label' type must be one of WT, TC, ET, and None")
    # print("image", img.shape)
    return img[np.newaxis, :]
class BTS_data(Dataset):
    def __init__(self, opt, file):
        super(BTS_data, self).__init__()
        self.crop_size = opt.crop_size
        self.file = file
        self.path = opt.data_folder
    def __len__(self):
        return len(self.file)
    def __getitem__(self, index):
        path = os.path.join(self.path, self.file[index])
        imgs_npy = np.load(path)[0]
        cur_with_label = imgs_npy.copy()
        cur_with_label = random_crop(cur_with_label, self.crop_size)
        cur_with_label = random_mirror_flip(cur_with_label)
        source_data = cur_with_label[0:12]
        label = preprocess_label(cur_with_label[-1])
        return torch.from_numpy(source_data.copy()), torch.from_numpy(label.copy())