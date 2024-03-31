import torch
from torch.utils.data import Dataset

import cv2
from metrics_and_losses import get_random_crop
import os

from albumentations import Compose, Flip

class GTRainDataset(Dataset):
  def __init__(self, scene_root_dir, keys=('HQ', 'LQ'), is_val=False):
    self._is_val = is_val
    
    self.img_paths = self._get_files(scene_root_dir)

    self.hq_key = keys[0]
    self.lq_key = keys[1]

    self.Aug = Compose([
        Flip(p=0.5)]
    )

  def _get_files(self, path):
    drs = sorted(os.listdir(path))
    drs = [f'{path}/{dr}' for dr in drs]
    files = []
    for dr in drs:
        files_temp = [f'{dr}/{file}' for file in os.listdir(dr) if '-R-' in file]
        files += files_temp
    
    return files

  def _prepare_data(self, image):
    image = torch.tensor(image)
    image = torch.permute(image, (2, 0, 1))
    image = image / 255.0
    
    return image

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, index):
    rainy_image = self.img_paths[index]
    gt_image = self.img_paths[index][:-9] + 'C-000.png'

    rainy_image = cv2.cvtColor(cv2.imread(rainy_image), cv2.COLOR_BGR2RGB)
    gt_image = cv2.cvtColor(cv2.imread(gt_image), cv2.COLOR_BGR2RGB)

    if not self._is_val:
      gt_image, rainy_image = get_random_crop(gt_image, rainy_image, crop_h=128, crop_w=128)
      transformed = self.Aug(image=rainy_image, mask=gt_image)
      rainy_image = transformed['image']
      gt_image = transformed['mask']

    rainy_image = self._prepare_data(rainy_image)
    gt_image = self._prepare_data(gt_image)
    
    return {
       self.hq_key: gt_image,
       self.lq_key: rainy_image,
       'name': self.img_paths[index].split('/')[-1]
    }