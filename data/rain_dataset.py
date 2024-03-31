import torch
from torch.utils.data import Dataset

import cv2
from data.utils import get_random_crop

from albumentations import Compose, Flip

class RainDataset(Dataset):
  def __init__(self, HQ_path, LQ_path, keys=('HQ', 'LQ'), is_val=False):
    self._is_val = is_val
    
    self.rainy_images = self._get_files(LQ_path)
    self.gt_images = self._get_files(HQ_path)
    

    self.hq_key = keys[0]
    self.lq_key = keys[1]

    self.Aug = Compose([
        Flip(p=0.5)]
    )

  def _get_files(self, path):
    files = sorted(os.listdir(path))
    files = [f'{path}/{file}' for file in files]
    return files

  def _prepare_data(self, image):
    image = torch.tensor(image)
    image = torch.permute(image, (2, 0, 1))
    image = image / 255.0
    
    return image

  def __len__(self):
    return len(self.rainy_images)

  def __getitem__(self, index):
    rainy_image = self.rainy_images[index]
    gt_image = self.gt_images[index]

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
    }