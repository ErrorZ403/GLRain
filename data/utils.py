import numpy as np
import cv2

def get_random_crop(HQ, LQ, crop_h, crop_w):
  assert len(HQ.shape) == 3

  max_w = HQ.shape[1] - crop_w
  max_h = HQ.shape[0] - crop_h

  x = np.random.randint(0, max_w)
  y = np.random.randint(0, max_h)

  crop_HQ = HQ[y : y + crop_h, x : x + crop_w, :]
  crop_LQ = LQ[y : y + crop_h, x : x + crop_w, :]

  return crop_HQ, crop_LQ

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))