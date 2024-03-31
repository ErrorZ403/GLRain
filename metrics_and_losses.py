import torch
import pytorch_ssim
import torch.nn as nn

def PSNR(target, predicted):
  mse = ((target - predicted) ** 2).mean()
  #print(mse)
  
  if mse == 0:
    return 100

  #mx_value = predicted.max()

  psnr = 20 * torch.log10(1.0 / mse.sqrt())

  return psnr

def SSIM(target, predicted):
  SSIM = pytorch_ssim.SSIM()

  return SSIM(target, predicted)

class L1Loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, target, predicted):
    L1 = torch.abs(target - predicted).mean()

    return L1

class L2Loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, target, predicted):
    #print(target.shape)
    #print(predicted.shape)
    L2 = ((target - predicted) ** 2).mean()

    return L2

class SSIMLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.SSIM_calc = pytorch_ssim.SSIM()

  def forward(self, target, predicted):
    ssim = self.SSIM_calc(target, predicted)

    return 1 - ssim

class LossCombinier(nn.Module):
  def __init__(self, losses, weights):
    assert len(losses) == len(weights)

    self.losses = losses
    self.weights = weights

  def forward(self, target, predicted):
    overall_loss = 0.0

    for i in range(len(self.losses)):
      overall_loss += self.losses[i](target, predicted) * self.weights[i]

    return overall_loss