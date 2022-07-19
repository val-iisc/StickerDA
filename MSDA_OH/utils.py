import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F1
import torch.nn.functional as F
import torch.nn as nn


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = F1.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)



class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    if self.count:
      return self.sum/self.count
    else:
      return self.sum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def torch_save(weights, path, **kwargs):
  '''
  kwargs can be used to path things like optimizer weights / iteartion number etc.
  '''
  data = {'weights': weights}
  data.update(kwargs)

  if not os.path.isdir(os.path.split(path)[0]): os.mkdir(os.path.split(path)[0])

  torch.save(data, path)

def torch_load(path, key="weights", device=torch.device('cpu')):
  '''
  Possible keys should be known beforehand

  load_state_dict should be done in client code
  '''
  if not os.path.exists(path):
    raise Exception("Checkpoint doesn't exist at {}".format(path))
  checkpoint = torch.load(path, map_location = device)
  if not key in checkpoint:
    raise Exception("Key {} doesn't exist".format(key))
  return checkpoint[key]


def plot_samples(image_tensors, targets, outputs, step = None, idx_to_class=None, title = "Sample Predictions"):
    """
    for plotting on wandb
    l_to_t: labels to text
    """

    images = image_tensors.cpu().clone()

    unorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    images = unorm(images)

    images = [Image.fromarray(np.transpose(np.uint8(images[i]*255), (1,2,0))) for i in range(images.shape[0])]
    outputs = torch.max(nn.Softmax(dim=1)(outputs), 1)[1]

    classes_text = [(targets[i].cpu().item(), outputs[i].cpu().item()) for i, img in enumerate(images)]
    if idx_to_class:
      classes_text = [(idx_to_class[str(a)], idx_to_class[str(b)]) for a,b in classes_text]

    if step:
      wandb.log({title: [wandb.Image(img, caption="Target: %s; Predicted: %s" % (classes_text[i][0], classes_text[i][1])) for i, img in enumerate(images)]}, step=step)
    else:
      wandb.log({title: [wandb.Image(img, caption="Target: %s; Predicted: %s" % (classes_text[i][0], classes_text[i][1])) for i, img in enumerate(images)]})

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def sticker_train(resize_size=224):
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
    ])

def sticker_test(resize_size=224, alexnet=False):
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
    ])
