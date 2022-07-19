import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F


SEED = 2020
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def shuffle_2(imgs, labels=None, parts=[6, 6]):
    bs = len(imgs)
    
    if len(imgs) < parts[0]*parts[1]:
        imgs = imgs.repeat((parts[0]*parts[1]+bs-1)//bs, 1, 1, 1)
        
        imgs = imgs[:parts[0]*parts[1]]
    random.shuffle(parts)

    if labels != None:
        assert(labels.size(1) > 1)

    ver, hor = parts[0], parts[1]   # [2,1]
    num = parts[0]*parts[1]

    b = imgs.size(0)
    
    assert(b%num==0)

    bs_ver = int(imgs.size(2)/ver)
    bs_hor = int(imgs.size(3)/hor)


    if ver!=1:
        img1 = [imgs[ :, :, i*bs_ver:(i+1)*bs_ver, :] for i in range(ver-1)]
        img1.append(imgs[ :, :, (ver-1)*bs_ver:, :])
        if ver==2:
            img1.reverse()
        else:
            random.shuffle(img1)
        img1 = torch.cat(img1, 2)
    else:
        img1 = imgs

    if hor!=1:
        img2 = [img1[:, :, :, i*bs_hor:(i+1)*bs_hor] for i in range(hor-1)]
        img2.append(img1[:, :, :, (hor-1)*bs_hor:])
        if hor==2:
            img2.reverse()
        else:
            random.shuffle(img2)
        img2 = torch.cat(img2, 3)
    else:
        img2 = img1

    img3 = torch.stack([img2[i*(b//num):(i+1)*(b//num)] for i in range(num)])

    if labels != None:
        labels1 = torch.stack([labels[i*(b//num):(i+1)*(b//num)] for i in range(num)])
    
    mask = torch.zeros_like(img3)
    k=0
    for i in range(ver):
        si = i*bs_ver
        ei = (i+1)*bs_ver if i < (ver-1) else imgs.size(2)
        for j in range(hor):
            sj = j*bs_hor
            ej = (j+1)*bs_hor if j < (hor-1) else imgs.size(3)

            mask[k,:,:,si:ei,sj:ej] = 1
            k+=1

    img3 = img3*mask
    img3 = torch.sum(img3, 0)
    if labels != None:
        labels2 = torch.sum(labels, 0)
    else:
        labels2 = None

    return img3

def make_dataset(image_list, labels):
    images = []
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def rgba_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGBA')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', specified_len = None):
        imgs = make_dataset(image_list, labels) # list of tuples
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        elif mode == "RGBA":
            self.loader = rgba_loader

        if not specified_len:
            self.len = len(self.imgs)
        else:
            self.len = specified_len

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.len

class ShuffledImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', specified_len = None):
        imgs = make_dataset(image_list, labels) # list of tuples
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        elif mode == "RGBA":
            self.loader = rgba_loader

        if not specified_len:
            self.len = len(self.imgs)
        else:
            self.len = specified_len

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        img = torch.squeeze(shuffle_2(torch.unsqueeze(img,0)),0)
        target = torch.tensor(10)

        return img, target

    def __len__(self):
        return self.len

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class StickerList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', specified_len = None):
        imgs = make_dataset(image_list, labels) # list of tuples
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        elif mode == "RGBA":
            self.loader = rgba_loader

        if not specified_len:
            self.len = len(self.imgs)
        else:
            self.len = specified_len

    def __getitem__(self, index):
        path, target = self.imgs[random.choice(np.arange(0, len(self.imgs)))]
        img = self.loader(path)

        """
        If img is PIL Image, mode “1”, “L”, “I”, “F” and modes with transparency (alpha channel) are not supported.

        Hence, removing the alpha channel, jittering and then putting it back.
        """

        choice = np.random.uniform()
        jitter_transform = self.get_hsv_transform(choice)
        alpha = img.split()[-1]
        jitter_img = jitter_transform(img)
        img = Image.new("RGBA", img.size, (255,255,255,0))
        img.paste(jitter_img, mask=alpha)

        # if(random.uniform(0,1) >= 0.5):
        #     img = transforms.RandomAffine(degrees=30)(img)
        # else:
        #     img = transforms.RandomAffine(degrees=0, shear=(-45,45))(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_hsv_transform(self, prob):
        if prob <= 0.33:
            return transforms.ColorJitter(hue=(-0.5, 0.5)) 
        elif prob <= 0.66:
            return transforms.ColorJitter(brightness=(0.66, 0.88), hue=(-0.5, 0.5)) 
        else:
            return transforms.ColorJitter(saturation=(0.55, 1.0), hue=(-0.5, 0.5))

    def __len__(self):
        return self.len

class BackGround(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', specified_len = None):
        self.image_list = image_list
        self.paths = os.listdir(image_list)
        self.transform=transform

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        elif mode == "RGBA":
            self.loader = rgba_loader

        self.len = len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.loader(os.path.join(self.image_list, path))
        img  = self.transform(img)
        label = torch.tensor(10)

        return img, label

    def __len__(self):
        return self.len
        

def get_x_y_mixup(img1:Image, img2:Image):

    x_locations = np.arange(start=0, stop=abs(img1.size[0] - img2.size[0]), step=1)
    y_locations = np.arange(start=0, stop=abs(img1.size[1] - img2.size[1]), step=1)
    x,y = random.choice(x_locations), random.choice(y_locations)

    return x, y

class Denormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = F.normalize(tensor, self.demean, self.destd, self.inplace)
        return torch.clamp(tensor, 0.0, 1.0)

def mixup_batch(img_batch, sticker_batch):
    """
    Images are denormalized
    Both are converted to PIL
    Pasted batch is made
    Normalization is done
    """

    denormalize = Denormalize()
    toPIL = transforms.ToPILImage()
    toTens = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    img_batch = denormalize(img_batch)

    start = True

    for i in range(img_batch.shape[0]):

        img = toPIL(img_batch[i])
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        
        sticker = toPIL(sticker_batch[i]).convert('RGBA')
     
        white_img = Image.new("RGBA", (224, 224), (0, 0, 0, 0))
        percent_sticker = random.choice([0.1, 0.15, 0.2, 0.25, 0.3 ,0.35, 0.4])
        sticker = sticker.resize((round(224 * percent_sticker), round(224 * percent_sticker)))
        x,y = get_x_y_mixup(white_img, sticker)
        white_img.paste(sticker, (x,y), sticker)
        white_img = cv2.cvtColor(np.array(white_img), cv2.COLOR_RGB2BGR)
        alpha = np.random.uniform(0.3, 0.7)

        result = np.around(alpha *img + (1-alpha)*white_img).astype(np.uint8)
        result[white_img==0] = img[white_img == 0]
        mixed_up = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        mixed_up = normalize(toTens(mixed_up)).unsqueeze(0)

        if start:
            pasted_batch = mixed_up    
            start = False           
        else:
            pasted_batch = torch.cat((pasted_batch, mixed_up), 0)      

    return pasted_batch
