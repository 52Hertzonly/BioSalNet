import os
import torch
import numpy as np
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from loss import *


class SaliconTrainDataset(Dataset):
    def __init__(self,root,df_x, df_y,transform=None,size=(352,352)) -> None:
        super().__init__()
        self.root = root
        self.img_transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD),
            ])
        self.size = size 
        self.images = df_x.tolist()
        self.maps = df_y.tolist()

    def __getitem__(self, idx):
        img_path = os.path.join(self.root,'images','train',self.images[idx])
        depth_path = os.path.join(self.root,'salicon_depth','train',self.maps[idx])
        map_path = os.path.join(self.root,'maps','train',self.maps[idx])
        

        image = Image.open(img_path).convert("RGB")
#         depth_image = Image.open(depth_path).convert("L")
        depth_image = np.array(Image.open(depth_path).convert("L"))
        depth_image = depth_image.astype('float')
        
        # ground-truth
        map = np.array(Image.open(map_path).convert("L"))
        map = map.astype('float')
        map2 = cv2.resize(map,(self.size[0]//8,self.size[1]//8))
        map3 = cv2.resize(map,(self.size[0]//4,self.size[1]//4))
        map = cv2.resize(map,(self.size[1],self.size[0])) 
        depth_image = cv2.resize(depth_image,(self.size[1],self.size[0]))

        # transform
        image = self.img_transform(image)
#         depth_image = self.img_transform(depth_image)
        if np.max(map) > 1.0:
            map = map / 255.0
        assert np.min(map) >= 0.0 and np.max(map) <= 1.0
        if np.max(depth_image) > 1.0:
            depth_image = depth_image / 255.0
        assert np.min(depth_image) >= 0.0 and np.max(depth_image) <= 1.0

        return image,torch.FloatTensor(depth_image),torch.FloatTensor(map)

        #return image, torch.FloatTensor(map)

    def __len__(self):
        return len(self.images)

class SaliconValDataset(Dataset):
    def __init__(self,root,df_x, df_y,transform=None,size=(352,352)) -> None:
        super().__init__()
        self.root = root
        self.img_transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD),
            ])
        self.size = size 
        self.images = df_x.tolist()
        self.maps = df_y.tolist()


    def __getitem__(self, idx):
        img_path = os.path.join(self.root,'images','val',self.images[idx])
        depth_path = os.path.join(self.root,'salicon_depth','val',self.maps[idx])
        map_path = os.path.join(self.root,'maps','val',self.maps[idx])

        image = Image.open(img_path).convert("RGB")
#         depth_image = Image.open(depth_path).convert("L")
        depth_image = np.array(Image.open(depth_path).convert("L"))
        depth_image = depth_image.astype('float')
        # ground-truth
        map = np.array(Image.open(map_path).convert("L"))
        map = map.astype('float')
        map = cv2.resize(map,(self.size[1],self.size[0])) 
        
        depth_image = cv2.resize(depth_image,(self.size[1],self.size[0]))

        # transform
        image = self.img_transform(image)
#         depth_image = self.img_transform(depth_image)
        if np.max(map) > 1.0:
            map = map / 255.0
        assert np.min(map) >= 0.0 and np.max(map) <= 1.0, "Ground-truth not in [0,1].{} {}".format(np.min(map), np.max(map))
        if np.max(depth_image) > 1.0:
            depth_image = depth_image / 255.0
        assert np.min(depth_image) >= 0.0 and np.max(depth_image) <= 1.0

        return image,torch.FloatTensor(depth_image),torch.FloatTensor(map)

    def __len__(self):
        return len(self.images)
