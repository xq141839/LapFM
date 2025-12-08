import os
from skimage import io, transform, color, img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as pytorch_transforms
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensor 
import albumentations as A
from PIL import Image


class SurgLoader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'dataset/v3/'
            self.data_name = data_name
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.sam_img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = self.jsfiles[idx]

            image_path = os.path.join(self.path,'image/',image_id)
            mask_path = os.path.join(self.path,'npy/',image_id)
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32') 

            mask_map = np.load(mask_path+'.npy').astype(np.uint8)
            mask_map_p = mask_map[:, :, 0].copy().astype(np.uint8)
            mask_map_c1 = mask_map[:, :, 1].copy().astype(np.uint8)
            mask_map_c2 = mask_map[:, :, 2].copy().astype(np.uint8)
            mask_map_c3 = mask_map[:, :, 3].copy().astype(np.uint8)
            mask_map_p1 = (mask_map_p == 1).astype(np.uint8)
            mask_map_p2 = (mask_map_p == 2).astype(np.uint8)
            mask_map_p3 = (mask_map_p == 3).astype(np.uint8)
            mask_map_p = cv2.resize(mask_map_p, (self.sam_img_size, self.sam_img_size), interpolation=cv2.INTER_NEAREST)
            mask_map_p1 = cv2.resize(mask_map_p1, (self.sam_img_size, self.sam_img_size), interpolation=cv2.INTER_NEAREST)
            mask_map_p2 = cv2.resize(mask_map_p2, (self.sam_img_size, self.sam_img_size), interpolation=cv2.INTER_NEAREST)
            mask_map_p3 = cv2.resize(mask_map_p3, (self.sam_img_size, self.sam_img_size), interpolation=cv2.INTER_NEAREST)
            mask_map_c1 = cv2.resize(mask_map_c1, (self.sam_img_size, self.sam_img_size), interpolation=cv2.INTER_NEAREST)
            mask_map_c2 = cv2.resize(mask_map_c2, (self.sam_img_size, self.sam_img_size), interpolation=cv2.INTER_NEAREST)
            mask_map_c3 = cv2.resize(mask_map_c3, (self.sam_img_size, self.sam_img_size), interpolation=cv2.INTER_NEAREST)

            # print(type(mask_map_p))
            mask_map_p = torch.from_numpy(mask_map_p).long().unsqueeze(0)
            mask_map_p1 = torch.from_numpy(mask_map_p1).long().unsqueeze(0)
            mask_map_p2 = torch.from_numpy(mask_map_p2).long().unsqueeze(0)
            mask_map_p3 = torch.from_numpy(mask_map_p3).long().unsqueeze(0)
            mask_map_c1 = torch.from_numpy(mask_map_c1).long().unsqueeze(0)
            mask_map_c2 = torch.from_numpy(mask_map_c2).long().unsqueeze(0)
            mask_map_c3 = torch.from_numpy(mask_map_c3).long().unsqueeze(0)

            
            data_group = self.transforms(image=img)
            img_re = data_group['image']
            # mask_map_p = data_group['mask']
            # mask_map_c1 = data_group['mask2']
            # mask_map_c2 = data_group['mask3']
            # mask_map_c3 = data_group['mask4']


            data_dict = {"image_id": image_id,
                        "image": img_re,
                        "mask_map_p": mask_map_p,
                        "mask_map_p1": mask_map_p1,
                        "mask_map_p2": mask_map_p2,
                        "mask_map_p3": mask_map_p3,
                        "mask_map_c1": mask_map_c1,
                        "mask_map_c2": mask_map_c2,
                        "mask_map_c3": mask_map_c3
            }

            return data_dict
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x


class SurgLoaderOne(Dataset):
        def __init__(self, parent_cls, sub_cls, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'dataset/'
            self.parent_cls = parent_cls
            self.sub_cls = sub_cls
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.sam_img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = self.jsfiles[idx]

            image_path = os.path.join(self.path,'image/',image_id)
            mask_path = os.path.join(self.path,'npy/',image_id)
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32') 

            mask_map = np.load(mask_path+'.npy').astype(np.uint8)
           
            mask_map_p = mask_map[:, :, self.parent_cls].copy().astype(np.uint8)
            mask_map_p[mask_map_p != self.sub_cls] = 0
            mask_map_p[mask_map_p > 0] = 255
            

            data_group = self.transforms(image=img, mask=mask_map_p)
            img_re = data_group['image']
            mask = data_group['mask']

            

            data_dict = {"image_id": image_id,
                        "image": img_re,
                        "mask": mask,
            }

            return data_dict
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x

class SurgLoaderP(Dataset):
        def __init__(self, parent_cls, sub_cls, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'dataset/'
            self.parent_cls = parent_cls
            self.sub_cls = sub_cls
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.sam_img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = self.jsfiles[idx]

            image_path = os.path.join(self.path,'image/',image_id)
            mask_path = os.path.join(self.path,'npy/',image_id)
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32') 

            mask_map = np.load(mask_path+'.npy').astype(np.uint8)
           
            mask_map_p = mask_map[:, :, 0].copy().astype(np.uint8)
            mask_map_p[mask_map_p != self.parent_cls] = 0
            mask_map_p[mask_map_p > 0] = 255
            

            data_group = self.transforms(image=img, mask=mask_map_p)
            img_re = data_group['image']
            mask = data_group['mask']

            

            data_dict = {"image_id": image_id,
                        "image": img_re,
                        "mask": mask,
            }

            return data_dict
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.sam_img_size - h
            padw = self.sam_img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x