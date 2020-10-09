import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as Transforms
import random
from PIL import Image
import ctypes
import multiprocessing as mp
import cv2
import numpy as np
import glob
import os
from natsort import natsorted

class DataLoad:
    def __init__(self, x_root, y_root, batch_size=1, num_workers=0, img_size=283, crop_size=256, zero_pad = 6, fast_load=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.crop_size = crop_size
        self.zero_pad = zero_pad
        self.fast_load = fast_load


        self.data_set_x = self.load_data(x_root)
        self.size_x = len(self.data_set_x) 
        self.x_iter = iter(self.data_set_x)
        
        self.data_set_y = self.load_data(y_root)
        self.size_y = len(self.data_set_y)
        self.y_iter = iter(self.data_set_y)
#         print("Made")
    
    def load_data(self, dataroot):
        if self.fast_load == True:
            dataset = FromFolder(dataroot, self.zero_pad, self.img_size, self.crop_size)
        else:
            dataset = MyFolder(root=dataroot,
                           transform=Transforms.Compose([
                               Transforms.Resize(self.img_size),
                               Transforms.RandomCrop(self.crop_size),
                               Transforms.ToTensor(),
                               Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
            
        data_load = DataLoader(dataset, batch_size=self.batch_size,
                                        shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)
        return data_load
    
    def next_pair(self, device):
        try:
            img_x = self.x_iter.next()

        except StopIteration:
            self.x_iter = iter(self.data_set_x)
            img_x = self.x_iter.next()
        
        try:
            img_y = self.y_iter.next()

        except StopIteration:
            self.y_iter = iter(self.data_set_y)
            img_y = self.y_iter.next()
        return img_x.to(device), img_y.to(device)



# From folder with caching of dataset, but roughly 33% faster after first epoch, not thoroughly tested
# ref: https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960
class FromFolder(Dataset):
    def __init__(self,root, zero_pad, input_size, crop_size, channels = 3):
        paths = glob.glob(root + "/*.jpg")
        self.root = root
        self.total_pixels = 0
        self.nb_samples = len(paths)
        self.input_size = input_size
        self.channels = channels
        self.zero_pad = zero_pad
        self.crop_size = crop_size
        # This code below is if the training images are all different sizes
        # print(self.nb_samples)
        # for x in paths:
        #     # print(x)
        #     img = cv2.imread(x)
        #     h, w, c = img.shape
        #     self.total_pixels += h*w*c

        shared_array_base = mp.Array(ctypes.c_float, self.nb_samples * input_size * input_size * channels)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.shared_array = shared_array.reshape(self.nb_samples, channels, input_size, input_size)
        self.shared_array[:] = -2
    
    def __getitem__(self, index):
        index_image = index + 1
        x = self.shared_array[index]
        if x.max() == -2:
#             print('Filling cache for index {}'.format(index))
            im = Image.open(self.root + (str(index_image)).zfill(self.zero_pad) + ".jpg")
            im = Transforms.Resize(self.input_size)(im)
            im = np.array(im)
            im = (im  - 127.5) / 127.5
            self.shared_array[index] = im.transpose(2,0,1)
            x = self.shared_array[index]
        x = self.random_crop(x, self.crop_size)
        x = torch.from_numpy(x)
        return x
            
    
    def __len__(self):
        return self.nb_samples
    
    def random_crop(self, img, dim):
        x = random.randint(0, img.shape[2] - dim)
        y = random.randint(0, img.shape[1] - dim)
        img = img[:, y:y+dim, x:x+dim]
        return img

# Simple from folder remake that doesnt require subfolders
class MyFolder(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        all_imgs = os.listdir(root)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

def main():
    root = "/media/fico/Data/Kaggle/vangogh2photo/trainB/"
    data = FromFolder(root, 6, 283, 256)
    print(data.__len__())
    print(data.__getitem__(0).shape)



if __name__ == "__main__":      
    main()