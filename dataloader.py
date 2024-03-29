import os
import random
import torch as th
import torchvision as tv
import torch.utils.data as Data
from PIL import Image
from functools import partial

def get_image_transformer(resize= 256, normalized= False, mean= None, std= None):
    compose_list = [tv.transforms.Resize(resize), 
                    tv.transforms.CenterCrop(resize), 
                    tv.transforms.ToTensor()]
    if normalized:
        if (mean is not None) and (std is not None):
            compose_list.append(tv.transforms.Normalize(mean= mean, 
                                                        std=  std))                         
    return tv.transforms.Compose(compose_list)

def get_image_list(dataset_path: str, datanum= None, if_random = False): 
    list = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    if datanum is not None: 
        if datanum == 'all':
            pass 
        elif isinstance(datanum, int): 
            if datanum > len(list): 
                datanum = len(list)
            if if_random: 
                list = random.sample(list, datanum)
            else:
                list = list[:datanum]
        else:
            raise RuntimeError("datanum not a integer!\n")
    return list 

def image_transform(img_dir: str, transformer): 
    img = Image.open(img_dir).convert('RGB')         
    img = transformer(img)
    return img

class ImageLoder(Data.Dataset): 
    def __init__(self, dataset_path: str, datanum = 'all', 
                 if_random = False, preload= False, 
                 resize = 256, normalized = False, 
                 std = None, mean = None):
        datadir = get_image_list(dataset_path, datanum, if_random)
        if preload: 
            transformer = get_image_transformer(resize, normalized, std, mean)
            transtool = partial(image_transform, transformer= transformer)
            datalist = list(map(transtool, datadir))
            self.data = datalist
        else: 
            self.datalist = datadir
            self.transformer = get_image_transformer(resize, normalized, std, mean)
        self.preload = preload
        self.length = len(datadir)

    def __getitem__(self, index):
        if self.preload: 
            return self.data[index]
        else: 
            return image_transform(self.datalist[index], self.transformer)

    def __len__(self): 
        return self.length                                

    