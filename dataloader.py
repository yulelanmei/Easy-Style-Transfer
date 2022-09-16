import os
import torch as th
import torchvision as tv
import torch.utils.data as Data
from PIL import Image

def get_image_transform(resize= 256, mean= (0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)): 
    return tv.transforms.Compose([  tv.transforms.Resize(resize), 
                                    tv.transforms.CenterCrop(resize), 
                                    tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean= mean, 
                                                            std=  std)])

def get_image_list(dataset_path: str, datanum= 1000): 
    return [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)][:datanum] 

def image_transform(img_dir: str): 
    img = Image.open(img_dir).convert('RGB')         
    img = get_image_transform()(img)
    return img

class ImageLoder(Data.Dataset): 
    def __init__(self, dataset_path: str, datanum = 1000, preload= False):
        datadir = get_image_list(dataset_path, datanum)
        if preload: 
            datalist = map(image_transform, datadir)
            self.data = th.concat(datalist)

        else: 
            self.datalist = datadir
            self.transform = get_image_transform()
        self.preload = preload
        self.length = len(datadir)

    def __getitem__(self, index):
        if self.preload: 
            return self.data[index]
        else: 
            return self.transform(self.datalist[index])

    def __len__(self): 
        return self.length                                

    