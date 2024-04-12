import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import (ToTensor, Normalize)

def get_pilimg(img_dir: str):
    pilimg = Image.open(img_dir).convert('RGB')
    return pilimg

def pilimg2tensor(pilimg, cuda= False, mean= None, std= None): 
    tensor = ToTensor()(pilimg)
    if (mean is not None) and (std is not None): 
        tensor = Normalize(mean, std, inplace= True)(tensor)
    return tensor.unsqueeze(0).cuda() if cuda else tensor

def tensor2img(tensor, std= None, mean= None):
    img = tensor.cpu().detach().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))
    if std is not None:
        img = img * np.array(std) 
    if mean is not None: 
        img = img + np.array(mean)
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)
    return img 

def img_display(img, name: str): 
    plt.figure("Image")
    plt.imshow(img)
    plt.axis('on')
    plt.title(name)
    plt.show()
    
def train_process_visualable(loss_log: list):
    plt.figure('train loss')
    

def get_logPath(log_dir: str): 
    num_of_exp = 0 
    for path in os.listdir(log_dir): 
        if os.path.isdir(os.path.join(log_dir, path)): 
            num_of_exp += 1
    return os.path.join(log_dir, "exp{:d}".format(num_of_exp))

