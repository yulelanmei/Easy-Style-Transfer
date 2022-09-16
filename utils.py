import os
import numpy as np
import matplotlib.pyplot as plt

def img_transfer(tensor, std= (0.229, 0.224, 0.225), mean= (0.485, 0.456, 0.406)):
    img = tensor.cpu().detach().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)
    return img 

def img_display(img, name: str): 
    plt.figure("Image")
    plt.imshow(img)
    plt.axis('on')
    plt.title(name)
    plt.show()

def get_logPath(log_dir: str): 
    num_of_exp = 0 
    for path in os.listdir(log_dir): 
        if os.path.isdir(os.path.join(log_dir, path)): 
            num_of_exp += 1
    return os.path.join(log_dir, "exp{:d}".format(num_of_exp))

