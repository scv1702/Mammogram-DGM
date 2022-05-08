import os
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from math import floor

def open_dcm(img_path):
    ds = dicom.dcmread(img_path).pixel_array
    # Convert to grayscale
    image = (np.maximum(ds, 0) / ds.max()) * 255.0
    image = Image.fromarray(image)
    return image

def show_image(image):
    image = image[0].squeeze()
    plt.imshow(image, cmap="gray")
    plt.show()

def save_image(image, path):
    image = image.cpu().detach().numpy()
    image = image[0].squeeze()
    print(image.shape)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)
    
# return the number of AdaIN parameters needed by the model
def get_num_adain_params(model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def save_loss_graph(path, loss_list, title, label):
    plt.title(title)
    plt.plot(loss_list, label)
    plt.legend()
    plt.savefig(os.path.join(path, f"{title}.png"))

def conv2d_output_shape(height, kernel_size, stride, padding=0, dilation=1):
    return floor((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)