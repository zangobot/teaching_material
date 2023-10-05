# ----- Standard Imports
import matplotlib.pyplot as plt
import os
import numpy as np

# ----- Third Party Imports
from torchvision.transforms import Normalize
from torchvision.utils import make_grid

# ----- Library Imports


class InvNormalize(Normalize):
    def __init__(self, normalizer):
        inv_mean = [-mean / std for mean, std in list(zip(normalizer.mean, normalizer.std))]
        inv_std = [1 / std for std in normalizer.std]
        super().__init__(inv_mean, inv_std)

def _tensor_to_show(img, transforms=None):
    if transforms is not None:
        for transform in transforms.transforms:
            if isinstance(transform, Normalize):
                normalizer = transform
                break
        inverse_transform = InvNormalize(normalizer)
        img = inverse_transform(img)

    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

def imshow(img, transforms=None, figsize=(10, 20)):
    npimg = _tensor_to_show(img, transforms)
    plt.figure(figsize=figsize)
    plt.imshow(npimg, interpolation=None)

def show_batch(x, transforms=None, figsize=(10, 20)):
    imshow(make_grid(x.cpu().detach(), nrow=5),
           transforms=transforms, figsize=figsize)
    plt.axis('off')
    plt.show()

def show_image(img):
    plt.imshow(img.permute(1, 2, 0).detach().cpu())
    plt.show()

def show_patches(name_list):
    fig, axs = plt.subplots(1,5, figsize=(10,15))
    for i, fn in enumerate(name_list):
        trn_patch = np.load(os.path.join("data",'patches',f"{fn}_script.npy"))
        axs[i].set_title(f"{fn.replace('_', ' ')}%", size=15)
        axs[i].imshow(trn_patch)
        axs[i].axis('off')
    fig.tight_layout()
