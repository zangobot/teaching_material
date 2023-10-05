# ----- Standard Imports
import numpy as np
import random

# ----- Third Party Imports
import torch as tr
from torchvision import transforms
from torchvision.transforms import functional as F

# ----- Library Imports


def evaluate_accuracy(eval_model, data_loader, data_transform=None, target_transform=None):
    device = next(eval_model.parameters()).device
    corrects, num_samples = 0, 0
    for x,y in data_loader:
        x = x.to(device)
        y = y.to(device)

        if data_transform is not None:
            x = data_transform(x)
        if target_transform is not None:
            y = target_transform(y)

        pred = eval_model(x).argmax(dim=-1)
        corrects += (pred == y).sum().item()
        num_samples += len(x)
    return corrects/num_samples

def set_all_seed(seed):
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed), tr.Generator().manual_seed(seed)


class ConstantTargetTransform:
    def __init__(self, target):
        self._target = target
    
    def __call__(self, y):
        return tr.ones_like(y) * self._target


class MyCompose(transforms.Compose):
    def __call__(self, img, mask):
        for t in self.transforms:
            if isinstance(t, MyRandomAffine):
                img = t(img, mask)
            else:
                img = t(img)
        return img


class MyRandomAffine(transforms.RandomAffine):
    def forward(self, img, mask):
        fill = self.fill
        if isinstance(img, tr.Tensor):
            if isinstance(fill, (int, float)):
                try:
                    fill = [float(fill)] * F.get_image_num_channels(img)
                except:
                    fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        try:
            img_size = F.get_image_size(img)
        except:
            img_size = F._get_image_size(img)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        transf_img = F.affine(img, *ret, interpolation=self.interpolation, fill=fill)
        transf_mask = F.affine(mask, *ret, interpolation=self.interpolation, fill=fill)
        return transf_img, transf_mask
