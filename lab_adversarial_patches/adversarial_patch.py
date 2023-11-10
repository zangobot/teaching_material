# ----- Standard Imports
import numpy as np
from tqdm import tqdm

# ----- Third Party Imports
import torch as tr
from torchvision.transforms import Compose

# ----- Library Imports
from utils import MyRandomAffine, ConstantTargetTransform, evaluate_accuracy


class AdversarialPatch:
    def __init__(self,
                 input_shape,
                 patch_type: str = "square", # "square" or "circle"
                 patch_position: str = 'centered', #'centered' or 'top_left'
                 patch_size: int = 50,
                 preprocess:list = [],
                 optimize_location: bool = True,
                 rotation_range: int = 45,
                 scale_range: tuple = (0.5, 1),
                 ):
        self.mask = self._generate_mask(input_shape, patch_type, patch_position, patch_size)
        self.patch = tr.rand(input_shape)
        self.patch.requires_grad = True
        self.patch.data[self.mask == 0] = 0

        self._preprocess = Compose(preprocess)
        self._transforms = MyRandomAffine(rotation_range,
                                          (0.3, 0.3) if optimize_location else None,
                                          scale_range)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def transforms(self):
        return self._transforms

    def train(self,
              model,
              train_loader,
              test_loader,
              target_class: int,
              lr: float = 1.,
              num_epochs: int = 100, 
              device_idx: int = -1,
              verbose: bool = False,
              ):

        device = tr.device(f'cuda:{device_idx}' if tr.cuda.is_available() and device_idx != -1 else 'cpu')
        self.mask = self.mask.to(device)
        self.patch = self.patch.to(device)
        model.eval()
        model = model.to(device)

        #ce = tr.nn.CrossEntropyLoss(reduction='sum')
        ce = tr.nn.CrossEntropyLoss(reduction='mean')

        if verbose:
            with tr.no_grad():
                print("Evaluation before training:")
                tst_clean = evaluate_accuracy(model, test_loader, data_transform=self._preprocess)
                print(f"\tAccuracy on clean test set: {tst_clean:.2f}")
                tst_adv = evaluate_accuracy(model, test_loader, data_transform=self.apply_patch)
                print(f"\tAccuracy on adversarial test set: {tst_adv:.2f}")
                adv_succ = evaluate_accuracy(model, test_loader, data_transform=self.apply_patch,
                                             target_transform=ConstantTargetTransform(target_class))
                print(f"\tAdversarial patch success rate: {adv_succ:.2f}\n")

        for ep in range(1, num_epochs+1):
            cum_loss = 0
            batch = tqdm(train_loader) if verbose else train_loader
            for b_idx, (x, y) in enumerate(batch):
                x = x.to(device)
                y = (tr.ones_like(y) * target_class).to(device)
 
                # the for loop is required in order to apply a different random patch transformation to each sample
                for i in range(len(x)):
                    x[i] = self.apply_patch(x[i])
                # x = self.apply_patch(x)

                # compute predictions, average loss, and gradients
                out = model(x)
                loss = ce(out, y)
                grad = tr.autograd.grad(loss, self.patch)[0]

                with tr.no_grad():
                    # no need to average the gradients because the loss is already averaged across the batch
                    #grad /= len(x)

                    # adjust learning rate and lower-bound it
                    step = lr / ep
                    if step < 1e-4: step = 1e-4

                    # update patch with gradient step
                    self.patch = self.patch - step * grad
                    #self.patch = self.patch - lr * grad

                    # clip patch to be in the image range
                    self.patch = tr.clamp(self.patch, min=0, max=1)
                    
                # patch assignments within tr.no_grad() scope set its requires_grad = False
                self.patch.requires_grad = True
                
                if verbose:
                    cum_loss += loss.item()
                    batch.set_description(f"@epoch {ep}/{num_epochs} average train loss: {cum_loss/(b_idx+1):.3f}")
            
            if verbose:
                with tr.no_grad():
                    tst_clean = evaluate_accuracy(model, test_loader, data_transform=self._preprocess)
                    print(f"\tAccuracy on clean test set: {tst_clean:.2f}")
                    tst_adv = evaluate_accuracy(model, test_loader, data_transform=self.apply_patch)
                    print(f"\tAccuracy on adversarial test set: {tst_adv:.2f}")
                    adv_succ = evaluate_accuracy(model, test_loader, data_transform=self.apply_patch,
                                                target_transform=ConstantTargetTransform(target_class))
                    print(f"\tAdversarial patch success rate: {adv_succ:.2f}\n")

        return self.patch.detach().cpu(), adv_succ

    @staticmethod
    def _generate_mask(input_shape, patch_type, patch_position, patch_size):
        if patch_type == "square":
            mask = tr.ones(input_shape)
            if patch_position == 'centered':   # centered
                upp_l_x = input_shape[2] // 2 - patch_size // 2
                upp_l_y = input_shape[1] // 2 - patch_size // 2
                bott_r_x = upp_l_x + patch_size
                bott_r_y = upp_l_y + patch_size
                mask[:, :upp_l_x, :] = 0
                mask[:, :, :upp_l_y] = 0
                mask[:, bott_r_x:, :] = 0
                mask[:, :, bott_r_y:] = 0
            elif patch_position == 'top_left': # top left
                mask[:, patch_size:, :] = 0
                mask[:, :, patch_size:] = 0
        elif patch_type == "circle":
            mask = tr.zeros(input_shape)
            center_x = input_shape[2] // 2
            center_y = input_shape[1] // 2
            radius = patch_size // 2
            y, x = np.ogrid[:input_shape[1], :input_shape[2]]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            circle = int(dist_from_center <= radius)
            mask[:, :, :] = tr.from_numpy(circle)
        else:
            raise ValueError("The patch type has to be either `circle` or `square`.")
        return mask

    def apply_patch(self, img):
        patch, mask = self.transforms(self.patch, self._mask)
        inv_mask = tr.zeros_like(mask)
        inv_mask[mask == 0] = 1
        return self._preprocess(img*inv_mask + patch*mask)
