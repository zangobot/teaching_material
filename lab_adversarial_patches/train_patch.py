# ----- Standard Imports
import os
import numpy as np
import argparse

# ----- Third Party Imports
import torch as tr
from torchvision import transforms, datasets

# ----- Library Imports
from utils import set_all_seed
from adversarial_patch import AdversarialPatch
from torch_model_wrapper import TorchModelWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_class',
                        type = str,
                        choices = ['book','cellphone','mouse','pencilcase','ringbinder'],
                        help = 'Target class for the adversarial attack')
    args = parser.parse_args()

    target_class = [args.target_class]
    if target_class[0] is None:
        target_class = ['book','cellphone','mouse','pencilcase','ringbinder']
    for t_c in target_class:
        print(f"\n\n{t_c}\n\n")
        
        np_gen, tr_gen = set_all_seed(42)

        # ----- initialize train, validation, test data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        data = datasets.ImageFolder(os.path.join('data','dataset'),
                                    transform = transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        #normalize,
                                    ]))
        num_classes = len(data.classes)
        num_samples = 10000 # HARD-CODED samples number
        subset_idxs = np_gen.choice(len(data), size=num_samples, replace=False)
        # use val data for patch training and exclude target class
        val_idxs = [i for i in subset_idxs[int(num_samples*.4):int(num_samples*.7)] if data[i][1] != data.class_to_idx[t_c]]
        val_split = tr.utils.data.Subset(data, val_idxs)
        tst_idxs = subset_idxs[int(num_samples*.7):]
        tst_split = tr.utils.data.Subset(data, tst_idxs)

        val_loader = tr.utils.data.DataLoader(val_split, batch_size=50, shuffle=True, generator=tr_gen)
        tst_loader = tr.utils.data.DataLoader(tst_split, batch_size=50, shuffle=True, generator=tr_gen)

        # ----- load trained model
        model_wrp = TorchModelWrapper(model_name='alexnet', num_classes=num_classes)
        model_wrp.load_model(os.path.join('data','models'))
        model_wrp.set_model_gradients(False)

        # ----- initialize adversarial patch with active gradients
        patch = AdversarialPatch(input_shape = val_split[0][0].shape,
                                 patch_type = "square", # "square" or "circle"
                                 patch_position = 'centered', #'centered' or 'top_left'
                                 patch_size = 50,
                                 preprocess = [normalize],
                                 optimize_location = True,
                                 rotation_range = 45,
                                 scale_range = (0.5, 1),
                                 )

        adv_patch, succ_rate = patch.train(model_wrp.model, val_loader, tst_loader,
                                           target_class = data.class_to_idx[t_c],
                                           lr = 5.,
                                           num_epochs = 200,
                                           device_idx = 1,
                                           verbose = True)

        # store patch
        adv_patch = adv_patch.numpy().transpose(1,2,0)
        adv_patch = (adv_patch*255).astype(np.uint8)
        np.save(os.path.join('data','patches',f'{t_c}_{int(100*succ_rate)}_script.npy'), adv_patch)
    

if __name__ == '__main__':
    main()