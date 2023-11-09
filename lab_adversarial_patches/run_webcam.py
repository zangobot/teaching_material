# ----- Standard Imports
from pathlib import Path

import numpy as np
from PIL import Image
import os
import zipfile

# ----- Third Party Imports
import cv2
import torch as tr
from torchvision import transforms
import gdown

# ----- Library Imports
from utils import MyRandomAffine
from torch_model_wrapper import TorchModelWrapper

def webcam_inference(model, numpy_patch):
    model.eval()
    apply_patch = False
    affine_transf = MyRandomAffine(degrees=45, translate=(.3,.3), scale=(.5,1))
    transform = transforms.Compose([
        transforms.Resize(numpy_patch.shape[0]),
        transforms.CenterCrop(numpy_patch.shape[0]),
        transforms.ToTensor(),
        ])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    patch = tr.tensor(numpy_patch.transpose(2,0,1))
    mask = tr.ones_like(patch)
    mask[patch == 0] = 0

    classes_dict = {0:{'name':'book','color':(215,25,28)},
                    1:{'name':'cellphone','color':(253,174,97)},
                    2:{'name':'mouse','color':(255,255,191)},
                    3:{'name':'pencilcase','color':(171,217,233)},
                    4:{'name':'ringbinder','color':(44,123,182)}}

    video_capture = cv2.VideoCapture(0)
    while True:
        frame = video_capture.read()[1]
        tens_frame = transform(Image.fromarray(frame))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('p'):
            apply_patch = not apply_patch
            if apply_patch:
                transf_patch, transf_mask = affine_transf(patch, mask)
                inv_transf_mask = tr.zeros_like(transf_mask)
                inv_transf_mask[transf_mask == 0] = 1

        if apply_patch:
            tens_frame = tens_frame*inv_transf_mask + transf_patch*transf_mask

        with tr.no_grad():
            pred = model(normalize(tens_frame).unsqueeze(0)).argmax(dim=-1).item()

        frame = cv2.UMat((tens_frame.numpy().transpose(1,2,0)*255).astype(np.uint8))
        cv2.putText(frame,
            classes_dict[pred]['name'],
            (10, 20), # origin
            cv2.FONT_HERSHEY_SIMPLEX, # FONT_HERSHEY_PLAIN, FONT_HERSHEY_SIMPLEX
            1.1, # scale
            classes_dict[pred]['color'],
            2, # thickness
            cv2.LINE_AA,
            )
        cv2.imshow("Video", frame)

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # download data.zip which contains models weights and adversarial patches
    data_path = Path(__file__).parent
    if not (data_path / 'data.zip').exists():
        gdown.download(id='1S5l8Bn_oTckD5iiMSX82m9pxJp0762Qc')
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall('')

    # load trained model
    model_wrp = TorchModelWrapper(model_name='alexnet', num_classes=5) # HARD-CODED
    model_wrp.load_model(os.path.join('data','models'))
    model_wrp.set_model_gradients(False)

    # load patch
    name = ['book_91','cellphone_47','mouse_96','pencilcase_57','ringbinder_87'][4]
    np_patch = np.load(os.path.join('data','patches',f"{name}_script.npy"))
    webcam_inference(model_wrp.model, np_patch)


if __name__ == '__main__':
    main()