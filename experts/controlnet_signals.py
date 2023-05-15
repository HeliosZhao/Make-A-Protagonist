import sys
sys.path.insert(0, './')
import os
import cv2
import torch
from glob import glob

import argparse
from tqdm import tqdm
from PIL import Image

from controlnet_aux import MidasDetector, OpenposeDetector

torch.set_grad_enabled(False)

estimators = {
    'depth': MidasDetector, 
    'openpose': OpenposeDetector,
    'openposefull': OpenposeDetector,
}

def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data',
        type=str,
        help='dir for images: data/dir/images',
        default=None,
        required=True,
    )

    parser.add_argument(
        '-c',
        '--which_cond',
        type=str,
        required=True,
        help='which condition modality you want to test',
    )

    return parser



def main():
    parser = get_base_argument_parser()
    opt = parser.parse_args()

    which_cond = opt.which_cond


    outdir = opt.data.replace("images", which_cond) ## path of save
    os.makedirs(outdir, exist_ok=True)

    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image_paths = sorted(glob(os.path.join(opt.data, "*.jpg"))) + sorted(glob(os.path.join(opt.data, "*.png")))
    print("Processing video : {}, video length {}".format(opt.data, len(image_paths)))


    # prepare models
    cond_model = None
    cond_model = estimators[which_cond].from_pretrained("lllyasviel/Annotators")#.

    # inference
    for test_idx, cond_path in enumerate(tqdm(image_paths)):
        image = Image.open(cond_path).convert('RGB')
        fname = os.path.basename(cond_path).split('.')[0] # *.jpg
        width, height = image.size
        if which_cond == 'depth':
            new_w = width // 64 * 64
            new_h = height // 64 * 64
            image = image.resize((new_w, new_h))

        if which_cond == 'openposefull':
            cond = cond_model(image, hand_and_face=True)
        else:
            cond = cond_model(image)

        cond.resize((width, height))
        cond.save(os.path.join(outdir, f'{fname}.png'))


if __name__ == '__main__':
    main()
