import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)
import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundedSAM.GroundingDINO.groundingdino.datasets.transforms as T
from GroundedSAM.GroundingDINO.groundingdino.models import build_model
from GroundedSAM.GroundingDINO.groundingdino.util import box_ops
from GroundedSAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundedSAM.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from GroundedSAM.segment_anything.segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import ipdb
import imageio
from tqdm import tqdm


'''
processing multiple images with grounded sam
only one text one time
'''

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases, logits_filt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("-d", "--data", type=str, required=True, help="path to image file")
    parser.add_argument("-t", "--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=False, help="output directory"
    )

    parser.add_argument("--config", type=str, 
    default="experts/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py", 
    help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="checkpoints/groundingdino_swinb_cogcoor.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="checkpoints/sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

    parser.add_argument("--masked_out", action='store_true', help="save the masked image")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    # image_path = args.data
    text_prompt = args.text_prompt
    output_dir = os.path.dirname(os.path.dirname(args.data))
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make dir
    text_prompt_dir = "-".join(text_prompt.split(" "))

    # text_prompt_dir
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "{}.viz".format(text_prompt_dir)), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "{}.mask".format(text_prompt_dir)), exist_ok=True)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    if os.path.isdir(args.data):
        images = sorted(glob(os.path.join(args.data,  "*.jpg"))) + sorted(glob(os.path.join(args.data,  "*.png")))
    else:
        images = [args.data]
    
    for image_path in tqdm(images):
        fname = os.path.basename(image_path).split('.')[0]
        # load image
        image_pil, image = load_image(image_path)

        # run grounding dino model
        boxes_filt, pred_phrases, logits_filt = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "{}.viz".format(text_prompt_dir), fname + ".jpg"), 
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        # ipdb.set_trace()
        max_logit_index = logits_filt.max(-1)[0].argmax().item()
        _mask = masks[max_logit_index,0].cpu().numpy().astype(np.uint8) * 255
        imageio.imwrite(os.path.join(output_dir, "{}.mask".format(text_prompt_dir), fname + ".png"), _mask)

        if args.masked_out:
            masked_image = np.asarray(image_pil).astype(np.float32) * _mask[:,:,None].astype(np.float32) / 255
            imageio.imwrite(os.path.join(output_dir, "masked_" + fname + ".png"), masked_image.astype(np.uint8))
        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

