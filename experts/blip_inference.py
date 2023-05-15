from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch
import os
from glob import glob
import argparse
from glob import glob

from BLIP2.blip_video_model import Blip2ForVideoConditionalGeneration as Blip2ForConditionalGeneration

from termcolor import colored, cprint

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_root", type=str, required=True)
parser.add_argument("-fn" , "--frame_num", type=int, default=8)
parser.add_argument("-fps" , "--frame_rate", type=int, default=1)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

# Salesforce/blip2-flan-t5-xxl
# Salesforce/blip2-opt-6.7b
blip2_version = "Salesforce/blip2-flan-t5-xl"
# blip2_version = "Salesforce/blip2-opt-6.7b"

weight_dtype = torch.bfloat16 if "flan" in blip2_version else torch.float16
# weight_dtype = torch.float16

processor = Blip2Processor.from_pretrained(blip2_version)
model = Blip2ForConditionalGeneration.from_pretrained(
    blip2_version, torch_dtype=weight_dtype
)
model.to(device)


if not os.path.isdir(args.data_root):
    image_list = [args.data_root]
else:
    # ipdb.set_trace()
    all_image_list = sorted(glob(os.path.join(args.data_root,  "*.jpg"))) + sorted(glob(os.path.join(args.data_root,  "*.png")))
    image_list = [all_image_list[f] for f in range(0, args.frame_num*args.frame_rate, args.frame_rate)]
    assert len(image_list) == args.frame_num


images = []
for image_path in image_list:
    image = Image.open(image_path).convert("RGB")
    images.append(image)

def blip2_call(prompt=None, max_new_tokens=20):
    inputs = processor(images, text=prompt, return_tensors="pt").to(device, weight_dtype)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if prompt is not None:
        cprint(prompt, "red")
    else:
        cprint("No prompt", "red")

    print(generated_text)


## prompt captioning
prompt = "this is a video of"

print("Captioning")
blip2_call(prompt, 20)


prompt = "Question: what is the protagonist in this video? Answer: "

blip2_call(prompt, 10)


