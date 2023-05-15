
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from einops import rearrange
import os
import os.path as osp
from glob import glob
import imageio
import cv2
import numpy as np
import random
import ipdb

class MakeAProtagonistDataset(Dataset):
    def __init__(
            self,
            video_dir: str,
            prompt: str,
            condition: list[str] = 'openpose', ## type of condition used
            video_suffix: str = '.jpg',
            condition_suffix: str = '.png',
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            random_sample: bool = False,
            mask_dir: str = None, 
            **kwargs,
    ):
        self.video_dir = video_dir ## path to the video dir
        self.video_path = osp.join(self.video_dir, 'images')

        self.condition = condition
        if isinstance(condition, str):
            condition = [condition]
        self.condition_path = {_condition: osp.join(self.video_dir, _condition) for _condition in condition}
        self.video_suffix = video_suffix
        self.condition_suffix = condition_suffix
        self.random_sample = random_sample
        self.mask_dir = mask_dir
        if mask_dir:
            self.mask_dir = osp.join(self.video_dir, mask_dir)

        ## get frame path
        frame_list_path = osp.join(self.video_dir, 'frame_list.txt')
        if not osp.isfile(frame_list_path):
            all_frames = sorted(glob(osp.join(self.video_path, '*')))
            self.frame_list = []
            with open(frame_list_path, 'w') as f:
                for _frame_path in all_frames:
                    _frame_name = osp.basename(_frame_path).split('.')[0]
                    self.frame_list.append(_frame_name)
                    f.write(_frame_name + '\n')
        
        else:
            with open(frame_list_path, 'r') as f:
                self.frame_list = f.read().splitlines()
        
        self.video_length = len(self.frame_list)

        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.img_embeddings = []

        print('Training on Video {} \t totally {} frames'.format(self.video_dir.split('/')[-1], self.video_length))
    
    @torch.no_grad()
    def preprocess_img_embedding(self, feature_extractor, image_encoder):
        for f_name in self.frame_list:
            image = imageio.imread(osp.join(self.video_path, f_name + self.video_suffix))
            image = feature_extractor(images=image, return_tensors="pt").pixel_values
            image_embeds = image_encoder(image).image_embeds
            self.img_embeddings.append(image_embeds[0]) # 1,768 --> 768


    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        video_indices = list(range(self.sample_start_idx, self.video_length, self.sample_frame_rate))
        video = []
        conditions = {_condition: [] for _condition in self.condition}

        mask = []
        if self.random_sample:
            start_index = random.randint(0,len(video_indices) - self.n_sample_frames) ## [a,b] include both
        else:
            start_index = 0
        sample_index = video_indices[start_index:start_index+self.n_sample_frames]

        for _f_idx in sample_index:
            _frame = imageio.imread(osp.join(self.video_path, self.frame_list[_f_idx] + self.video_suffix))
            if self.mask_dir:
                _mask = imageio.imread(osp.join(self.mask_dir, self.frame_list[_f_idx] + '.png')).astype(np.float32) ## H,W 0 and 255
                _mask /= 255 # 0 and 1
            else:
                _mask = np.ones(_frame.shape[:2])
            video.append(_frame)
            mask.append(_mask)

            for _control_type, _control_path in self.condition_path.items():
                _condition = imageio.imread(osp.join(_control_path, self.frame_list[_f_idx] + self.condition_suffix)) ## 
                conditions[_control_type].append(_condition)
        
        ref_idx = random.choice(sample_index) # idx random sample one ref image index from the select video clip

        video = torch.from_numpy(np.stack(video, axis=0)).float() # f,h,w,c
        
        video = rearrange(video, "f h w c -> f c h w")
        video = F.interpolate(video, size=(self.height, self.width), mode='bilinear')

        # ipdb.set_trace()
        conditions_transform = {}
        for _control_type, condition in conditions.items():
            condition = torch.from_numpy(np.stack(condition, axis=0)).float() # f,h,w,c
            condition = rearrange(condition, "f h w c -> f c h w")
            condition = F.interpolate(condition, size=(self.height, self.width), mode='bilinear')
            conditions_transform[_control_type] = condition / 255

        mask = torch.from_numpy(np.stack(mask, axis=0)).float() # f,h,w
        mask = rearrange(mask[:,:,:,None], "f h w c -> f c h w")
        mask = F.interpolate(mask, size=(self.height, self.width), mode='nearest')

        ref_img = imageio.imread(osp.join(self.video_path, self.frame_list[ref_idx] + self.video_suffix)) # read ref image
        ref_img = torch.from_numpy(ref_img).float() # h,w,c convert to tensor
        ref_img = ref_img.permute(2,0,1).unsqueeze(0).repeat(self.n_sample_frames,1,1,1)  ## h,w,c -> c,h,w -> 1,c,h,w -> f,c,h,w
        ref_img = F.interpolate(ref_img, size=(self.height, self.width), mode='bilinear')

        ref_condition = torch.zeros_like(ref_img)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "conditions": conditions_transform,
            "prompt_ids": self.prompt_ids,
            "ref_img": (ref_img / 127.5 - 1.0),
            "ref_condition": ref_condition / 255,
            "masks": mask, 
            "sample_indices": torch.LongTensor(sample_index),  

        }

        ref_imbed = None
        if len(self.img_embeddings):
            ref_imbed = self.img_embeddings[ref_idx]
            example["ref_imbed"] = ref_imbed


        return example


