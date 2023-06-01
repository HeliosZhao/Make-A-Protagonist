# Make-A-Protagonist

This repository is the official implementation of **Make-A-Protagonist**.

**[Make-A-Protagonist: Generic Video Editing with An Ensemble of Experts](https://arxiv.org/abs/2305.08850)**
<br/>
[Yuyang Zhao](https://yuyangzhao.com), [Enze Xie](https://xieenze.github.io/), [Lanqing Hong](https://scholar.google.com.sg/citations?user=2p7x6OUAAAAJ&hl=en), [Zhenguo Li](https://scholar.google.com.sg/citations?user=XboZC1AAAAAJ&hl=en), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/)
<br/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://make-a-protagonist.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2305.08850-b31b1b.svg)](https://arxiv.org/abs/2305.08850)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Make-A-Protagonist/Make-A-Protagonist-inference)

<p align="center">
<img src="./assets/teaser-video-small.gif" width="1080px"/>  
<br>
<em>The first framework for generic video editing with both visual and textual clues.</em>
</p>


## Abstract
> The text-driven image and video diffusion models have achieved unprecedented success in generating realistic and diverse content. Recently, the editing and variation of existing images and videos in diffusion-based generative models have garnered significant attention. However, previous works are limited to editing content with text or providing coarse personalization using a single visual clue, rendering them unsuitable for indescribable content that requires fine-grained and detailed control. In this regard, we propose a generic video editing framework called Make-A-Protagonist, which utilizes textual and visual clues to edit videos with the goal of empowering individuals to become the protagonists. Specifically, we leverage multiple experts to parse source video, target visual and textual clues, and propose a visual-textual-based video generation model that employs mask-guided denoising sampling to generate the desired output. Extensive results demonstrate the versatile and remarkable editing capabilities of Make-A-Protagonist.

## News
- [16/05/2023] Code released!
- [26/05/2023] [Hugging Face Demo](https://huggingface.co/spaces/Make-A-Protagonist/Make-A-Protagonist-inference) released!
- [01/06/2023] Upload more video demos with 32 frames in this repo and [Project Page](https://make-a-protagonist.github.io/)! Happy Children's Day!

### Todo
- [ ] Release training code for ControlNet UnCLIP Small
- [x] Release inference demo


## Setup

### Requirements
- Python>=3.9 and Pytorch>=1.13.1
- xformers 0.0.17
- Other packages in `requirements.txt`
- Build GroundedSAM expert
```bash
cd experts/GroundedSAM
pip install -e GroundingDINO
pip install -e segment_anything
```

### Weights

The following weights from HuggingFace are used in this project. You can download them into `checkpoints` or load them from HuggingFace repo.
- [Stable Diffusion UnCLIP Small](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip-small)
- [BLIP-2 Flan T5-xL](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
- [CLIP ViT-L](https://huggingface.co/openai/clip-vit-large-patch14)
- [DALL-E 2 Prior](https://huggingface.co/kakaobrain/karlo-v1-alpha)

ControlNet for Stable Diffusion UnCLIP Small should be downloaded manually into `checkpoints`:
- [ControlNet UnCLIP Small](https://huggingface.co/Make-A-Protagonist/controlnet-2-1-unclip-small)

The code for training these models will be released soon.

Pre-trained model for other experts should be downloaded manually into `checkpoints`:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) `wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth`
- [Segment Anything](https://github.com/facebookresearch/segment-anything) `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`
- [XMem](https://github.com/hkchengrex/XMem) `wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth`



## Usage

### Data Preprocess

#### Source Video Parsing

**Captioning and VQA**:
```bash
python experts/blip_inference.py -d data/<video_name>/images
```

**Protagonist Segmentation**:

- Frame segmentation with GroundedSAM
```bash
python experts/grounded_sam_inference.py -d data/<video_name>/images/0000.jpg -t <protagonist>
```
Note: Since GroundingDINO detects bounding boxes for each noun in the sentence, it is better to use only one noun here. For example, use `-t man` instead of `-t "a man with a basketball"`.

- Video object segmentation through the video
```bash
python experts/xmem_inference.py -d data/<video_name>/images -v <video_name> --mask_dir <protagonist>.mask
```

**Control Signals Extraction**:
```bash
python experts/controlnet_signal_extraction.py -d data/<video_name>/images -c <control>
```
Currently we only support two types of control signals: depth and openposefull.

#### Visual Clue Parsing

**Reference Protagonist Segmentation**:
```bash
python experts/grounded_sam_inference.py -d data/<video_name>/reference_images/<reference_image_name> -t <protagonist> --masked_out
```

### Training

To fine-tune the text-to-image diffusion models with visual and textual clues, run this command:

```bash
python train.py --config="configs/<video_name>/train.yaml"
```

Note: At least 24 GB is requires to train the model.

### Inference

Once the training is done, run inference:

```bash
python eval.py --config="configs/<video_name>/eval.yaml"
```
**Applications**: Three applications are supported by Make-A-Protagonist, which can be achieved by modifying the inference configuration file.
- Protagonist Editing: `source_protagonist: true`
- Background Editing: `source_background: true`
- Text-to-Video Editing with Protagonist: `source_protagonist: false & source_background: false`

## Results

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Reference Image</b></td>
  <td style="text-align:center;"><b>Generated Video</b></td>
</tr>
<tr>
  <td><img src="https://make-a-protagonist.github.io/assets/data/yanzi.gif"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/reference/yanzi/panda.jpeg"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/results/yanzi/panda.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man walking down the street"</td>
  <td width=25% style="text-align:center;"></td>
  <td width=25% style="text-align:center;">"A <span style="color: darkred">panda</span> walking down <span style="color: steelblue">the snowy street</span>"</td>
</tr>

<tr>
  <td><img src="https://make-a-protagonist.github.io/assets/data/ikun.gif"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/reference/ikun/zhongli.jpg"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/results/ikun/ikun-beach.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man playing basketball"</td>
  <td width=25% style="text-align:center;"></td>
  <td width=25% style="text-align:center;">"A <span style="color: darkred">man</span> playing basketball <span style="color: steelblue">on the beach, anime style</span>"</td>
</tr>

<tr>
  <td><img src="https://make-a-protagonist.github.io/assets/data/huaqiang.gif"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/reference/huaqiang/musk.jpg"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/results/huaqiang/huaqiang-musk.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man walking down the street"</td>
  <td width=25% style="text-align:center;"></td>
  <td width=25% style="text-align:center;">"<span style="color: darkred">Elon Musk</span> walking down the street"</td>
</tr>

<tr>
  <td><img src="https://make-a-protagonist.github.io/assets/data/car-turn.gif"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/reference/car-turn/0000.jpg"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/results/car-turn/sj-rain.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A Suzuki Jimny driving down a mountain road"</td>
  <td width=25% style="text-align:center;"></td>
  <td width=25% style="text-align:center;">"A <span style="color: darkred">Suzuki Jimny</span> driving down a mountain road <span style="color: steelblue">in the rain</span>"</td>
</tr>

<tr>
  <td><img src="https://make-a-protagonist.github.io/assets/data/girl-dance.gif"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/reference/girl-dance/beel.jpeg"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/results/girl-dance/girl-dance-beel.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A girl in white dress dancing on a bridge"</td>
  <td width=25% style="text-align:center;"></td>
  <td width=25% style="text-align:center;">"A <span style="color: darkred">girl</span> dancing <span style="color: steelblue">on the beach, anime style</span>"</td>
</tr>

<tr>
  <td><img src="https://make-a-protagonist.github.io/assets/data/man-dance.gif"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/reference/man-dance/peng.png"></td>
  <td><img src="https://make-a-protagonist.github.io/assets/results/man-dance/man-dance-peng.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man dancing in a room"</td>
  <td width=25% style="text-align:center;"></td>
  <td width=25% style="text-align:center;">"A <span style="color: darkred">man</span>  <span style="color: steelblue">in dark blue suit with white shirt</span> dancing <span style="color: steelblue">on the beach</span>"</td>
</tr>

</table>


## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhao2023makeaprotagonist,
    title={Make-A-Protagonist: Generic Video Editing with An Ensemble of Experts},
    author={Zhao, Yuyang and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Lee, Gim Hee},
    journal={arXiv preprint arXiv:2305.08850},
    year={2023}
}
```

## Acknowledgements

- This code is heavily derived from [diffusers](https://github.com/huggingface/diffusers) and [Tune-A-Video](https://github.com/showlab/Tune-A-Video). If you use this code in your research, please also acknowledge their work.
- This project leverages [Stable Diffusion UnCLIP](https://github.com/Stability-AI/stablediffusion/blob/main/doc/UNCLIP.MD), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [CLIP](https://github.com/openai/CLIP), [DALL-E 2 from Karlo](https://github.com/kakaobrain/karlo), [ControlNet](https://github.com/lllyasviel/ControlNet), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem). We thank them for open-sourcing the code and pre-trained models.
- We thank AK and Hugging Face for providing computational resources on the demo [![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Make-A-Protagonist/Make-A-Protagonist-inference).

