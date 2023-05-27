# Demo Guidance

[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Make-A-Protagonist/Make-A-Protagonist-inference)

## Introduction
We build a hugging face demo to better illustrate the inference of Make-A-Protagonist. We provide four models trained on the source videos ([ikun](https://github.com/Make-A-Protagonist/Make-A-Protagonist/tree/main/data/ikun), [huaqiang](https://github.com/Make-A-Protagonist/Make-A-Protagonist/tree/main/data/huaqiang), [yanzi](https://github.com/Make-A-Protagonist/Make-A-Protagonist/tree/main/data/yanzi), and [car-turn](https://github.com/Make-A-Protagonist/Make-A-Protagonist/tree/main/data/car-turn)). For each source video, the protagonist and caption are also provided.

Users can choose a model, upload a reference image with corresponding protagonist prompt, and input a text prompt to generate video. Five examples are provided as reference.


## Parameters

Video generation parameters:
- `Video Length`: control the length of the generated video (it should be less than 6 due to the limited computational resources)
- `FPS`: control the smoothness of the generated video
- `Seed`: control the initial random noise in the denoising process

ControlNet parameters. Two ControlNet models are used:
- `Pose`: the weight of pose control information, only applicable for human
- `Depth`: the weight of depth control information

Editing function:
- `Source Protagonist`: using the source protagonist while changing the background
- `Source Background`: using the source background while changing the protagonist

Other parameters:
- `Number of Steps`: the number of denoising steps
- `Mask Starting Step`: the starting step to use mask-guided denoising sampling
- `CFG Scale`: classifier-free guidance scale
- `Noise Level`: the level of noise introduced to the reference image

## Acknowledgements
We thank AK and Hugging Face for providing computational resources on the demo.
