## ___***DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos***___
<div align="center">
<img src='https://depthcrafter.github.io/img/logo.png' style="height:140px"></img>



 <a href='https://arxiv.org/abs/2409.02095'><img src='https://img.shields.io/badge/arXiv-2409.02095-b31b1b.svg'></a> &nbsp;
 <a href='https://depthcrafter.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/spaces/tencent/DepthCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;


_**[Wenbo Hu<sup>1* &dagger;</sup>](https://wbhu.github.io), 
[Xiangjun Gao<sup>2*</sup>](https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en), 
[Xiaoyu Li<sup>1* &dagger;</sup>](https://xiaoyu258.github.io), 
[Sijie Zhao<sup>1</sup>](https://scholar.google.com/citations?user=tZ3dS3MAAAAJ&hl=en), 
[Xiaodong Cun<sup>1</sup>](https://vinthony.github.io/academic), <br>
[Yong Zhang<sup>1</sup>](https://yzhang2016.github.io), 
[Long Quan<sup>2</sup>](https://home.cse.ust.hk/~quan), 
[Ying Shan<sup>3, 1</sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)**_
<br><br>
<sup>1</sup>Tencent AI Lab
<sup>2</sup>The Hong Kong University of Science and Technology
<sup>3</sup>ARC Lab, Tencent PCG

arXiv preprint, 2024

</div>

## 🔆 Introduction

- `[24-10-19]` 🤗🤗🤗 DepthCrafter now has been integrated into [ComfyUI](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes)!
- `[24-10-08]` 🤗🤗🤗 DepthCrafter now has been integrated into [Nuke](https://github.com/Theo-SAMINADIN-td/NukeDepthCrafter), have a try!
- `[24-09-28]` Add full dataset inference and evaluation scripts for better comparison use. :-)
- `[24-09-25]` 🤗🤗🤗 Add huggingface online demo [DepthCrafter](https://huggingface.co/spaces/tencent/DepthCrafter). 
- `[24-09-19]` Add scripts for preparing benchmark datasets. 
- `[24-09-18]` Add point cloud sequence visualization.
- `[24-09-14]` 🔥🔥🔥 **DepthCrafter** is released now, have fun!


🔥 DepthCrafter can generate temporally consistent long-depth sequences with fine-grained details for open-world videos, 
without requiring additional information such as camera poses or optical flow.

🤗 If you find DepthCrafter useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

## 🎥 Visualization
We provide demos of unprojected point cloud sequences, with reference RGB and estimated depth videos. 
Please refer to our [project page](https://depthcrafter.github.io) for more details.


https://github.com/user-attachments/assets/62141cc8-04d0-458f-9558-fe50bc04cc21




## 🚀 Quick Start

### 🤖 Gradio Demo
- Online demo: [DepthCrafter](https://huggingface.co/spaces/tencent/DepthCrafter) 
- Local demo:
    ```bash
    gradio app.py
    ``` 

### 🌟 Community Support
- [NukeDepthCrafter](https://github.com/Theo-SAMINADIN-td/NukeDepthCrafter): 
    a plugin allows you to generate temporally consistent Depth sequences inside Nuke, 
    which is widely used in the VFX industry.
- [ComfyUI-Nodes](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes): creating consistent depth maps for your videos using DepthCrafter in ComfyUI.


### 🛠️ Installation
1. Clone this repo:
```bash
git clone https://github.com/Tencent/DepthCrafter.git
```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```



### 🤗 Model Zoo
[DepthCrafter](https://huggingface.co/tencent/DepthCrafter) is available in the Hugging Face Model Hub.

### 🏃‍♂️ Inference
#### 1. High-resolution inference, requires a GPU with ~26GB memory for 1024x576 resolution:
- Full inference (~0.6 fps on A100, recommended for high-quality results):

    ```bash
    python run.py  --video-path examples/example_01.mp4
    ```


- Fast inference through 4-step denoising and without classifier-free guidance （~2.3 fps on A100）:

    ```bash
    python run.py  --video-path examples/example_01.mp4 --num-inference-steps 4 --guidance-scale 1.0
    ```


#### 2. Low-resolution inference requires a GPU with ~9GB memory for 512x256 resolution:

- Full inference (~2.3 fps on A100):

    ```bash
    python run.py  --video-path examples/example_01.mp4 --max-res 512
    ```

- Fast inference through 4-step denoising and without classifier-free guidance (~9.4 fps on A100):
    ```bash
    python run.py  --video-path examples/example_01.mp4  --max-res 512 --num-inference-steps 4 --guidance-scale 1.0
    ```

## 🚀 Dataset Evaluation
Please check the `benchmark` folder. 
- To create the dataset we use in the paper, you need to run `dataset_extract/dataset_extract_${dataset_name}.py`.
- Then you will get the `csv` files that save the relative root of extracted RGB video and depth npz files. We also provide these csv files.
- Inference for all datasets scripts:
  ```bash
  bash benchmark/infer/infer.sh
  ```
  (Remember to replace the `input_rgb_root` and `saved_root` with your own path.)
- Evaluation for all datasets scripts:
  ```bash
  bash benchmark/eval/eval.sh
  ```
   (Remember to replace the `pred_disp_root` and `gt_disp_root` with your own path.)
####

## 🤝 Contributing
- Welcome to open issues and pull requests.
- Welcome to optimize the inference speed and memory usage, e.g., through model quantization, distillation, or other acceleration techniques.

## 📜 Citation
If you find this work helpful, please consider citing:
```bibtex
@article{hu2024-DepthCrafter,
            author      = {Hu, Wenbo and Gao, Xiangjun and Li, Xiaoyu and Zhao, Sijie and Cun, Xiaodong and Zhang, Yong and Quan, Long and Shan, Ying},
            title       = {DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos},
            journal     = {arXiv preprint arXiv:2409.02095},
            year        = {2024}
    }
```
