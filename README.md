<div align="center">

# AutoGaze

[![Website](https://img.shields.io/badge/Website-76b900?style=for-the-badge&logo=safari&labelColor=555555)](https://autogaze.github.io/)
[![Arxiv](https://img.shields.io/badge/Arxiv-b31b1b?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/abs/2603.12254)
[![Models & Data & Benchmark](https://img.shields.io/badge/Models%20%26%20Data%20%26%20Benchmark-ffd21e?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/collections/bfshi/autogaze)
[![Demo](https://img.shields.io/badge/Demo-ff6e00?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/spaces/bfshi/AutoGaze)

</div>

AutoGaze (Autoregressive Gazing) is a model that automatically selects informative patches and remove redundant ones in any video, such that downstream ViTs/MLLMs can process fewer patches without informaiton loss. This makes downstream ViTs/MLLMs much more scalable to high-resolution, high-FPS, long-form videos (e.g., 4K-resolution 1K-frame videos).


## 📷 Demo

See the video below for a quick peek of what AutoGaze is capable of! Meanwhile you can also try out the [demo](https://huggingface.co/spaces/bfshi/AutoGaze) on your own video!

https://github.com/user-attachments/assets/ffc4cc72-e519-4ebe-aaa0-8d3ac3094160

## 🎉 Announcement

[2026.4.8] AutoGaze was selected as conference highlight at CVPR2026!
[2026.2.20] AutoGaze got accepted to CVPR2026!

## 📦 Installation

```bash
# Create conda environment
conda create -n autogaze python=3.11
conda activate autogaze

# Install CUDA toolkit
# Note: If you've already installed PyTorch, change the cuda version here to the one your PyTorch was built on!
conda install -c nvidia cuda-toolkit=12.8

# Using uv to speedup installations
pip install uv

# Install AutoGaze and its dependencies
uv pip install -e .
```


## 💥 Quick Start

[QUICK_START.md](https://github.com/NVlabs/AutoGaze/blob/main/QUICK_START.md) provides some simple code snippets to get you started with AutoGaze!



## ⬇️ Download Models, Data, and Benchmark

The collection of all open-sourced models, data, benchmark can be found in [AutoGaze Collection](https://huggingface.co/collections/bfshi/autogaze).

| Name | Type | Description | HuggingFace Link |
|------------|-------------|---------------|---------------|
| **AutoGaze** | Model | Official pre-trained AutoGaze model. | [nvidia/AutoGaze](https://huggingface.co/nvidia/AutoGaze) |
| **NVILA-HD-Video** | Model | A video MLLM scaled to 1K frames, 4K resolution with AutoGaze | [nvidia/NVILA-8B-HD-Video](https://huggingface.co/nvidia/NVILA-8B-HD-Video) |
| **VideoMAE_AutoGaze** | Model | VideoMAE used to train AutoGaze. | [bfshi/VideoMAE_AutoGaze](https://huggingface.co/bfshi/VideoMAE_AutoGaze) |
| **AutoGaze-Training-Data** | Data | Training data for AutoGaze | [bfshi/AutoGaze-Training-Data](https://huggingface.co/datasets/bfshi/AutoGaze-Training-Data) |
| **HLVid** | Benchmark | A high-resolution, long-form video QA benchmark. | [bfshi/HLVid](https://huggingface.co/datasets/bfshi/HLVid) |


## 🤖 Training AutoGaze

See [TRAIN.md](https://github.com/NVlabs/AutoGaze/blob/main/TRAIN.md) for how to train AutoGaze.

## 🧩 Integrating AutoGaze into ViTs and MLLMs

We introduce [NVILA-8B-HD-Video](https://huggingface.co/nvidia/NVILA-8B-HD-Video), an efficient MLLM using AutoGaze. NVILA-8B-HD-Video uses AutoGaze to remove redundant patches before its vision encoder (SigLIP) and LLM, enabling efficient understanding of up to 4K-resolution, 1K-frame videos. This provides an example of how to integrate AutoGaze into ViTs and MLLMs. See the [VILA repo](https://github.com/NVlabs/VILA/tree/main/vila_hd/nvila_hd_video) for instructions on how to use NVILA-8B-HD-Video. See [INTEGRATION.md](https://github.com/NVlabs/AutoGaze/blob/main/INTEGRATION.md) for detailed guidelines on how to integrate AutoGaze into SigLIP and NVILA-HD-Video.

## 🌲 Code Structure

The main package `autogaze` is mainly structured as follows:

```bash
autogaze/
├── configs/
│   ├── algorithm/
│   ├── dataset/
│   ├── model/
│   ├── task/
│   └── trainer/
├── algorithms/
│   ├── grpo.py
│   ├── ntp.py
│   └── ...
├── datasets/
│   ├── video_folder.py
│   └── ...
├── models/
│   ├── autogaze/
│   └── ...
├── tasks/
│   ├── video_mae_reconstruction/
│   └── ...
├── vision_encoders/
│   ├── siglip/
│   └── ...
├── trainer.py
├── train.py
```

The package mainly consists of several components:

- `models`: Definition of **gaze models**, such as AutoGaze. The gaze model is responsible for predicting the gazing for each input video. The model usually takes an input and returns the `gazing_pos` as well as some other auxiliary information for training/inference such as `log_action_probs` of the gazing position and corresponding `gazing_mask`.
- `tasks`: Definition of the **task** serving as the training objective for gaze models, such as video mae reconstruction. Here everything that's related to the task is defined, including the task model (i.e., the model used in the task such as VideoMAE), the loss of the task (e.g., MAE reconstruction loss), or the reward function used to train the gaze model (e.g., reconstruction reward), the metrics used for logging, the visualization methods used during training, etc. Usually, the task class takes in the input video as well as the outputs from the gaze model, and return everything related to the task, such as the outputs from the task model, the task loss, the task reward, the task metrics, etc. As a side use, the task can also be used to train the task model (e.g., VideoMAE) itself.
- `algorithms`: The **algorithm** used to train the gaze model, e.g., next token prediction (NTP) or GRPO. Everything that's related to the algorithm is defined here, such as how the final RL loss is calculated based on the rewards provided by the task class. The algorithm takes in the input video as well as the outputs from the gazing model and task, and outputs the final loss for training the gazing model. Note that the algoirithm is **solely** responsible for training the gaze model, not the task model! The loss for training the task model is already defined in the task class.
- `datasets`: Datasets we use to train gazing models or task models.
- `vision_encoders`: Vision encoders that can be used with AutoGaze. Here you can customize existing vision encoders such as SigLIP or DINOv2 to make them compatible with AutoGaze. We've already implemented SigLIP.
- `trainer.py`: The trainer used to train the gaze model or task model. It takes in the model, task, algorithm, dataset, and train/val the model/task.
- `train.py`: The entry script for training. It instantiates the model, task, RL algorithm, dataset, and trainer, and then launches the trainer.
- `configs`: Configs for everything above.

This modularized structure allows easily adding new models, tasks, and algorithms, data, etc. For example, to add a task of DINOv2 feature reconstruction, one only needs to define a new task class without touching other parts. Sometimes adding some new features require changing multiple components, for example, adding a new task of Kinetics video classification will require defining the new task and adding a new Kinetics dataset.


## ✏️ Citation

If you find this work useful, please consider citing:
```
@misc{shi2026attendattentionefficientscalable,
      title={Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing}, 
      author={Baifeng Shi and Stephanie Fu and Long Lian and Hanrong Ye and David Eigen and Aaron Reite and Boyi Li and Jan Kautz and Song Han and David M. Chan and Pavlo Molchanov and Trevor Darrell and Hongxu Yin},
      year={2026},
      eprint={2603.12254},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12254}, 
}
```


