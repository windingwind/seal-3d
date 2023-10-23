# Seal-3D: Interactive Pixel-Level Editing for Neural Radiance Fields

![teaser](https://github.com/windingwind/seal-3d/assets/33902321/15e4898e-7658-4e46-8d90-96401340f4b2)

The official implementation of the paper [Seal-3D: Interactive Pixel-Level Editing for Neural Radiance Fields](), the first interactive pixel-level NeRF editing tool.

Accepted by ICCV 2023.

[Project Page](https://windingwind.github.io/seal-3d/) ｜ 
[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Seal-3D_Interactive_Pixel-Level_Editing_for_Neural_Radiance_Fields_ICCV_2023_paper.html) ｜ 
[ArXiv](https://arxiv.org/abs/2307.15131) ｜ 
[Code](https://github.com/windingwind/seal-3d)

This project is built on [ashawkey/torch-ngp](https://github.com/ashawkey/torch-ngp)'s NGP and TensoRF implementation.

## Installation

To find more details about the development environment setup, please refer to [torch-ngp#install](https://github.com/ashawkey/torch-ngp#install).

```bash
git clone --recursive https://github.com/windingwind/seal-3d.git
cd seal-3d
```

### Install with pip

```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install with conda

```bash
conda env create -f environment.yml
conda activate torch-ngp
```

### Build extension (optional)

By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, we also provide the `setup.py` to build each extension:

```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

## Dataset

We use the same data format as instant-ngp. Please download and put them under `./data`.

To find more details about the supported dataset, please refer to [torch-ngp#usage](https://github.com/ashawkey/torch-ngp#usage).

## Usage

### Code Structure

Based on the implementation of the repo, we slightly modified the files in `nerf` (the NGP implementation) and `tensoRF` (the TensoRF implementation) to fit our needs.

The main entrances are `main_SealNeRF.py` (NGP backbone) and `main_SealTensoRF.py` (TensoRF backbone).

In `SealNeRF`:

- `trainer.py` defines the trainer class dynamically depending on the backbone and character (student/teacher).

- `network.py` defines the network class dynamically depending on the backbone and character (student/teacher).

- `provider.py` defines the dataset update strategy under our two-stage local-global teacher-student framework.

- `seal_utils.py` defines the proxy functions we proposed in the paper.

- `renderer.py` defines how the proxy functions are applied to our pipeline.

### Train

Follow the steps below to apply the editing operation on an existing NeRF model:

1. Train an NGP/TensoRF model following the instructions of [torch-ngp#usage](https://github.com/ashawkey/torch-ngp#usage). For example:

```bash
# NGP backbone, Lego
python main_nerf.py data/nerf_synthetic/lego/ --workspace exps/lego_ngp -O --bound 1.0 --scale 0.8 --dt_gamma 0
```

2. Train Seal3D on the model you get in the previous step (headless mode).

```bash
# Headless mode, bounding shape editing, NGP backbone, Lego
# pretraining_epochs: pretraining stage epochs
# extra_epochs: total epochs (pretraining + finetuning)
# pretraining_*_point_step: pretraining sample step
# ckpt: the input student model checkpoint
# teacher_workspace: teacher model workspace
# teacher_ckpt: teacher model checkpoint
# seal_config: the editing config directory used in headless mode. the config file is $seal_config/seal.json.
# eval_interval & eval_count: control eval behavior
python main_SealNeRF.py data/nerf_synthetic/lego/\
    --workspace exps/lego_ngp_bbox -O --bound 1.0 --scale 0.8 --dt_gamma 0\
    --pretraining_epochs 100 --extra_epochs 150\
    --pretraining_local_point_step 0.005 --pretraining_surrounding_point_step -1\
    --pretraining_lr 0.05 --ckpt exps/lego_ngp/checkpoints/ngp_ep0300.pth\
    --teacher_workspace exps/lego_ngp --teacher_ckpt exps/lego_ngp/checkpoints/ngp_ep0300.pth\
    --seal_config data/seal/lego_bbox/\
    --eval_interval 100 --eval_count 10
```

The `seal_config` files used by examples in the paper can be downloaded from [Google Drive link](https://drive.google.com/file/d/1PWTtO9EqOas5Qh-sRccYVRiJO6wgHoN7/view?usp=sharing). The explanation for the parameters of `seal_config` can be found in the corresponding class of the proxy function in [SealNeRF/seal_utils](SealNeRF/seal_utils.py).

The full argument list and descriptions can be found in the corresponding entrance file (`main_*.py`).

To start in GUI mode, use `--gui`.

> Currently, GUI mode supports *Color*, *Anchor*, *Brush*, and *Texture* editing.

## BibTeX

```bibtex
@misc{wang2023seal3d,
      title={Seal-3D: Interactive Pixel-Level Editing for Neural Radiance Fields}, 
      author={Xiangyu Wang and Jingsen Zhu and Qi Ye and Yuchi Huo and Yunlong Ran and Zhihua Zhong and Jiming Chen},
      year={2023},
      eprint={2307.15131},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

Use this code under the MIT License. No warranties are provided. Keep the laws of your locality in mind!

Please refer to [torch-ngp#acknowledgement](https://github.com/ashawkey/torch-ngp#acknowledgement) for the acknowledgment of the original repo.
