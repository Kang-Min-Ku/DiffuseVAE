
# Description

This repo is for the team project of the Deep Generative Model lecture.

# Requirements (diffuseVAE)

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics==0.6.0 pytorch_lightning==1.4.9 lmdb click matplotlib wandb hydra-core scipy joblib scikit-learn torch-fidelity
```

- Python==3.11.0 (Align the Python version with the PyTorch compatibility)
- pytorch==2.2.0 (diffuseVAE is not strongly constrained by the PyTorch version)
- torchmetrics==0.6.0
- pytorch_lightning==1.4.9
- lmdb
- click
- matplotlib
- wandb
- hydra-core
- scipy
- joblib
- scikit-learn
- torch-fidelity

# Requirements (evaluation metric)
To calculate **off-the-shelf Inception Score (IS)**, the latest version of `torchmetrics` is required. However, this version causes a conflict when running **diffuseVAE**, necessitating the creation of a new Conda virtual environment.

While the repository also supports IS calculation compatible with **diffuseVAE**, using the off-the-shelf IS calculation is recommended for better accuracy and convenience

- torchmetrics>=1.6.0
- clean-fid

# Requirements (find similar ones)
To find similar images, you may need these ones.

- opencv-python
- scikit-image

# How to setup?

1. Pull main branch.
2. Make a directory for the dataset. Dataset itself would be downloaded automatically in the designated path, but you must make the directory to store it.
3. Set root paths for the datasets. There are two options:
   1. (Recommended) **modify `data.root`, in execution scripts**. The execution scripts are located in "**scripts/**".
   2. **modify the root path in the config file.** The configuration files for execution are located as .yaml files under subdirectories named after the dataset within the "**main/configs**" directory.
4. Set execution scripts (241117. Only train scripts now)
   1. VAE train
      - You may see an example in "**scripts/samples/**".
      - Set a dataset path at `dataset.vae.data.root`. An absolute path is recommended.
      - Set a device at `dataset.vae.training.device`.
      - Set `dataset.vae.training.results_dir` and `dataset.vae.training.chkpt_prefix`. These are the argument for `dataset.ddpm.training.vae_chkpt_path` in the DDPM training script.
   2. DDPM train
      - Set a dataset path at `dataset.ddpm.data.root`. An absolute path is recommended.
      - Set a device at `dataset.ddpm.training.device`.
      - Set `dataset.ddpm.training.vae_chkpt_path`. This is an absolute path of the output `.ckpt` file of VAE training.
      - Set `dataset.ddpm.training.results_dir` and `dataset.ddpm.training.chkpt_prefix`. These are argument for test (maybe).
5. Add permissions for the scripts.
   * For example, `chomd u+x scripts/samples/*.sh`.

# Tip
1. If you have warning about worker, then increase the number of workers!

# For avoiding hassle

1. make auto directory code

---
<span style="font-size:40px; color:#4e79a7; font-weight:600">Official README</span>

# DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents

This repo contains the official implementation of the paper: [DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents](https://arxiv.org/abs/2201.00308) by [Kushagra Pandey](https://kpandey008.github.io/), [Avideep Mukherjee](https://www.cse.iitk.ac.in/users/avideep/), [Piyush Rai](https://www.cse.iitk.ac.in/users/piyush/), [Abhishek Kumar](http://www.abhishek.umiacs.io/)

---
## Overview

 DiffuseVAE is a novel generative framework that integrates a standard VAE within a diffusion model by conditioning the diffusion model samples on the VAE generated reconstructions. The resulting model can significantly improve upon the blurry samples generated from a standard VAE while at the same time equipping diffusion models with the low-dimensional VAE inferred latent code which can be used for downstream tasks like controllable synthesis and image attribute manipulation. In short, DiffuseVAE presents a generative model which combines the benefits of both VAEs and Diffusion models.

![architecture!](./assets/diffusevae_tmlr-methodology.png)

Our core contributions are as follows:

1. We propose a generic DiffuseVAE conditioning framework and show that our framework can be reduced to a simple *generator-refiner* framework in which blurry samples generated from a VAE are refined using a conditional DDPM formulation.

1. **Controllable synthesis** from a low-dimensional latent using diffusion models.

1. **Better speed vs quality tradeoffs**: We show that DiffuseVAE inherently provides a better speed vs quality tradeoff as compared to standard DDPM/DDIM models on several image benchmarks

1. **State-of-the-art synthesis**:  We show that DiffuseVAE exhibits synthesis quality comparable to recent state-of-the-art on standard image synthesis benchmarks like CIFAR-10, CelebA-64 and CelebA-HQ while maintaining access to a low-dimensional latent code representation.

1. **Generalization to noisy conditioning signals**: We show that a pre-trained DiffuseVAE model exhibits generalization to different noise types in the DDPM conditioning signal exhibiting the effectiveness of our conditioning framework.

![High res samples!](./assets/diffusevae_tmlr-main.png)

---

## Code overview

This repo uses [PyTorch Lightning](https://www.pytorchlightning.ai/) for training and [Hydra](https://hydra.cc/docs/intro/) for config management so basic familiarity with both these tools is expected. Please clone the repo with `DiffuseVAE` as the working directory for any downstream tasks like setting up the dependencies, training and inference.

## Setting up the dependencies

We use `pipenv` for a project-level dependency management. Simply [install](https://pipenv.pypa.io/en/latest/#install-pipenv-today) `pipenv` and run the following command:

```
pipenv install
```

## Config Management
We manage `train` and `test` configurations separately for each benchmark/dataset used in this work. All configs are present in the `main/configs` directory. This directory has subfolders named according to the dataset. Each dataset subfolder contains the training and evaluation configs as `train.yaml` and `test.yaml`. 

**Note**: The configuration files consists of many command line options. The meaning of each of these options is explained in the config for CIFAR-10.

## Training
Please refer to the scripts provided in the table corresponding to some training tasks possible using the code.

|          **Task**          	|      **Reference**      	|
|:--------------------------:	|:-----------------------:	|
|  Training First stage VAE  	|  `scripts/train_ae.sh`  	|
| Training Second stage DDPM 	| `scripts/train_ddpm.sh` 	|

## Inference

Please refer to the scripts provided in the table corresponding to some inference tasks possible using the code.

|                          **Task**                         	|         **Reference**         	|
|:---------------------------------------------------------:	|:-----------------------------:	|
|            Sample/Reconstruct from Baseline VAE           	|      `scripts/test_ae.sh`     	|
|                   Sample from DiffuseVAE                  	|     `scripts/test_ddpm.sh`    	|
|          Generate reconstructions from DiffuseVAE         	| `scripts/test_recons_ddpm.sh` 	|
| Interpolate in the VAE/DDPM latent space using DiffuseVAE 	|    `scripts/interpolate.sh`   	|

For computing the evaluation metrics (FID, IS etc.), we use the [torch-fidelity](https://github.com/toshas/torch-fidelity) package. See `scripts/fid.sh` for some sample usage examples.


## Pretrained checkpoints
All pretrained checkpoints have been organized by dataset and can be accessed [here](https://drive.google.com/drive/folders/1GzIh75NnpgPa4A1hSb_viPowuaSHnL7R?usp=sharing).

## Citing
To cite DiffuseVAE please use the following BibTEX entries:

```
@misc{pandey2022diffusevae,
      title={DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents}, 
      author={Kushagra Pandey and Avideep Mukherjee and Piyush Rai and Abhishek Kumar},
      year={2022},
      eprint={2201.00308},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@inproceedings{
pandey2021vaes,
title={{VAE}s meet Diffusion Models: Efficient and High-Fidelity Generation},
author={Kushagra Pandey and Avideep Mukherjee and Piyush Rai and Abhishek Kumar},
booktitle={NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications},
year={2021},
url={https://openreview.net/forum?id=-J8dM4ed_92}
}
```

Since our model uses diffusion models please consider citing the original [Diffusion model](https://arxiv.org/abs/1503.03585), [DDPM](https://arxiv.org/abs/2006.11239) and [VAE](https://arxiv.org/abs/1312.6114) papers.

## Contact
Kushagra Pandey (@kpandey008)
