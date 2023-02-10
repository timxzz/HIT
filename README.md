# Trading Information between Latents in Hierarchical Variational Autoencoders [(ICLR 2023)](https://openreview.net/forum?id=eWtMdr6yCmL)

<div align="center">

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](http://arxiv.org/abs/2302.04855)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  <a href="http://timx.me" target="_blank">Tim&nbsp;Z.&nbsp;Xiao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://robamler.github.io" target="_blank">Robert&nbsp;Bamler</a> 
</div>

<div align="center">

**Table of Contents**: - <a href="#training">Training</a> - <a href="#evaluation">Evaluation</a> - <a href="#citation">Citation</a> -

</div>
<br>

## About The Project
This is the official github repository for our work [Trading Information between Latents in Hierarchical Variational Autoencoders](http://arxiv.org/abs/2302.04855), where we propose a _Hierarchical Information Trading (HIT)_ framework for VAEs.

> **Abstract**
>
> Variational Autoencoders (VAEs) were originally motivated (Kingma & Welling, 2014) as probabilistic generative models in which one performs approximate Bayesian inference. The proposal of $\beta$-VAEs (Higgins et al., 2017) breaks this interpretation and generalizes VAEs to application domains beyond generative modeling (e.g., representation learning, clustering, or lossy data compression) by introducing an objective function that allows practitioners to trade off between the information content ("bit rate") of the latent representation and the distortion of reconstructed data (Alemi et al., 2018). In this paper, we reconsider this rate/distortion trade-off in the context of hierarchical VAEs, i.e., VAEs with more than one layer of latent variables. We identify a general class of inference models for which one can split the rate into contributions from each layer, which can then be tuned independently. We derive theoretical bounds on the performance of downstream tasks as functions of the individual layers' rates and verify our theoretical findings in large-scale experiments. Our results provide guidance for practitioners on which region in rate-space to target for a given application.

## Environment: 

Python 3.8.11;
Other dependencies are in `requirements.txt`


## Training

Example training command:
```bash
python train.py \
--vae_type HVAE \
--dataset CIFAR10 \
--px_y_family_ll GaussianFixedSigma \
--sigma 0.71 \
--qz_family DiagonalGaussian \
--num_epochs 500 \
--batch_size 256 \
--conv_channels 32 \
--z_dims 32 \
--beta_y 1. \
--beta_z 1. \
--run_name <run_name> \
--run_batch_name <run_batch_name> \
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Evaluation
Involve 4 separate steps (also need to train classifier for measuring inception score):

#### 1.
```bash
python eval_beta_grid.py --run_batch_name <run_batch_name> --exclude_mi --with_classifier
```

#### 2.
Execute for all model in `<run_batch_name>`
```bash
python eval_beta_single.py \
--run_path <run_path_for_single_model> 
```

#### 3.
```bash
python eval_beta_grid.py --get_mi_cal_acc --cpu_n <num_cpu> --run_batch_name <run_batch_name> 
```

#### 4.
```bash
python eval_beta_grid.py --run_batch_name <run_batch_name> --recons_acc
```

<p align="right">(<a href="#top">back to top</a>)</p>

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Citation:
Following is the Bibtex if you would like to cite our paper :

```bibtex
@inproceedings{xiao2023trading,
  title={Trading Information between Latents in Hierarchical Variational Autoencoders},
  author={Xiao, Tim Z. and Bamler, Robert},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>