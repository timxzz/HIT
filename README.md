# Trading Information between Latents in Hierarchical Variational Autoencoders [(ICLR 2023)](https://openreview.net/forum?id=eWtMdr6yCmL)

<div align="center">
  <a href="http://timx.me" target="_blank">Tim&nbsp;Z.&nbsp;Xiao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://robamler.github.io" target="_blank">Robert&nbsp;Bamler</a> 
</div>
<br>
<br>

### Environment: 

Python 3.8.11;
Other dependencies are in `requirements.txt`


### Training

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

### Evaluation
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

## License
MIT OR Apache-2.0 OR BSL-1.0

## Bibtex:
If you would like to cite our paper, following is the Bibtex:

```
@inproceedings{xiao2023trading,
  title={Trading Information between Latents in Hierarchical Variational Autoencoders},
  author={Xiao, Tim Z. and Bamler, Robert},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```