import sys
import os
import time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import distributions as D

import numpy as np

from models import HVAE_Fixed_z, HVAE_Conv_z
import models_lvae
import dataloader
import utils_runs

rseed = 2
torch.manual_seed(rseed)
torch.cuda.manual_seed_all(rseed)

def train(args):
    out_dir = './runs/' + args.run_batch_name

    num_epochs = args.num_epochs
    qz_family = args.qz_family
    use_gpu = True

    if qz_family == "GumbelSoftmax":
        # Gumbel Softmax annealing
        gs_temp_min = 0.4
        GS_TEMP_ANNEAL_RATE = 0.007
        gs_temp = 1
    else:
        gs_temp = None

    if qz_family == "GumbelSoftmax" and args.use_conv_z:
        raise NotImplementedError('GumbelSoftmax is not support in HVAE_Conv_z')


    config_string = (
        f"-Epochs_{num_epochs}"
        f"-BatchSize_{args.batch_size}"
        f"-c_{args.conv_channels}"
        f"-{args.dataset}"
        f"-{args.vae_type}"
        f"-px_y_family_ll_{args.px_y_family_ll}"
        f"-qz_family_{qz_family}"
        f"-Beta_y_{args.beta_y}"
        f"-Beta_z_{args.beta_z}"
    )
    if args.px_y_family_ll == "GaussianFixedSigma":
        config_string += f"-ll_sigma_{args.sigma}"
    if args.dataset != "ImageNet" and not args.use_conv_z:
        config_string += f"-z_dim_{args.z_dims}"
    else:
        config_string += f"-use_conv_z"
    if args.linear_y_with_dims != -1:
        config_string += f"-linear_y_dims_{args.linear_y_with_dims}"
    if args.vae_type == "LVAE" and args.no_warmup:
        config_string += f"-no_warmup"

    if args.run_name == "":
        run_name = "NoName" + config_string
    else:
        run_name = args.run_name + config_string
    print(run_name)

    train_dataloader, test_dataloader = dataloader.load_data(args.dataset, args.batch_size)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    print(device)
    if args.dataset == "ImageNet" or args.use_conv_z:
        if args.vae_type == "HVAE":
            vae = HVAE_Conv_z(device, qz_family=qz_family, px_y_family_ll=args.px_y_family_ll,
                            sigma=args.sigma, 
                            dataset=args.dataset, num_c=args.conv_channels,
                            beta_y=args.beta_y, beta_z=args.beta_z)
        elif args.vae_type == "LVAE":
            vae = models_lvae.LVAE_Conv_z(device, qz_family=qz_family, px_y_family_ll=args.px_y_family_ll, 
                            sigma=args.sigma,
                            dataset=args.dataset, num_c=args.conv_channels,
                            beta_y=args.beta_y, beta_z=args.beta_z)
    else:
        if args.vae_type == "HVAE":
            vae = HVAE_Fixed_z(device, qz_family=qz_family, px_y_family_ll=args.px_y_family_ll, 
                            sigma=args.sigma,
                            dataset=args.dataset, num_c=args.conv_channels, z_dims=args.z_dims,
                            linear_y_with_dims=args.linear_y_with_dims,
                            beta_y=args.beta_y, beta_z=args.beta_z)
        elif args.vae_type == "LVAE":
            vae = models_lvae.LVAE_Fixed_z(device, qz_family=qz_family, px_y_family_ll=args.px_y_family_ll, 
                            sigma=args.sigma,
                            dataset=args.dataset, num_c=args.conv_channels, z_dims=args.z_dims,
                            linear_y_with_dims=args.linear_y_with_dims,
                            beta_y=args.beta_y, beta_z=args.beta_z)
    vae = vae.to(device)

    # Record number of parameters in the model
    num_of_model_params = sum(p.numel() for p in vae.parameters())
    print('Number of model parameters: {}'.format(num_of_model_params))


    optimizer = torch.optim.Adam(params=vae.parameters(), lr=args.learning_rate)

    # set to training mode
    vae.train()


    # Save training config
    utils_runs.save_train_config(out_dir, run_name, vars(args))

    train_loss_avg = []
    kl_z_avg = []
    kl_y_avg = []
    ll_avg = []

    print('Training ...')
    batch_idx = 0
    step_idx = 0
    step_record_gap = 100
    wu_temp = 0
    wu_max_epoch = (num_epochs // 5)
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        ll_avg.append(0)
        kl_z_avg.append(0)
        kl_y_avg.append(0)
        num_batches = 0
        
        for image_batch, _ in train_dataloader:
            optimizer.zero_grad()
            
            image_batch = image_batch.to(device)

            if args.vae_type == "HVAE":
                if qz_family == "GumbelSoftmax":
                    loss, ll, kl_z, kl_y, \
                        qz_pi, qy_mu, qy_std, py_mu, py_std,_ = vae(image_batch, gs_temp)
                elif qz_family == "DiagonalGaussian":
                    loss, ll, kl_z, kl_y, \
                        qz_mu, qz_std, qy_mu, qy_std, py_mu, py_std,_ = vae(image_batch)
            elif args.vae_type == "LVAE":
                if qz_family == "GumbelSoftmax":
                    loss, ll, kl_z, kl_y, \
                        qz_pi, qy_mu, qy_std, py_mu, py_std,_ = vae(image_batch, gs_temp, wu_temp=wu_temp)
                elif qz_family == "DiagonalGaussian":
                    loss, ll, kl_z, kl_y, \
                        qz_mu, qz_std, qy_mu, qy_std, py_mu, py_std,_ = vae(image_batch, wu_temp=wu_temp)
            
            # backpropagation
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            if qz_family == "GumbelSoftmax":
                gs_temp = np.maximum(np.exp(-GS_TEMP_ANNEAL_RATE * epoch), gs_temp_min)
            
            if args.vae_type == "LVAE" and args.no_warmup:
                wu_temp = 1
            elif args.vae_type == "LVAE" and epoch < wu_max_epoch:
                wu_temp = float(epoch) / wu_max_epoch
            elif args.vae_type == "LVAE":
                wu_temp = 1
            
            train_loss_avg[-1] += loss.item()
            ll_avg[-1] += ll.item()
            kl_z_avg[-1] += kl_z.item()
            kl_y_avg[-1] += kl_y.item()
            num_batches += 1
            batch_idx += 1
            step_idx += 1

            if step_idx % step_record_gap == 1:
                print(f"Epoch-Step[{epoch+1}/{num_epochs}-{batch_idx}/{len(train_dataloader)}]")
            
        # Logging for each epoch:
        train_loss_avg[-1] /= num_batches
        ll_avg[-1] /= num_batches
        kl_z_avg[-1] /= num_batches
        kl_y_avg[-1] /= num_batches
        print(f"Epoch [{epoch+1} / {num_epochs}] average negative ELBO: {train_loss_avg[-1]}, "
                f"KL_z : {kl_z_avg[-1]}, KL_y : {kl_y_avg[-1]}, temp : {gs_temp}")

        likelihood_sigma = torch.exp(vae.log_sigma).item()

        if qz_family == "GumbelSoftmax":
            qz_pi = qz_pi.detach().cpu().numpy()
        elif qz_family == "DiagonalGaussian":
            qz_mu = qz_mu.detach().cpu().numpy()
            qz_logstd = qz_std.detach().cpu().log().numpy()


        qy_mu = qy_mu.detach().cpu().numpy()
        qy_logstd = qy_std.detach().cpu().log().numpy()
        py_mu = py_mu.detach().cpu().numpy()
        py_logstd = py_std.detach().cpu().log().numpy()


        # Save model
        utils_runs.save_model(out_dir, run_name, vae)





if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--vae_type", choices=["HVAE", "LVAE"],
        help="Choose the type of HVAE")
    parser.add_argument(
        '--no_warmup', action='store_true', default=False,
        help='Disable warmup in LVAE')
    parser.add_argument(
        "--dataset", choices=["BinaryMNIST", "MNIST", "CIFAR10", "ImageNet", "SVHN"],
        help="Choose the dataset for the experment.")
    parser.add_argument(
        '--use_conv_z', action='store_true', default=False,
        help='Use convolutional layers for z')
    parser.add_argument(
        "--qz_family", choices=["GumbelSoftmax", "DiagonalGaussian"],
        help="Choose the distribution family for qz.")
    parser.add_argument(
        "--px_y_family_ll", choices=["GaussianFixedSigma", "GaussianLearnedSigma", "Bernoulli", "MoL"],
        help="Choose the likelihood / distribution family for p(x|y).")

    parser.add_argument(
        "--num_epochs", type=int, default=200,
        help="Number of epochs for training.")
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for training.")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="Learning rate.")

    parser.add_argument(
        "--conv_channels", type=int, default=16,
        help="Number of channels in conv layers.")
    parser.add_argument(
        "--z_dims", type=int, default=10,
        help="Number of dimensions for z.")
    parser.add_argument(
        "--sigma", type=float, default=0.03,
        help="Standard deviation of gaussian p(x|y).")
    parser.add_argument(
        "--linear_y_with_dims", type=int, default=-1,
        help="If specified, y will be modelled in the dims with linear layers. "
            + "Otherwise, the dims of y will be determined based on the input image size under CNNs.")
    parser.add_argument(
        "--beta_y", type=float, default=1.,
        help="Beta weight for the EBLO term: < KL[q(y|z,x) || p(y|z)] >_q(z|x).")
    parser.add_argument(
        "--beta_z", type=float, default=1.,
        help="Beta weight for the EBLO term: KL[q(z|x) || p(z)].")

    parser.add_argument(
        "--run_name", default="",
        help="Specified the name of the run.")
    parser.add_argument(
        "--run_batch_name", default="singles",
        help="Specified the name of the batch for runs if doing a batch grid search etc.")

    

    args = parser.parse_args()

    train(args)
