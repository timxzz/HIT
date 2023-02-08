import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as D

from abc import abstractmethod
import math
import numpy as np

import utils


def sample_MoL(fy):
    '''
    Sample from mixture of logistic given the network output
    TODO: Also need mean/mode for MoL
    '''
    samples = utils.discretized_mix_logistic_rsample(fy)
    samples = (samples + 1) / 2  # Transform from [-1, 1] to [0, 1]
    samples = samples.clamp(min=0.0, max=1.0)
    return samples


class BaseLVAE(nn.Module):
  def __init__(self, device, qz_family, px_y_family_ll, dataset, sigma=0.03, n_mix=10, beta_y=1, beta_z=1, hard_gs=False):
    super(BaseLVAE, self).__init__()

    self.hard_gs = hard_gs
    self.device = device
    self.dataset = dataset
    self.beta_y = beta_y
    self.beta_z = beta_z
    if dataset == "MNIST" or dataset == "BinaryMNIST":
        self.img_HW = 28
        self.img_channels = 1
    elif dataset == "CIFAR10" or dataset == "SVHN":
        self.img_HW = 32
        self.img_channels = 3
    elif dataset == "ImageNet":
        self.img_HW = 256
        self.img_channels = 3

    self.likelihood_family = px_y_family_ll
    self.qz_family = qz_family

    if self.likelihood_family == 'MoL':
        # Number of mix logistic components for MoL
        self.n_mix = n_mix
        self.px_out_channels = (self.img_channels * 3 + 1) * self.n_mix # mean, variance and mixture coeff per channel plus logits
    else:
        self.px_out_channels = self.img_channels

    # Set standard deviation of p(x|z)
    # self.log_sigma = 0
    self.log_sigma = torch.tensor(sigma).log()
    if self.likelihood_family == 'GaussianLearnedSigma':
        ## Sigma VAE
        self.log_sigma = nn.Parameter(torch.full((1,), 0, dtype=torch.float32)[0])

  @abstractmethod    
  def q_z(self, x, temp=0.5):
    return

  @abstractmethod 
  def q_y(self, dy, py_mu, py_std):
    return

  @abstractmethod 
  def p_y(self, z):
    return 

  @abstractmethod 
  def sample_x(self, num=10, z=None):
    return

  def reparameterize_Normal(self, mu, std):
    eps = torch.randn(mu.size())
    eps = eps.to(self.device)

    return mu + eps * std

  def reparameterize_Gumbel_Softmax(self, z_logit, temperature):
    """
    Refer to: https://github.com/YongfeiYan/Gumbel_Softmax_VAE
    """
    z_gs = utils.gumbel_softmax_sample(z_logit, temperature)

    if not self.hard_gs:
        return z_gs

    shape = z_gs.size()
    _, ind = z_gs.max(dim=-1)
    z_gs_hard = torch.zeros_like(z_gs).view(-1, shape[-1])
    z_gs_hard.scatter_(1, ind.view(-1, 1), 1)
    z_gs_hard = z_gs_hard.view(*shape)
    # Set gradients w.r.t. z_gs_hard gradients w.r.t. z_gs
    z_gs_hard = (z_gs_hard - z_gs).detach() + z_gs
    return z_gs_hard

  def reconstruction(self, x, temp=0.5):
    if self.qz_family == "GumbelSoftmax":
        _, qz_pi, dy = self.q_z(x, temp)

        shape = qz_pi.size()
        _, ind = qz_pi.max(dim=-1)
        z_onehot = torch.zeros_like(qz_pi).view(-1, shape[-1])
        z_onehot.scatter_(1, ind.view(-1, 1), 1)
        z_onehot = z_onehot.view(*shape)

        _, py_mu, py_std = self.p_y(z_onehot)
        _, y_mu, _ = self.q_y(dy, py_mu, py_std)
    elif self.qz_family == "DiagonalGaussian":
        _, z_mu, _, dy = self.q_z(x)
        _, py_mu, py_std = self.p_y(z_mu)
        _, y_mu, _ = self.q_y(dy, py_mu, py_std)
    
    fy = self.p_x(y_mu)

    if self.likelihood_family == "MoL":
        fy = sample_MoL(fy)     # TODO: Need mean/mode for MoL

    return fy

  def loglikelihood_x_y(self, x, fy):
    """ Computer the loglikelihood: <log p(x|y)>_q
    - For MNIST, we use Bernoulli for p(x|y)
    - For Colour Image, we can try out:
      1. N(f(y), (c I)^2), gaussian with constant variance
      2. N(f(y), (sigma I)^2), gaussian with shared learnt variance
      3. Mixture of logistics:
            Assume input data to be originally uint8 (0, ..., 255) and then rescaled
        by 1/255: discrete values in {0, 1/255, ..., 255/255}.
        When using the original discretize logistic mixture logprob implementation,
        this data should be rescaled to be in [-1, 1].
      etc.

      see paper 'Simple and Effective VAE Training with Calibrated Decoders'
        by Oleh Rybkin, Kostas Daniilidis, Sergey Levine 
      https://arxiv.org/pdf/2006.13202.pdf

      code : https://github.com/orybkin/sigma-vae-pytorch/blob/master/model.py
    """

    if self.likelihood_family == 'GaussianFixedSigma':
        # For constant variance, assume it's c: i.e. self.log_sigma
        log_sigma = self.log_sigma
    elif self.likelihood_family == 'GaussianLearnedSigma':
        # Sigma VAE learns the variance of the decoder as another parameter
        log_sigma = self.log_sigma

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        min = -6
        log_sigma = min + F.softplus(log_sigma - min)
    elif self.likelihood_family == 'MoL':
        x = x * 2 - 1  # Transform from [0, 1] to [-1, 1]
    elif self.likelihood_family == 'Bernoulli':
        x = x.view(-1, self.img_channels * self.img_HW**2)
        fy = fy.view(-1, self.img_channels * self.img_HW**2)
    else:
        raise NotImplementedError

    if self.likelihood_family == 'MoL':
        # mixture of logistic likelihood
        ll = utils.log_discretized_mix_logistic(x, fy)
    elif self.likelihood_family == 'GaussianFixedSigma' or self.likelihood_family == 'GaussianLearnedSigma':
        # gaussian log likelihood
        nll = 0.5 * (((x - fy)**2) * torch.exp(-2*log_sigma) + 2*log_sigma + torch.log(torch.tensor(2 * math.pi)))
        nll = torch.sum(torch.flatten(nll, start_dim=1), dim=-1)
        ll = -nll
    elif self.likelihood_family == 'Bernoulli':
        ll = torch.sum(torch.flatten(x * torch.log(fy + 1e-8)
                                + (1 - x) * torch.log(1 - fy + 1e-8),
                                start_dim=1),
                        dim=-1)
    return ll

  def forward(self, x, temp=0.5, wu_temp=1):
    if self.qz_family == "GumbelSoftmax":
        z, qz_pi, dy = self.q_z(x, temp)
    elif self.qz_family == "DiagonalGaussian":
        z, qz_mu, qz_std, dy = self.q_z(x)

    _, py_mu, py_std = self.p_y(z)
    y, qy_mu, qy_std = self.q_y(dy, py_mu, py_std)
    fy = self.p_x(y)

    # For likelihood : <log p(x|y)>_q :
    ll = self.loglikelihood_x_y(x, fy)

    qy = D.normal.Normal(qy_mu, qy_std)
    py = D.normal.Normal(py_mu, py_std)
    if self.y_by_Conv:
        # keep the c x H x W shape
        qy = D.independent.Independent(qy, 3)
        py = D.independent.Independent(py, 3)
    else:
        qy = D.independent.Independent(qy, 1)
        py = D.independent.Independent(py, 1)

    # For: -KL[q(z|x) || p(z)]
    if self.qz_family == "GumbelSoftmax":
        kl_z = utils.kl_categorical(qz_pi, self.z_dims)
    elif self.qz_family == "DiagonalGaussian":
        qz = D.normal.Normal(qz_mu, qz_std)
        pz = D.normal.Normal(torch.zeros_like(z), torch.ones_like(z))
        if self.z_by_Conv:
            qz = D.independent.Independent(qz, 3)
            pz = D.independent.Independent(pz, 3)
        else:
            qz = D.independent.Independent(qz, 1)
            pz = D.independent.Independent(pz, 1)
        kl_z = D.kl.kl_divergence(qz, pz)

    # For: - < KL[q(y|z,x) || p(y|z)] >_q(z|x)
    kl_y = D.kl.kl_divergence(qy, py)

    elbo = ll - wu_temp * self.beta_y * kl_y - wu_temp * self.beta_z * kl_z

    if self.qz_family == "GumbelSoftmax":
        return -elbo.mean(), ll.mean(), kl_z.mean(), kl_y.mean(), \
            qz_pi, qy_mu, qy_std, py_mu, py_std, z
    elif self.qz_family == "DiagonalGaussian":
        return -elbo.mean(), ll.mean(), kl_z.mean(), kl_y.mean(), \
            qz_mu, qz_std, qy_mu, qy_std, py_mu, py_std, z




class LVAE_Fixed_z(BaseLVAE):
  def __init__(self, device, qz_family, px_y_family_ll, dataset, sigma=0.03, num_c=16, z_dims=10, linear_y_with_dims=-1,
                beta_y=1, beta_z=1,
                hard_gs=False):
    super(LVAE_Fixed_z, self).__init__(device, qz_family, px_y_family_ll, dataset, sigma=sigma,
                    beta_y=beta_y, beta_z=beta_z,
                    hard_gs=hard_gs)

    self.z_by_Conv = False
    self.c = num_c
    self.z_dims = z_dims
    if dataset == "MNIST" or dataset == "BinaryMNIST":
        self.kernel_size_mid = 4
        self.stride_mid = 1
        self.padding_mid = 0
    elif dataset == "CIFAR10" or dataset == "SVHN":
        self.kernel_size_mid = 4
        self.stride_mid = 2
        self.padding_mid = 1
    self.kernel_size1 = 4
    self.stride1 = 2
    self.padding1 = 1
    self.kernel_size_qy_last = 3
    self.stride_qy_last = 1
    self.padding_qy_last = 1
    self.mid_HW_1 = ((self.img_HW - self.kernel_size1 + 2*self.padding1) // self.stride1 + 1)
    self.mid_HW_2 = ((self.mid_HW_1 - self.kernel_size1 + 2*self.padding1) // self.stride1 + 1)
    self.mid_HW_3 = ((self.mid_HW_2 - self.kernel_size_mid + 2*self.padding_mid) // self.stride_mid + 1)
    if linear_y_with_dims == -1:
        self.y_by_Conv = True
        self.y_dims = self.c*self.mid_HW_3*self.mid_HW_3
    else:
        self.y_by_Conv = False
        self.y_dims = linear_y_with_dims

    # Layers for q(z|x):
    self.qz_conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=self.c, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1) # out: c x 16 x 16 or 14 x 14
    self.qz_conv2_dy = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1) # out: c x 8 x 8 or 7 x 7
    self.qz_conv3 = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.kernel_size_mid, stride=self.stride_mid, padding=self.padding_mid) # out: c x 4 x 4
    if self.qz_family == "GumbelSoftmax":
        self.qz_logit = nn.Linear(in_features=self.c*self.mid_HW_3*self.mid_HW_3, out_features=self.z_dims)
    elif self.qz_family == "DiagonalGaussian":
        self.qz_mu = nn.Linear(in_features=self.c*self.mid_HW_3*self.mid_HW_3, out_features=self.z_dims)
        self.qz_pre_sp = nn.Linear(in_features=self.c*self.mid_HW_3*self.mid_HW_3, out_features=self.z_dims)

    # Layers for q(y|z,x):
    self.qy_conv1_dy = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.kernel_size_mid, stride=self.stride_mid, padding=self.padding_mid) # out: c x 4 x 4
    if self.y_by_Conv:
        self.qy_mu = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.kernel_size_qy_last, stride=self.stride_qy_last, padding=self.padding_qy_last) # out: c x 4 x 4
        self.qy_pre_sp = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.kernel_size_qy_last, stride=self.stride_qy_last, padding=self.padding_qy_last) # out: c x 4 x 4
    else:
        self.qy_mu = nn.Linear(in_features=self.c*self.mid_HW_3*self.mid_HW_3, out_features=self.y_dims)
        self.qy_pre_sp = nn.Linear(in_features=self.c*self.mid_HW_3*self.mid_HW_3, out_features=self.y_dims)

    # Layers for p(y|z):
    h_dims = self.y_dims // 2
    self.py_l1 = nn.Linear(in_features=self.z_dims, out_features=h_dims)
    self.py_mu = nn.Linear(in_features=h_dims, out_features=self.y_dims)
    self.py_pre_sp = nn.Linear(in_features=h_dims, out_features=self.y_dims)

    # Layers for p(x|y):
    if self.y_by_Conv:
        self.px_conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=self.kernel_size_mid, stride=self.stride_mid, padding=self.padding_mid)
    else:
        self.px_l1 = nn.Linear(in_features=self.y_dims, out_features=self.c*self.mid_HW_2*self.mid_HW_2)
    self.px_conv2 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
    self.px_conv3 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.px_out_channels, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)

  def q_z(self, x, temp=0.5):
    h = F.relu(self.qz_conv1(x))
    dy = F.relu(self.qz_conv2_dy(h))
    h = F.relu(self.qz_conv3(dy))
    h = h.view(h.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors

    if self.qz_family == "GumbelSoftmax":
        z_logit = self.qz_logit(h)
        return self.reparameterize_Gumbel_Softmax(z_logit, temp), F.softmax(z_logit, dim=-1), dy
    elif self.qz_family == "DiagonalGaussian":
        z_mu = self.qz_mu(h)
        z_pre_sp = self.qz_pre_sp(h)
        z_std = F.softplus(z_pre_sp)
        return self.reparameterize_Normal(z_mu, z_std), z_mu, z_std, dy

  def q_y(self, dy, py_mu, py_std):
    h = F.relu(self.qy_conv1_dy(dy))
    if not self.y_by_Conv:
        h = h.view(h.size(0), -1) # flatten 
    y_mu_hat = self.qy_mu(h)
    y_pre_sp = self.qy_pre_sp(h)
    y_var_hat = F.softplus(y_pre_sp)

    py_var = py_std**2
    y_std = torch.sqrt((y_var_hat * py_var) / (y_var_hat + py_var))
    y_mu = (y_mu_hat * py_var + py_mu * y_var_hat) / (y_var_hat + py_var)

    return self.reparameterize_Normal(y_mu, y_std), y_mu, y_std

  def p_y(self, z):
    h = F.relu(self.py_l1(z))
    y_mu = self.py_mu(h)
    y_pre_sp = self.py_pre_sp(h)
    y_std = F.softplus(y_pre_sp)

    if self.y_by_Conv:
        y_mu = y_mu.view(y_mu.size(0), self.c, self.mid_HW_3, self.mid_HW_3) # unflatten 
        y_std = y_std.view(y_std.size(0), self.c, self.mid_HW_3, self.mid_HW_3) # unflatten 
    return self.reparameterize_Normal(y_mu, y_std), y_mu, y_std

  def p_x(self, y):
    if self.y_by_Conv:
        h = F.relu(self.px_conv1(y))
    else:
        h = F.relu(self.px_l1(y))
        h = h.view(h.size(0), self.c, self.mid_HW_2, self.mid_HW_2) # unflatten 
    h = F.relu(self.px_conv2(h))
    x = self.px_conv3(h)
    if self.likelihood_family == 'Bernoulli':
        x = torch.sigmoid(x) # last layer before output is sigmoid if we are using Bernoulli
    return x

  def sample_x(self, num=10, z=None):
    if z is None:
        if self.qz_family == "GumbelSoftmax":
            # sample latent vectors from 10 different z
            z = F.one_hot(torch.arange(0, self.z_dims), num_classes=self.z_dims).float()
            z = z.repeat(1, num).view(-1, self.z_dims) # Repeat for sampling y
        elif self.qz_family == "DiagonalGaussian":
            # sample latent vectors from the normal distribution
            z = torch.randn(num, self.z_dims)

    z = z.to(self.device)

    y_hat, _, _ = self.p_y(z)
    fy = self.p_x(y_hat)

    if self.likelihood_family == "MoL":
        fy = sample_MoL(fy)

    return fy



class LVAE_Conv_z(BaseLVAE):
  def __init__(self, device, qz_family, px_y_family_ll, dataset, sigma=0.03, num_c=192, 
                beta_y=1, beta_z=1,
                hard_gs=False):
    super(LVAE_Conv_z, self).__init__(device, qz_family, px_y_family_ll, dataset, sigma=sigma,
                    beta_y=beta_y, beta_z=beta_z,
                    hard_gs=hard_gs)

    if qz_family == "GumbelSoftmax":
        NotImplementedError('GumbelSoftmax is not support in LVAE_Conv_z')
    if self.dataset == "ImageNet":
        NotImplementedError('ImageNet is not support in LVAE_Conv_z')
    self.y_by_Conv = True
    self.z_by_Conv = True

    self.y_dims = -1

    self.c = num_c
    self.k_size_1 = 5
    self.stride_1 = 2
    self.padding_1 = self.k_size_1 // 2
    self.output_padding_1 = self.stride_1 - 1
    self.k_size_2 = 3
    self.stride_2 = 1
    self.padding_2 = self.k_size_2 // 2
    self.output_padding_2 = self.stride_2 - 1
    self.k_size_3 = 5
    self.stride_3 = 1
    self.padding_3 = 2

    # Refer to architecture of Minnen's JointAutoregressive paper
    qz_convs_layers_dy = []
    qz_convs_layers_dz = []
    qz_convs_layers_dy.append(nn.Conv2d(in_channels=self.img_channels, out_channels=self.c, kernel_size=self.k_size_1, stride=self.stride_1, padding=self.padding_1))
    qz_convs_layers_dy.append(nn.ReLU())
    qz_convs_layers_dy.append(nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=self.k_size_2, stride=self.stride_2, padding=self.padding_2))
    qz_convs_layers_dy.append(nn.ReLU())
    qz_convs_layers_dz.append(nn.Conv2d(in_channels=self.c, out_channels=self.c+(self.c*2), kernel_size=self.k_size_1, stride=self.stride_1, padding=self.padding_1))
    qz_convs_layers_dz.append(nn.ReLU())
    qz_convs_layers_dz.append(nn.Conv2d(in_channels=self.c+(self.c*2), out_channels=self.c*2, kernel_size=self.k_size_1, stride=self.stride_1, padding=self.padding_1))
    self.qz_convs_dy = nn.Sequential(*qz_convs_layers_dy)
    self.qz_convs_dz = nn.Sequential(*qz_convs_layers_dz)

    qy_convs_dy_layers = []
    qy_convs_dy_layers.append(nn.Conv2d(in_channels=self.c, out_channels=self.c+(self.c*2), kernel_size=self.k_size_1, stride=self.stride_1, padding=self.padding_1))
    qy_convs_dy_layers.append(nn.ReLU())
    qy_convs_dy_layers.append(nn.Conv2d(in_channels=self.c+(self.c*2), out_channels=self.c*2, kernel_size=self.k_size_3, stride=self.stride_3, padding=self.padding_3))
    self.qy_convs_dy = nn.Sequential(*qy_convs_dy_layers)

    py_deconvs_layers = []
    py_deconvs_layers.append(nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c+(self.c//2), kernel_size=self.k_size_1, stride=self.stride_1, padding=self.padding_1, output_padding=self.output_padding_1))
    py_deconvs_layers.append(nn.ReLU())
    py_deconvs_layers.append(nn.ConvTranspose2d(in_channels=self.c+(self.c//2), out_channels=self.c*2, kernel_size=self.k_size_2, stride=self.stride_2, padding=self.padding_2, output_padding=self.output_padding_2))
    self.py_deconvs = nn.Sequential(*py_deconvs_layers)

    px_deconvs_layers = []
    px_deconvs_layers.append(nn.ConvTranspose2d(in_channels=self.c, out_channels=self.c, kernel_size=self.k_size_1, stride=self.stride_1, padding=self.padding_1, output_padding=self.output_padding_1))
    px_deconvs_layers.append(nn.ReLU())
    px_deconvs_layers.append(nn.ConvTranspose2d(in_channels=self.c, out_channels=self.px_out_channels, kernel_size=self.k_size_1, stride=self.stride_1, padding=self.padding_1, output_padding=self.output_padding_1))
    self.px_deconvs = nn.Sequential(*px_deconvs_layers)

    # latent z size depends on input image size, compute the output size
    demo_input = torch.ones([1, self.img_channels, self.img_HW, self.img_HW])
    self.z_HW = self.qz_convs_dz(self.qz_convs_dy(demo_input)).shape[2]
    print('z_HW', self.z_HW)

  def q_z(self, x, temp=0.5):
    dy = self.qz_convs_dy(x)
    z_mu, z_pre_sp = self.qz_convs_dz(dy).chunk(2, dim=1) # Split over channels
    z_var = F.softplus(z_pre_sp)
    z_std = z_var.sqrt()
    return self.reparameterize_Normal(z_mu, z_std), z_mu, z_std, dy

  def q_y(self, dy, py_mu, py_std):
    y_mu_hat, y_pre_sp = self.qy_convs_dy(dy).chunk(2, dim=1) # Split over channels
    y_var_hat = F.softplus(y_pre_sp)  # Using assigning var instead of std here for numerical stability

    py_var = py_std**2
    y_std = torch.sqrt((y_var_hat * py_var) / (y_var_hat + py_var))
    y_mu = (y_mu_hat * py_var + py_mu * y_var_hat) / (y_var_hat + py_var)

    return self.reparameterize_Normal(y_mu, y_std), y_mu, y_std

  def p_y(self, z):
    y_mu, y_pre_sp = self.py_deconvs(z).chunk(2, dim=1) # Split over channels
    y_var = F.softplus(y_pre_sp)
    y_std = y_var.sqrt()
    return self.reparameterize_Normal(y_mu, y_std), y_mu, y_std

  def p_x(self, y):
    x = self.px_deconvs(y)
    if self.likelihood_family == 'Bernoulli':
        x = torch.sigmoid(x) # last layer before output is sigmoid if we are using Bernoulli
    return x

  def sample_x(self, num=10, z=None):
    if z is None:
        z = torch.randn(num, self.c, self.z_HW, self.z_HW, device=self.device)

    y_hat, _, _ = self.p_y(z)
    fy = self.p_x(y_hat)

    if self.likelihood_family == "MoL":
        fy = sample_MoL(fy)

    return fy