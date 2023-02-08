import os
import io
import json
import pickle
import gc
import time

import scipy.spatial as ss
from scipy.special import digamma
from scipy.stats import entropy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from math import log
import numpy as np

import torch
import torch.nn.functional as F
from torch import distributions as D

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import PeakSignalNoiseRatio

import dataloader
from models import HVAE_Fixed_z, HVAE_Conv_z
from beta_models import BetaVAE
import models_lvae
from mnist_classifier import MNIST_CNN
from svhn_classifier import ResNet18
from cifar_classifier import DenseNet121

class Config(object):
    """
    Load the json format config as an object
    """
    def __init__(self, file_object):
        self.__dict__ = json.load(file_object)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_runs_list_from_batch_dir(batch_dir):
    """
    Get a list of config and trained model from the batch of runs
    """
    runs_list = []
    for subdir, dirs, files in os.walk(batch_dir):
        single_run = {}
        for file in files:
            if file == "config.json":
                config_dir = os.path.join(subdir, file)
                print("Loading config from: " + config_dir)
                with open(config_dir, 'r') as f:
                    config = Config(f)
                single_run["config"] = config
            elif file == "vae.pth":
                model_dir = os.path.join(subdir, file)
                single_run["model_dir"] = model_dir
        if len(single_run) != 0:
            single_run["run_dir"] = subdir
            runs_list.append(single_run)

    return runs_list

def get_single_run(run_path):
    """
    Get single run config and trained model from run path
    """
    single_run = {}

    config_dir = os.path.join(run_path, "config.json")
    print("Loading config from: " + config_dir)
    with open(config_dir, 'r') as f:
        config = Config(f)
    single_run["config"] = config
    
    model_dir = os.path.join(run_path,  "vae.pth")
    single_run["model_dir"] = model_dir

    return single_run

def load_model(config, model_dir, device, hard_gs=False):
    if hasattr(config, 'use_conv_z'):
        use_conv_z = config.use_conv_z
    else:
        use_conv_z = False

    # Backward compatible
    if hasattr(config, 'vae_type'):
        vae_type = config.vae_type
    else:
        vae_type = "HVAE"

    if vae_type == "BVAE":
        model = BetaVAE(device, px_y_family_ll=config.px_y_family_ll,
                    sigma=config.sigma, 
                    dataset=config.dataset, num_c=config.conv_channels,
                    beta_y=config.beta_y)
    elif config.dataset == "ImageNet" or use_conv_z:
        if vae_type == "HVAE":
            model = HVAE_Conv_z(device, qz_family=config.qz_family, px_y_family_ll=config.px_y_family_ll, 
                            dataset=config.dataset, num_c=config.conv_channels,
                            beta_y=config.beta_y, beta_z=config.beta_z,
                            hard_gs=hard_gs)
        elif vae_type == "LVAE":
            model = models_lvae.LVAE_Conv_z(device, qz_family=config.qz_family, px_y_family_ll=config.px_y_family_ll, 
                            dataset=config.dataset, num_c=config.conv_channels,
                            beta_y=config.beta_y, beta_z=config.beta_z,
                            hard_gs=hard_gs)
    else:
        if vae_type == "HVAE":
            model = HVAE_Fixed_z(device, qz_family=config.qz_family, px_y_family_ll=config.px_y_family_ll, 
                            dataset=config.dataset, num_c=config.conv_channels, z_dims=config.z_dims,
                            linear_y_with_dims=config.linear_y_with_dims,
                            beta_y=config.beta_y, beta_z=config.beta_z,
                            hard_gs=hard_gs)
        elif vae_type == "LVAE":
            model = models_lvae.LVAE_Fixed_z(device, qz_family=config.qz_family, px_y_family_ll=config.px_y_family_ll, 
                            dataset=config.dataset, num_c=config.conv_channels, z_dims=config.z_dims,
                            linear_y_with_dims=config.linear_y_with_dims,
                            beta_y=config.beta_y, beta_z=config.beta_z,
                            hard_gs=hard_gs)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def load_mnist_clsfer(model_dir, device):
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    model.eval()

    return model

def load_svhn_clsfer(model_dir, device):
    model = ResNet18()
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    model.eval()

    return model

def load_cifar10_clsfer(model_dir, device):
    model = DenseNet121() # Acc: 95%
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def save_eval_results(out_dir, eval_results, save_name):
    # Save evaluation results
    file_dir = os.path.join(out_dir, save_name)
    print("Saving evaluation results to: " + file_dir)
    with open(file_dir, 'wb') as f:
        pickle.dump(eval_results, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_eval_results(out_dir, file_name, verbose_log=True):
    # load evaluation results
    file_dir = os.path.join(out_dir, file_name)
    if not os.path.isfile(file_dir):
        if verbose_log:
            print(f"File: {file_dir} doesn't exist")
        return None

    if verbose_log:
        print("Loading evaluation results from: " + file_dir)
    with open(file_dir, 'rb') as f:
        # eval_results = pickle.load(f)
        eval_results = CPU_Unpickler(f).load()
    return eval_results



def update_M_zl(M_zl, z, l):
    """
    Update teh count matrix M_{z,l}
    Both z and l are one_hot
    """
    assert z.shape == l.shape
    assert len(z.shape) == 2

    # Sum of outer product over batch dim in einsum notation
    M_zl += torch.einsum('nz,nl->zl', [z,l])
    return M_zl


def MI_from_discrete_z_l(M_zl):
    """
    Given the count matrix M_zl to calculate the 
    prob/freq matrix p(z,l)
    Then calculate the mutual information I(z;l)
    """
    
    # Avoid dividing zeros
    M_zl += 1e-5

    p_zl = M_zl / M_zl.sum()
    p_z = p_zl.sum(dim=1)
    p_l = p_zl.sum(dim=0)
    pz_py = torch.einsum('z,l->zl', [p_z,p_l])

    I = torch.sum(p_zl * torch.log(p_zl / pz_py))

    return I


def MI_by_classifying_samples(model, classifier, K, device):
    """
    Sampling then classifying
    """
    num_sample_pcpb = 100
    num_batches = 10
    Mp_zl = torch.zeros(K,K, device=device)

    for _ in range(num_batches):
        z_oh = F.one_hot(torch.arange(0, K), num_classes=model.z_dims).float().to(device)
        z_oh = z_oh.repeat(1, num_sample_pcpb).view(-1, model.z_dims) # Repeat for sampling y
        sample = model.sample_x(z=z_oh)
        logprob = classifier(sample)
        l_pred = logprob.argmax(dim=1) 
        l_pred_oh = F.one_hot(l_pred, num_classes=K).type(torch.FloatTensor).to(device)


        Mp_zl = update_M_zl(Mp_zl, z_oh, l_pred_oh)

    mutual_info_p = MI_from_discrete_z_l(Mp_zl)
    Mp_zl = Mp_zl.detach().cpu().numpy()

    return mutual_info_p, Mp_zl


def eval_expect_logprob_ratio_for_z_j(q_list, z_j, l):
    num_z = z_j.shape[0]
    z_dims = len(z_j.shape[1:])
    alpha_l = torch.zeros(num_z, len(q_list))
    i = 0
    num_q = 0
    num_q_l = 0
    for q_li_params in q_list:
        num_q_li = q_li_params.shape[0]
        if z_dims == 1:
            q_li = D.normal.Normal(q_li_params[:,0,:].repeat(num_z, 1), q_li_params[:,1,:].repeat(num_z, 1))
            q_li = D.independent.Independent(q_li, 1)     # Set the last dim for one event
        elif z_dims == 3:
            q_li = D.normal.Normal(q_li_params[:,0,:].repeat(num_z, 1,1,1), q_li_params[:,1,:].repeat(num_z, 1,1,1))
            q_li = D.independent.Independent(q_li, 3)     # Set the last dim for one event

        z_j_li = z_j.repeat_interleave(num_q_li, dim=0)

        alpha_l[:,i] = q_li.log_prob(z_j_li).reshape(num_z, num_q_li).logsumexp(dim=-1)

        num_q += num_q_li
        if i == l:
            num_q_l += num_q_li
        i += 1

    z_j_lpr = alpha_l[:,l] - alpha_l.logsumexp(dim=-1) - np.log(num_q_l / num_q)

    return z_j_lpr


def MI_given_dist_q_by_mc(q_list, z_batch_size=500, verbose_log=True):
    l = 0
    num_data = 0
    mi_q_sum = 0
    if verbose_log:
        print('Estimating MI given q by MC')
    for q_l_params in q_list:
        num_data += q_l_params.shape[0]
        q_l = D.normal.Normal(q_l_params[:,0,:], q_l_params[:,1,:])
        q_l = D.independent.Independent(q_l, 1)     # Set the last dim for one event

        z_j = q_l.sample()

        # Do it in batch
        z_size = z_j.shape[0]
        b_start = 0
        b_end = b_start + z_batch_size
        while b_end < z_size:
            expect_z_j = eval_expect_logprob_ratio_for_z_j(q_list, z_j[b_start:b_end], l)
            mi_q_sum += expect_z_j.sum()
            b_start = b_end
            b_end += z_batch_size
            # if not expect_z_j.is_cuda and verbose_log:
            if verbose_log:
                print('.', end='', flush=True)
        expect_z_j = eval_expect_logprob_ratio_for_z_j(q_list, z_j[b_start:], l)
        mi_q_sum += expect_z_j.sum()

        l += 1
    if verbose_log:
        print()

    return mi_q_sum / num_data


def qz_all_to_q_l_list(qz_mu_all, qz_std_all, l_all, K, in_tensor=False):

    if not in_tensor:
        use_gpu = True
        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"MI by mc on {device}")

        qz_mu_all = torch.from_numpy(qz_mu_all).to(device)
        qz_std_all = torch.from_numpy(qz_std_all).to(device)
        l_all = torch.from_numpy(l_all).to(device)


    q_list = []
    for l in range(K):
        q_mu_l = qz_mu_all[l_all == l]
        q_std_l = qz_std_all[l_all == l]
        q_list.append(torch.cat((q_mu_l[:,None,:], q_std_l[:,None,:]), dim=1))

    return q_list


def zs_ls_by_classifying_samples_GS(model, classifier, l_dim, device):
    """
    For qz being GumbelSoftmax.
    Sampling then classifying
    """
    num_sample_pcpb = 100
    num_batches = 10

    zs = None
    ls = None

    for _ in range(num_batches):
        z_oh = F.one_hot(torch.arange(0, model.z_dims), num_classes=model.z_dims).float().to(device)
        z_oh = z_oh.repeat(1, num_sample_pcpb).view(-1, model.z_dims) # Repeat for sampling y
        sample = model.sample_x(z=z_oh)
        logprob = classifier(sample)
        l_pred = logprob.argmax(dim=1) 
        l_pred_oh = F.one_hot(l_pred, num_classes=l_dim).type(torch.FloatTensor).to(device)

        if zs is None:
            zs = z_oh
            ls = l_pred_oh
        else:
            zs = torch.cat((zs, z_oh), dim=0)
            ls = torch.cat((ls, l_pred_oh), dim=0)
    return zs, ls

def zs_ls_by_classifying_samples_N(model, classifier, l_dim, device):
    """
    For qz being Diagonal Gaussian.
    Sampling then classifying
    """
    num_sample_pcpb = 200
    num_batches = 50

    zs = None
    ls = None

    for _ in range(num_batches):
        if model.z_by_Conv:
            z_rdm = torch.randn(num_sample_pcpb, model.c, model.z_HW, model.z_HW, device=device)
        else:
            z_rdm = torch.randn(num_sample_pcpb, model.z_dims, device=device)
        sample = model.sample_x(z=z_rdm)
        logprob = classifier(sample)
        l_pred = logprob.argmax(dim=1) 
        l_pred_oh = F.one_hot(l_pred, num_classes=l_dim).type(torch.FloatTensor).to(device)

        if zs is None:
            zs = z_rdm
            ls = l_pred_oh
        else:
            zs = torch.cat((zs, z_rdm), dim=0)
            ls = torch.cat((ls, l_pred_oh), dim=0)
    return zs, ls


def inception_score_on_samples_N(model, classifier, l_dim, device, splits=5, use_betaVAE=False):
    """
    For qz being Diagonal Gaussian.
    Sampling then classifying
    Refer: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    """
    num_sample_pcpb = 200
    num_batches = 50
    N = num_batches * num_sample_pcpb

    pl_x = None

    for _ in range(num_batches):
        if use_betaVAE:
            y_rdm = torch.randn(num_sample_pcpb, model.c, model.y_HW, model.y_HW, device=device)
            sample = model.sample_x(y=y_rdm)
        else:
            if model.z_by_Conv:
                z_rdm = torch.randn(num_sample_pcpb, model.c, model.z_HW, model.z_HW, device=device)
            else:
                z_rdm = torch.randn(num_sample_pcpb, model.z_dims, device=device)
            sample = model.sample_x(z=z_rdm)

        logprob = classifier(sample)

        if pl_x is None:
            pl_x = torch.exp(logprob).data.cpu().numpy()
        else:
            pl_x = np.concatenate((pl_x, torch.exp(logprob).data.cpu().numpy()), axis=0)

    assert N == pl_x.shape[0]

    # Now compute the mean kl-div
    split_scores = []
    split_div = []
    split_sharp = []

    for k in range(splits):
        part = pl_x[k * (N // splits): (k+1) * (N // splits), :]
        pl = np.mean(part, axis=0)
        scores = []
        Hl = entropy(pl)
        Hl_x = []
        for i in range(part.shape[0]):
            plx = part[i, :]
            scores.append(entropy(plx, pl))
            Hl_x.append(entropy(plx))
        split_scores.append(np.exp(np.mean(scores)))
        split_div.append(np.exp(Hl))
        split_sharp.append(np.exp(- np.mean(Hl_x)))

    return np.mean(split_scores), np.std(split_scores), np.mean(split_div), np.mean(split_sharp)


def Mixed_KSG(x,y,k=5):
    '''
        #Copyright Weihao Gao, UIUC

        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using *Mixed-KSG* mutual information estimator
        Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
        y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
        k: k-nearest neighbor parameter
        Output: one number of I(X;Y)

        Refer to: https://github.com/wgao9/mixed_KSG/blob/master/mixed.py
        Paper: http://arxiv.org/abs/1709.06212
    '''

    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])    
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
    return ans


def cal_mi_ksg_one_model(run_dir, verbose_log=True, Iq_or_Ip="Iq"):
    """
    Calculate mutual information given the save zs and ls of one model using KSG
    """
    if Iq_or_Ip=="Iq":
        zs_and_ls = load_eval_results(run_dir, "zs_and_ls.pkl", verbose_log=verbose_log)
    elif Iq_or_Ip=="Ip":
        zs_and_ls = load_eval_results(run_dir, "zs_and_ls_gen.pkl", verbose_log=verbose_log)

    # If the file doesn't exist, skip the calculation
    if zs_and_ls is None:
        return -1

    zs = zs_and_ls['zs']
    ls = zs_and_ls['ls']
    mutual_info_ksg = Mixed_KSG(zs,ls)

    if verbose_log:
        print(f"Finish MI {Iq_or_Ip} for {run_dir}")

    return mutual_info_ksg


def fid_cast_dtype(data):
    """
    update data format into the FID method acceptable input
    """
    channel_sizes = data.shape[1]

    if channel_sizes == 1:
        data = data.repeat(1,3,1,1)

    return (data * 255).to(dtype=torch.uint8)


def fid_eval_samples(fid, model):

    num_sample_pcpb = 200
    num_batches = 50

    with torch.no_grad():
        for _ in range(num_batches):
            samples = model.sample_x(num=num_sample_pcpb)
            fid.update(fid_cast_dtype(samples), real=False)
    
    return fid.compute()


def simple_clf_acc(zs, y, model_type, verbose_log=True):
    """
    Use different simple classifiers to evaluate the quality of representation learning
    Over 5 random seeds
    """
    num_data = zs.shape[0]
    X = zs.reshape(num_data, -1)

    test_size = int(num_data * 0.2)
    idx_rand = np.random.RandomState(seed=42).permutation(num_data)

    y = y[idx_rand]
    X = X[idx_rand]

    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    seeds = [0]
    if model_type == "logistic":
        seeds = list(range(5))

    acc = []
    start = time.time()
    for seed in seeds:
        if model_type == "logistic":
            if verbose_log:
                print("Evaluating using logistic regression")
            clf = LogisticRegression(random_state=seed, solver='saga', multi_class='multinomial', n_jobs=-1)
        elif model_type == "svm-rbf":
            if verbose_log:
                print("Evaluating using SVM with RBF")
            clf = SVC(kernel='rbf')
        elif model_type == "svm-linear":
            if verbose_log:
                print("Evaluating using SVM with Linear")
            clf = SVC(kernel='linear')
        elif model_type == "kNN":
            clf = KNeighborsClassifier()
        else:
            raise ValueError('unknown model type: {}'.format(probe))

        clf.fit(X_train, y_train)
        acc.append(clf.score(X_test, y_test))
        gc.collect()

    if verbose_log:
        print(f"Time for training {model_type}: {time.time() - start:.2f}s")

    return acc


def cal_MIq_by_mc(q_mu_all, q_std_all, l_all, q_family, K, z_batch_size=500, verbose_log=True, in_tensor=True):
    # MI_q by MC
    mi_q_mc = None
    if q_family == "DiagonalGaussian":
        start = time.time()
        q_list = qz_all_to_q_l_list(q_mu_all, q_std_all, l_all, K, in_tensor=in_tensor)
        mi_q_mc = MI_given_dist_q_by_mc(q_list, z_batch_size, verbose_log=verbose_log)
    if verbose_log:
        print(f"Time for MI_q_mc: {time.time() - start:.2f}s", flush=True)
        print(f"MI_q_mc: {mi_q_mc}", flush=True)

    return mi_q_mc

def cal_accs(q_mu_all, l_all, q_family, verbose_log=True):

    # LR acc
    lr_acc = None
    if q_family == "DiagonalGaussian":
        lr_acc = simple_clf_acc(q_mu_all, l_all, model_type="logistic", verbose_log=verbose_log)
    if verbose_log:
        print(f"LR_acc: {lr_acc}", flush=True)

    # SVM-rbf acc
    svm_rbf_acc = None
    if q_family == "DiagonalGaussian":
        svm_rbf_acc = simple_clf_acc(q_mu_all, l_all, model_type="svm-rbf", verbose_log=verbose_log)
    if verbose_log:
        print(f"SVM-RBF_acc: {svm_rbf_acc}", flush=True)

    # SVM-linear acc
    svm_linear_acc = None
    if q_family == "DiagonalGaussian":
        svm_linear_acc = simple_clf_acc(q_mu_all, l_all, model_type="svm-linear", verbose_log=verbose_log)
    if verbose_log:
        print(f"SVM_Linear_acc: {svm_linear_acc}", flush=True)

    # kNN acc
    kNN_acc = None
    if q_family == "DiagonalGaussian":
        kNN_acc = simple_clf_acc(q_mu_all, l_all, model_type="kNN", verbose_log=verbose_log)
    if verbose_log:
        print(f"kNN_acc: {kNN_acc}", flush=True)

    return lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc


def get_MIq_and_cal_accs_for_one_model(run_dir, verbose_log=True):
    """
    Calculate mutual information q given the save q distribution and label of one model
    Also calculate the accuracies of classifier given latents and labels
    """
    qz_and_qy = load_eval_results(run_dir, "qz_and_qy.pkl", verbose_log=verbose_log)
    mi_by_mc = load_eval_results(run_dir, "mi_by_mc.pkl", verbose_log=verbose_log)

    # If the file doesn't exist, skip the calculation
    if mi_by_mc is None:
        mi_q_mc = None
        mi_q_mc_y = None
    else:
        mi_q_mc = mi_by_mc['mi_q_mc']
        mi_q_mc_y = mi_by_mc['mi_q_mc_y']

    qz_mu_all = qz_and_qy['qz_mu']
    qz_family = qz_and_qy['qz_family']
    qy_mu_all = qz_and_qy['qy_mu']
    l_all = qz_and_qy['l']
    K = qz_and_qy['K']

    if verbose_log:
        print("cal accs for z")
    lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc = cal_accs(qz_mu_all, l_all, qz_family, verbose_log=verbose_log)
    if verbose_log:
        print("cal accs for y")
    lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y = cal_accs(qy_mu_all, l_all, "DiagonalGaussian", verbose_log=verbose_log)

    return mi_q_mc, lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc, mi_q_mc_y, lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y


def eval_one_model(config, model_dir, run_dir, device, load_classifier=False, exclude_mi=False, fid=None, exclude_fid=False):
    """
    Evaluate a specific model with given config.
    """
    gs_temp = 0.4

    _, test_dataloader = dataloader.load_data(config.dataset, config.batch_size)

    # hard_gs set to True for sample onehot z
    model = load_model(config, model_dir, device, hard_gs=True)

    test_loss_avg = 0
    ll_avg = 0
    kl_z_avg = 0
    kl_y_avg = 0
    num_batches = 0

    K = 10 # 10 class
    if config.qz_family == "GumbelSoftmax":
        Mq_zl = torch.zeros(model.z_dims,K, device=device)
    zs = None
    ls = None

    # For FID, initialise if not given, else reset fake features
    if not exclude_fid:
        if fid is None:
            fid_given = False
            fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
        else:
            fid_given = True
            fid.reset()

    # For reconstructions
    psnr = PeakSignalNoiseRatio().to(device)

    # For cal MI_q with MC:
    qz_mu_all = []
    qz_std_all = []
    qy_mu_all = []
    qy_std_all = []
    l_all = []

    for test_images, test_labels_int in test_dataloader:
        
        l_all.append(test_labels_int)

        test_images = test_images.to(device)
        test_labels = F.one_hot(test_labels_int, num_classes=K).type(torch.FloatTensor)
        test_labels = test_labels.to(device)

        # For FID real data
        if not exclude_fid and not fid_given:
            fid.update(fid_cast_dtype(test_images), real=True)

        with torch.no_grad():
            if config.qz_family == "GumbelSoftmax":
                loss, ll, kl_z, kl_y, \
                    qz_pi, _,_,_,_, z = model(test_images, temp=gs_temp)
                Mq_zl = update_M_zl(Mq_zl, z, test_labels)
            elif config.qz_family == "DiagonalGaussian":
                loss, ll, kl_z, kl_y, \
                    qz_mu, qz_std, qy_mu, qy_std, py_mu, py_std, z = model(test_images)
                qz_mu_all.append(qz_mu)
                qz_std_all.append(qz_std)
                qy_mu_all.append(qy_mu)
                qy_std_all.append(qy_std)

            # For evaluating reconstructions
            x_recons = model.reconstruction(test_images)
            psnr.update(x_recons, test_images)


        if zs is None:
            zs = z
            ls = test_labels
        else:
            zs = torch.cat((zs, z), dim=0)
            ls = torch.cat((ls, test_labels), dim=0)
        
        test_loss_avg += loss.item()
        ll_avg += ll.item()
        kl_z_avg += kl_z.item()
        kl_y_avg += kl_y.item()
        num_batches += 1
        
    # Avg over test set:
    test_loss_avg /= num_batches
    ll_avg /= num_batches
    kl_z_avg /= num_batches
    kl_y_avg /= num_batches
    if config.qz_family == "GumbelSoftmax":
        mutual_info_q = MI_from_discrete_z_l(Mq_zl)
        Mq_zl = Mq_zl.detach().cpu().numpy()
    else:
        mutual_info_q = None
        Mq_zl = None

    # Concate qz, qy and l list
    qz_mu_all_tensor = torch.cat(qz_mu_all, dim=0)
    qz_std_all_tensor = torch.cat(qz_std_all, dim=0)
    qy_mu_all_tensor = torch.cat(qy_mu_all, dim=0)
    qy_std_all_tensor = torch.cat(qy_std_all, dim=0)
    l_all_tensor = torch.cat(l_all, dim=0)
    qz_mu_all_numpy = qz_mu_all_tensor.cpu().numpy()
    qz_std_all_numpy = qz_std_all_tensor.cpu().numpy()
    qy_mu_all_numpy = qy_mu_all_tensor.cpu().numpy()
    qy_std_all_numpy = qy_std_all_tensor.cpu().numpy()
    l_all_numpy = l_all_tensor.cpu().numpy()

    # mutual_info_q with mixed_ksg
    zs = torch.flatten(zs, start_dim=1)
    zs = zs.detach().cpu().numpy()
    ls = ls.detach().cpu().numpy()
    
    mutual_info_q_ksg = None

    zs_and_ls = {}
    zs_and_ls['zs'] = zs
    zs_and_ls['ls'] = ls

    qz_and_qy = {}
    qz_and_qy['qz_mu'] = qz_mu_all_numpy
    qz_and_qy['qz_std'] = qz_std_all_numpy
    qz_and_qy['qz_family'] = config.qz_family
    qz_and_qy['qy_mu'] = qy_mu_all_numpy
    qz_and_qy['qy_std'] = qy_std_all_numpy
    qz_and_qy['l'] = l_all_numpy
    qz_and_qy['K'] = K

    save_eval_results(run_dir, zs_and_ls, "zs_and_ls.pkl")
    save_eval_results(run_dir, qz_and_qy, "qz_and_qy.pkl")
    


    # Calculate MI_q and different acc for the mapping of latent to label
    mi_q_mc, lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc = None, None, None, None, None
    mi_q_mc_y, lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y = None, None, None, None, None
    if not exclude_mi:
        mi_q_mc = cal_MIq_by_mc(qz_mu_all_tensor, qz_std_all_tensor, l_all_tensor, config.qz_family, K)
        mi_q_mc_y = cal_MIq_by_mc(qy_mu_all_tensor, qy_std_all_tensor, l_all_tensor, "DiagonalGaussian", K, z_batch_size=150)

    # Use classifier to eval I_p (z;l) and Inception Score
    zs_and_ls_gen = {}
    is_mean = None
    is_std = None
    clsfr = None
    if load_classifier:
        if config.dataset == "BinaryMNIST":
            clsfr = load_mnist_clsfer(model_dir="./pretrained/mnist_cnn.pt", device=device)
        elif config.dataset == "SVHN":
            clsfr = load_svhn_clsfer(model_dir="./pretrained/svhn_resnet.pth", device=device)
        elif config.dataset == "CIFAR10":
            clsfr = load_cifar10_clsfer(model_dir="./pretrained/cifar_densenet.pth", device=device)

        if config.qz_family == "GumbelSoftmax":
            zs, ls = zs_ls_by_classifying_samples_GS(model, clsfr, K, device)
        elif config.qz_family == "DiagonalGaussian":
            zs, ls = zs_ls_by_classifying_samples_N(model, clsfr, K, device)
            # Calculate IS score
            is_mean, is_std, is_div, is_sharp = inception_score_on_samples_N(model, clsfr, K, device)

        # mutual_info_q with mixed_ksg
        zs = torch.flatten(zs, start_dim=1)
        zs = zs.detach().cpu().numpy()
        ls = ls.detach().cpu().numpy()
        
        zs_and_ls_gen['zs'] = zs
        zs_and_ls_gen['ls'] = ls
        save_eval_results(run_dir, zs_and_ls_gen, "zs_and_ls_gen.pkl")

    print(f"IS mean: {is_mean}, std: {is_std}, Div: {is_div}, Sharp: {is_sharp}")

    if config.qz_family == "GumbelSoftmax" and clsfr is not None:
        mutual_info_p, Mp_zl = MI_by_classifying_samples(model, clsfr, K, device)
    else:
        mutual_info_p = None
        Mp_zl = None


    # Evaluating sampling quality with FID -> range >= 0, 0 corresponds to high quality
    fid_score = None
    if not exclude_fid:
        fid_score = fid_eval_samples(fid, model)
        print(f"FID: {fid_score}")

    # # Evaluating reconstruction quality with MS_SSIM -> range [0,1], 1 corresponds to high quality
    # ms_ssim_score = ms_ssim.compute()
    # print(f"MS-SSIM: {ms_ssim_score}")

    # Evaluating reconstruction quality with PSNR ->  high value corresponds to high quality
    psnr_score = psnr.compute()
    print(f"PSNR: {psnr_score}")
    
    return test_loss_avg, ll_avg, kl_z_avg, kl_y_avg, mutual_info_q, Mq_zl, mutual_info_p, Mp_zl, mutual_info_q_ksg, fid, fid_score, psnr_score, mi_q_mc, lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc, mi_q_mc_y, lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y, is_mean, is_std, is_div, is_sharp


def clsfr_test(dataset, batch_size, device):
    if dataset == "BinaryMNIST":
        clsfr = load_mnist_clsfer(model_dir="./pretrained/mnist_cnn.pt", device=device)
    elif dataset == "SVHN":
        clsfr = load_svhn_clsfer(model_dir="./pretrained/svhn_resnet.pth", device=device)
    elif dataset == "CIFAR10":
        clsfr = load_cifar10_clsfer(model_dir="./pretrained/cifar_densenet.pth", device=device)

    _, test_dataloader = dataloader.load_data(dataset, batch_size)

    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            logprob = clsfr(data)
            pred = logprob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nClassifier Test set: Accuracy: {}/{} ({})\n'.format(
        correct, len(test_dataloader.dataset),
        correct / len(test_dataloader.dataset)))
    
    return correct / len(test_dataloader.dataset)


def get_recons_acc_for_one_model(config, model_dir, device):

    _, test_dataloader = dataloader.load_data(config.dataset, config.batch_size)

    model = load_model(config, model_dir, device, hard_gs=True)

    correct = 0
    pl_x = None

    if config.dataset == "BinaryMNIST":
        clsfr = load_mnist_clsfer(model_dir="./pretrained/mnist_cnn.pt", device=device)
    elif config.dataset == "SVHN":
        clsfr = load_svhn_clsfer(model_dir="./pretrained/svhn_resnet.pth", device=device)
    elif config.dataset == "CIFAR10":
        clsfr = load_cifar10_clsfer(model_dir="./pretrained/cifar_densenet.pth", device=device)

    for test_images, test_labels in test_dataloader:
        
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        with torch.no_grad():
            # For evaluating reconstructions
            x_recons = model.reconstruction(test_images)
            logprob = clsfr(x_recons)
            pred = logprob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(test_labels.view_as(pred)).sum().item()

            if pl_x is None:
                pl_x = torch.exp(logprob).data.cpu().numpy()
            else:
                pl_x = np.concatenate((pl_x, torch.exp(logprob).data.cpu().numpy()), axis=0)
        
    accuracy = correct / len(test_dataloader.dataset)

    print('\nReconstruction classification accuracy: {}\n'.format(accuracy))

    Hl_x = []
    for i in range(pl_x.shape[0]):
        plx = pl_x[i, :]
        Hl_x.append(entropy(plx))
    # sharpness = np.exp(- np.mean(Hl_x))
    recons_pred_entropy = np.mean(Hl_x)
    print('Reconstruction sharpness: {}\n'.format(recons_pred_entropy))

    return accuracy, recons_pred_entropy