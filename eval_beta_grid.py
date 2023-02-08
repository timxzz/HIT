import torch
torch.multiprocessing.set_sharing_strategy('file_system') # Refer to: https://github.com/pytorch/pytorch/issues/973
import torch.nn.functional as F

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import utils_eval

from multiprocessing import Pool
from itertools import repeat
from functools import partial

import tqdm



def cal_reconstruction_accs(args):
    batch_dir = './runs/' + args.run_batch_name
    file_name = "eval_results.pkl"
    eval_results = utils_eval.load_eval_results(batch_dir, file_name)
    runs_list = eval_results['runs_list']

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    print(device)

    beta_ys = eval_results['beta_ys']
    beta_zs = eval_results['beta_zs']

    num_beta_y = len(beta_ys)
    num_beta_z = len(beta_zs)

    recons_acc_M = np.full((num_beta_y, num_beta_z), 0.)
    recons_pred_entropy_M = np.full((num_beta_y, num_beta_z), -1.)

    num_runs = len(runs_list)
    num_runs_done = 0
    # Evalulate each model
    for single_run in runs_list:
        dataset = single_run['config'].dataset
        batch_size = single_run['config'].batch_size
        beta_y = single_run['config'].beta_y
        beta_z = single_run['config'].beta_z

        """
        results in the shape of:
        b_y = high  *   *
        b_y = low   *   *
        /    b_z = low   b_z = high
        """

        beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
        beta_z_idx = beta_zs.index(beta_z)

        recons_acc, recons_pred_entropy = utils_eval.get_recons_acc_for_one_model(single_run['config'], single_run['model_dir'], device)
        recons_acc_M[beta_y_idx][beta_z_idx] = recons_acc
        recons_pred_entropy_M[beta_y_idx][beta_z_idx] = recons_pred_entropy

        num_runs_done += 1
        print(f"Evaluation progress for reconstruction accuracy: {num_runs_done}/{num_runs}.", flush=True)


    eval_results['recons_acc_M'] = recons_acc_M
    eval_results['clsfr_acc'] = utils_eval.clsfr_test(dataset, batch_size, device)
    eval_results['recons_pred_entropy_M'] = recons_pred_entropy_M

    save_name = "eval_results.pkl"
    utils_eval.save_eval_results(batch_dir, eval_results, save_name)

    print("Finished saving reconstruction accuracies!")


def get_mi_and_cal_acc_for_a_runs(beta_ys, beta_zs, verbose_log, single_run):
    beta_y = single_run['config'].beta_y
    beta_z = single_run['config'].beta_z
    num_beta_y = len(beta_ys)
    num_beta_z = len(beta_zs)
    beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
    beta_z_idx = beta_zs.index(beta_z)

    mi_q_mc, lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc, mi_q_mc_y, lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y = utils_eval.get_MIq_and_cal_accs_for_one_model(single_run['run_dir'], verbose_log=verbose_log)

    return (beta_y_idx, beta_z_idx, mi_q_mc, lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc, mi_q_mc_y, lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y)

def get_mi_and_cal_acc_for_a_batch_of_runs(args):
    batch_dir = './runs/' + args.run_batch_name
    file_name = "eval_results_no_MI.pkl"
    eval_results = utils_eval.load_eval_results(batch_dir, file_name)
    runs_list = eval_results['runs_list']

    beta_ys = eval_results['beta_ys']
    beta_zs = eval_results['beta_zs']

    num_beta_y = len(beta_ys)
    num_beta_z = len(beta_zs)

    mi_q_mc_M = np.full((num_beta_y, num_beta_z), -1.)
    lr_acc_mean_M = np.full((num_beta_y, num_beta_z), 0.)
    lr_acc_std_M = np.full((num_beta_y, num_beta_z), -1.)
    svm_rbf_acc_M = np.full((num_beta_y, num_beta_z), 0.)
    svm_linear_acc_M = np.full((num_beta_y, num_beta_z), 0.)
    kNN_acc_M = np.full((num_beta_y, num_beta_z), 0.)

    mi_q_mc_y_M = np.full((num_beta_y, num_beta_z), -1.)
    lr_acc_mean_y_M = np.full((num_beta_y, num_beta_z), 0.)
    lr_acc_std_y_M = np.full((num_beta_y, num_beta_z), -1.)
    svm_rbf_acc_y_M = np.full((num_beta_y, num_beta_z), 0.)
    svm_linear_acc_y_M = np.full((num_beta_y, num_beta_z), 0.)
    kNN_acc_y_M = np.full((num_beta_y, num_beta_z), 0.)

    print(f"Multiprocessing with {args.cpu_n} CPUs")
    # with Pool(number_of_cores) as pool:
    #     results = pool.starmap(cal_mi_a_runs, zip(repeat(beta_ys), repeat(beta_zs), runs_list))
    verbose_log = True
    with Pool(processes=args.cpu_n) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(
                    partial(get_mi_and_cal_acc_for_a_runs, beta_ys, beta_zs, verbose_log),
                    runs_list
                ),
                total=(len(runs_list))
            )
        )

    for beta_y_idx, beta_z_idx, mi_q_mc, lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc, mi_q_mc_y, lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y in results:
        mi_q_mc_M[beta_y_idx][beta_z_idx] = mi_q_mc
        lr_acc_mean_M[beta_y_idx][beta_z_idx] = np.mean(lr_acc) if lr_acc is not None else None
        lr_acc_std_M[beta_y_idx][beta_z_idx] = np.std(lr_acc) if lr_acc is not None else None
        svm_rbf_acc_M[beta_y_idx][beta_z_idx] = np.mean(svm_rbf_acc) if svm_rbf_acc is not None else None
        svm_linear_acc_M[beta_y_idx][beta_z_idx] = np.mean(svm_linear_acc) if svm_linear_acc is not None else None
        kNN_acc_M[beta_y_idx][beta_z_idx] = np.mean(kNN_acc) if kNN_acc is not None else None
        mi_q_mc_y_M[beta_y_idx][beta_z_idx] = mi_q_mc_y
        lr_acc_mean_y_M[beta_y_idx][beta_z_idx] = np.mean(lr_acc_y) if lr_acc_y is not None else None
        lr_acc_std_y_M[beta_y_idx][beta_z_idx] = np.std(lr_acc_y) if lr_acc_y is not None else None
        svm_rbf_acc_y_M[beta_y_idx][beta_z_idx] = np.mean(svm_rbf_acc_y) if svm_rbf_acc_y is not None else None
        svm_linear_acc_y_M[beta_y_idx][beta_z_idx] = np.mean(svm_linear_acc_y) if svm_linear_acc_y is not None else None
        kNN_acc_y_M[beta_y_idx][beta_z_idx] = np.mean(kNN_acc_y) if kNN_acc_y is not None else None


    eval_results['mi_q_mc_M'] = mi_q_mc_M
    eval_results['lr_acc_mean_M'] = lr_acc_mean_M
    eval_results['lr_acc_std_M'] = lr_acc_std_M
    eval_results['svm_rbf_acc_M'] = svm_rbf_acc_M
    eval_results['svm_linear_acc_M'] = svm_linear_acc_M
    eval_results['kNN_acc_M'] = kNN_acc_M
    eval_results['mi_q_mc_y_M'] = mi_q_mc_y_M
    eval_results['lr_acc_mean_y_M'] = lr_acc_mean_y_M
    eval_results['lr_acc_std_y_M'] = lr_acc_std_y_M
    eval_results['svm_rbf_acc_y_M'] = svm_rbf_acc_y_M
    eval_results['svm_linear_acc_y_M'] = svm_linear_acc_y_M
    eval_results['kNN_acc_y_M'] = kNN_acc_y_M

    save_name = "eval_results.pkl"
    utils_eval.save_eval_results(batch_dir, eval_results, save_name)

    print("Finished saving MI!")


def get_pca_for_a_runs(verbose_log, single_run):
    beta_y = single_run['config'].beta_y
    beta_z = single_run['config'].beta_z

    run_results = utils_eval.load_eval_results(single_run['run_dir'], "qz_and_qy.pkl", verbose_log=verbose_log)

    qz_mu_all = run_results['qz_mu']
    qy_mu_all = run_results['qy_mu']

    N = qy_mu_all.shape[0]
    if len(qy_mu_all.shape) > 2:
        qy_mu_all = qy_mu_all.reshape(N, -1)

    pca = PCA(n_components=3)
    pca.fit(qy_mu_all)
    if verbose_log:
        print(f'-- Explained y variance ratio {pca.explained_variance_ratio_}', flush=True)
    pca_y_proj = pca.transform(qy_mu_all)

    tsne_y_proj = None
    # tsne = TSNE(2, verbose=1)
    # tsne_y_proj = tsne.fit_transform(qy_mu_all)

    return (beta_y, beta_z, pca_y_proj, tsne_y_proj, run_results['l'], run_results['K'])

def get_pca_for_a_batch_of_runs(args):
    batch_dir = './runs/' + args.run_batch_name
    file_name = "eval_results_no_MI.pkl"
    eval_results = utils_eval.load_eval_results(batch_dir, file_name)
    runs_list = eval_results['runs_list']

    beta_ys = eval_results['beta_ys']
    beta_zs = eval_results['beta_zs']

    pca_results = {}
    tsne_results = {}
    labels = {}

    for beta_y in beta_ys:
        pca_results[str(beta_y)] = {}
        tsne_results[str(beta_y)] = {}
        labels[str(beta_y)] = {}
        for beta_z in beta_zs:
            pca_results[str(beta_y)][str(beta_z)] = None
            tsne_results[str(beta_y)][str(beta_z)] = None
            labels[str(beta_y)][str(beta_z)] = None


    print(f"Multiprocessing with {args.cpu_n} CPUs")
    verbose_log = True
    with Pool(processes=args.cpu_n) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(
                    partial(get_pca_for_a_runs, verbose_log),
                    runs_list
                ),
                total=(len(runs_list))
            )
        )

    l = None
    K = None
    for beta_y, beta_z, pca_y_proj, tsne_y_proj, l, K in results:
        pca_results[str(beta_y)][str(beta_z)] = pca_y_proj
        tsne_results[str(beta_y)][str(beta_z)] = tsne_y_proj
        labels[str(beta_y)][str(beta_z)] = l

    all_result = {}
    all_result['pca_results'] = pca_results
    all_result['tsne_results'] = tsne_results
    all_result['labels'] = labels
    all_result['K'] = K

    save_name = "pca_results.pkl"
    utils_eval.save_eval_results(batch_dir, all_result, save_name)

    print("Finished saving PCA!")


def eval_a_batch_of_runs(args):
    eval_results = {}

    # run_batch_name="batch_test"
    run_batch_name = args.run_batch_name
    eval_results['run_batch_name'] = run_batch_name

    batch_dir = './runs/' + run_batch_name
    eval_results['batch_dir'] = batch_dir

    eval_results['used_classifier'] = args.with_classifier

    runs_list = utils_eval.get_runs_list_from_batch_dir(batch_dir)

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    print(device)

    beta_y_set = set()
    beta_z_set = set()
    for single_run in runs_list:
        beta_y_set.add(single_run['config'].beta_y)
        beta_z_set.add(single_run['config'].beta_z)

    beta_ys = sorted(beta_y_set)
    beta_zs = sorted(beta_z_set)

    eval_results['beta_ys'] = beta_ys
    eval_results['beta_zs'] = beta_zs

    num_beta_y = len(beta_ys)
    num_beta_z = len(beta_zs)

    loss_M = np.full((num_beta_y, num_beta_z), np.inf)
    ll_M = np.full((num_beta_y, num_beta_z), np.inf)
    kl_z_M = np.full((num_beta_y, num_beta_z), -1.)
    kl_y_M = np.full((num_beta_y, num_beta_z), -1.)
    mi_q_M = np.full((num_beta_y, num_beta_z), -1.)
    mi_p_M = np.full((num_beta_y, num_beta_z), -1.)
    mi_q_mksg_M = np.full((num_beta_y, num_beta_z), -1.)
    mi_q_mc_M = np.full((num_beta_y, num_beta_z), -1.)
    fid = None
    fid_M = np.full((num_beta_y, num_beta_z), -1.)
    psnr_M = np.full((num_beta_y, num_beta_z), -np.inf)
    lr_acc_mean_M = np.full((num_beta_y, num_beta_z), 0.)
    lr_acc_std_M = np.full((num_beta_y, num_beta_z), -1.)
    svm_rbf_acc_M = np.full((num_beta_y, num_beta_z), 0.)
    svm_linear_acc_M = np.full((num_beta_y, num_beta_z), 0.)
    kNN_acc_M = np.full((num_beta_y, num_beta_z), 0.)

    mi_q_mc_y_M = np.full((num_beta_y, num_beta_z), -1.)
    lr_acc_mean_y_M = np.full((num_beta_y, num_beta_z), 0.)
    lr_acc_std_y_M = np.full((num_beta_y, num_beta_z), -1.)
    svm_rbf_acc_y_M = np.full((num_beta_y, num_beta_z), 0.)
    svm_linear_acc_y_M = np.full((num_beta_y, num_beta_z), 0.)
    kNN_acc_y_M = np.full((num_beta_y, num_beta_z), 0.)

    is_mean_M = np.full((num_beta_y, num_beta_z), 0.)
    is_std_M = np.full((num_beta_y, num_beta_z), -1.)
    is_div_M = np.full((num_beta_y, num_beta_z), -1.)
    is_sharp_M = np.full((num_beta_y, num_beta_z), -1.)

    M_zl_list = []
    num_runs = len(runs_list)
    num_runs_done = 0
    # Evalulate each model
    for single_run in runs_list:
        M_zl_item = {}
        beta_y = single_run['config'].beta_y
        beta_z = single_run['config'].beta_z
        M_zl_item['beta_y'] = beta_y
        M_zl_item['beta_z'] = beta_z

        """
        results in the shape of:
        b_y = high  *   *
        b_y = low   *   *
        /    b_z = low   b_z = high
        """

        beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
        beta_z_idx = beta_zs.index(beta_z)

        loss, ll, kl_z, kl_y, mi_q, Mq_zl, mi_p, Mp_zl, mutual_info_q_mksg, fid, fid_score, psnr_score, mi_q_mc, lr_acc, svm_rbf_acc, svm_linear_acc, kNN_acc, mi_q_mc_y, lr_acc_y, svm_rbf_acc_y, svm_linear_acc_y, kNN_acc_y, is_mean, is_std, is_div, is_sharp = utils_eval.eval_one_model(single_run['config'], single_run['model_dir'], 
                                                single_run['run_dir'], device, 
                                                load_classifier=args.with_classifier, exclude_mi=args.exclude_mi,
                                                fid=fid,
                                                exclude_fid=args.exclude_fid)
        loss_M[beta_y_idx][beta_z_idx] = loss
        ll_M[beta_y_idx][beta_z_idx] = ll
        kl_z_M[beta_y_idx][beta_z_idx] = kl_z
        kl_y_M[beta_y_idx][beta_z_idx] = kl_y
        mi_q_M[beta_y_idx][beta_z_idx] = mi_q
        mi_p_M[beta_y_idx][beta_z_idx] = mi_p
        mi_q_mksg_M[beta_y_idx][beta_z_idx] = mutual_info_q_mksg
        mi_q_mc_M[beta_y_idx][beta_z_idx] = mi_q_mc
        fid_M[beta_y_idx][beta_z_idx] = fid_score
        psnr_M[beta_y_idx][beta_z_idx] = psnr_score
        lr_acc_mean_M[beta_y_idx][beta_z_idx] = np.mean(lr_acc) if lr_acc is not None else None
        lr_acc_std_M[beta_y_idx][beta_z_idx] = np.std(lr_acc) if lr_acc is not None else None
        svm_rbf_acc_M[beta_y_idx][beta_z_idx] = np.mean(svm_rbf_acc) if svm_rbf_acc is not None else None
        svm_linear_acc_M[beta_y_idx][beta_z_idx] = np.mean(svm_linear_acc) if svm_linear_acc is not None else None
        kNN_acc_M[beta_y_idx][beta_z_idx] = np.mean(kNN_acc) if kNN_acc is not None else None
        mi_q_mc_y_M[beta_y_idx][beta_z_idx] = mi_q_mc_y
        lr_acc_mean_y_M[beta_y_idx][beta_z_idx] = np.mean(lr_acc_y) if lr_acc_y is not None else None
        lr_acc_std_y_M[beta_y_idx][beta_z_idx] = np.std(lr_acc_y) if lr_acc_y is not None else None
        svm_rbf_acc_y_M[beta_y_idx][beta_z_idx] = np.mean(svm_rbf_acc_y) if svm_rbf_acc_y is not None else None
        svm_linear_acc_y_M[beta_y_idx][beta_z_idx] = np.mean(svm_linear_acc_y) if svm_linear_acc_y is not None else None
        kNN_acc_y_M[beta_y_idx][beta_z_idx] = np.mean(kNN_acc_y) if kNN_acc_y is not None else None
        is_mean_M[beta_y_idx][beta_z_idx] = is_mean
        is_std_M[beta_y_idx][beta_z_idx] = is_std
        is_div_M[beta_y_idx][beta_z_idx] = is_div
        is_sharp_M[beta_y_idx][beta_z_idx] = is_sharp

        M_zl_item['Mq_zl'] = Mq_zl
        M_zl_item['Mp_zl'] = Mp_zl

        M_zl_item['config'] = single_run['config']
        M_zl_item['model_dir'] = single_run['model_dir']
        M_zl_list.append(M_zl_item)

        num_runs_done += 1
        print(f"Evaluation progress: {num_runs_done}/{num_runs}.")

    eval_results['loss_M'] = loss_M
    eval_results['ll_M'] = ll_M
    eval_results['kl_z_M'] = kl_z_M
    eval_results['kl_y_M'] = kl_y_M
    eval_results['mi_q_M'] = mi_q_M
    eval_results['mi_p_M'] = mi_p_M
    eval_results['M_zl_list'] = M_zl_list
    eval_results['mi_q_mksg_M'] = mi_q_mksg_M
    eval_results['mi_q_mc_M'] = mi_q_mc_M
    eval_results['fid_M'] = fid_M
    eval_results['psnr_M'] = psnr_M
    eval_results['lr_acc_mean_M'] = lr_acc_mean_M
    eval_results['lr_acc_std_M'] = lr_acc_std_M
    eval_results['svm_rbf_acc_M'] = svm_rbf_acc_M
    eval_results['svm_linear_acc_M'] = svm_linear_acc_M
    eval_results['kNN_acc_M'] = kNN_acc_M
    eval_results['mi_q_mc_y_M'] = mi_q_mc_y_M
    eval_results['lr_acc_mean_y_M'] = lr_acc_mean_y_M
    eval_results['lr_acc_std_y_M'] = lr_acc_std_y_M
    eval_results['svm_rbf_acc_y_M'] = svm_rbf_acc_y_M
    eval_results['svm_linear_acc_y_M'] = svm_linear_acc_y_M
    eval_results['kNN_acc_y_M'] = kNN_acc_y_M
    eval_results['is_mean_M'] = is_mean_M
    eval_results['is_std_M'] = is_std_M
    eval_results['is_div_M'] = is_div_M
    eval_results['is_sharp_M'] = is_sharp_M
    eval_results['runs_list'] = runs_list

    save_name = "eval_results_no_MI.pkl"

    utils_eval.save_eval_results(batch_dir, eval_results, save_name)

    print("Finished saving!")



def main(args):
    if args.get_mi_cal_acc:
        print("Gather the mutual information and calculate accs given zs and ls")
        get_mi_and_cal_acc_for_a_batch_of_runs(args)
    elif args.pca:
        print("Get PCA for latents into 2 dims")
        get_pca_for_a_batch_of_runs(args)
    elif args.recons_acc:
        print("Calculate reconstruction accuracy only")
        cal_reconstruction_accs(args)
    else:
        print("Evaluate a batch of runs")
        eval_a_batch_of_runs(args)





if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--run_batch_name",
        help="Specified the name of the batch of runs to evaluate.")
    parser.add_argument(
        '--with_classifier', action='store_true', default=False,
        help='Calculating zs and ls with classifer for I_p(z;l)')
    parser.add_argument(
        '--exclude_mi', action='store_true', default=False,
        help='Evaluate models without calculate the mutual information')
    parser.add_argument(
        '--exclude_fid', action='store_true', default=False,
        help='Evaluate models without calculate the FID')
    parser.add_argument(
        '--get_mi_cal_acc', action='store_true', default=False,
        help='Retrieve the mutual information and calculate Accs only')
    parser.add_argument(
        '--pca', action='store_true', default=False,
        help='PCA latent for 2 dims')
    parser.add_argument(
        '--recons_acc', action='store_true', default=False,
        help='Calculate reconstruction accuracy only')
    parser.add_argument(
        "--cpu_n", type=int, default=1,
        help="Number of cpus")

    args = parser.parse_args()

    main(args)