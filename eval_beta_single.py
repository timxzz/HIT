import torch
import torch.nn.functional as F

import numpy as np

import utils_eval


def cal_mi_for_a_runs(args):
    run_dir = args.run_path
    file_name = "qz_and_qy.pkl"
    run_results = utils_eval.load_eval_results(run_dir, file_name)

    qz_mu_all = run_results['qz_mu']
    qz_std_all = run_results['qz_std']
    qz_family = run_results['qz_family']
    qy_mu_all = run_results['qy_mu']
    qy_std_all = run_results['qy_std']
    l_all = run_results['l']
    K = run_results['K']

    mi_q_mc = utils_eval.cal_MIq_by_mc(qz_mu_all, qz_std_all, l_all, qz_family, K, in_tensor=False)

    mi_by_mc = {}
    mi_by_mc['mi_q_mc'] = mi_q_mc
    mi_by_mc['mi_q_mc_y'] = None    # Cannot approximate this as in z
    save_name = "mi_by_mc.pkl"
    utils_eval.save_eval_results(run_dir, mi_by_mc, save_name)

    print("Finished saving MI!")



def main(args):
    print("Calculate the mutual information between zs and ls given dist")
    cal_mi_for_a_runs(args)




if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--run_path",
        help="Specified the path of a run to evaluate.")

    args = parser.parse_args()

    main(args)