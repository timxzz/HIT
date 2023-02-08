import matplotlib
from matplotlib import cm
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
import colorsys
import seaborn as sns
import numpy as np
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import entropy
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

import utils_eval


def plot_heatmap(data_matrix, title, beta_ys, beta_zs, cmap=None, ax_given=None):
    xticklabels = beta_zs
    yticklabels = list(reversed(beta_ys))
    xticklabels = [ round(t, 2) for t in xticklabels ]
    yticklabels = [ round(t, 2) for t in yticklabels ]

    if ax_given is None:
        ax = sns.heatmap(data_matrix, cmap=cmap)
    else:
        ax = sns.heatmap(data_matrix, cmap=cmap, ax=ax_given)

    ax.set_xticks(range(len(beta_zs)))
    ax.set_xticklabels(xticklabels, rotation=60)
    ax.set_yticks(range(len(beta_ys)))
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)

    if ax_given is None:
        plt.show()

def plot_heatmap_all(results):
    ll = results['ll_M']
    mi_q = results['mi_q_M']
    kl_z = results['kl_z_M']
    kl_y = results['kl_y_M']

    fig, ax =plt.subplots(2,2, sharex=True)

    plot_heatmap(ll, title="LogLikelihood: <log p(x|y)>_q", beta_ys=results['beta_ys'], beta_zs=results['beta_zs'], cmap='BuPu', ax_given=ax[0,0])
    plot_heatmap(mi_q, title="I_q(z;l)", beta_ys=results['beta_ys'], beta_zs=results['beta_zs'], cmap='YlGnBu', ax_given=ax[0,1])
    plot_heatmap(kl_z, title="KL[q(z|x) || p(z)]", beta_ys=results['beta_ys'], beta_zs=results['beta_zs'], ax_given=ax[1,0])
    plot_heatmap(kl_y, title="< KL[q(y|z,x) || p(y|z)] >_q(z|x)", beta_ys=results['beta_ys'], beta_zs=results['beta_zs'], ax_given=ax[1,1])

    plt.show()


def beta_pair_to_rgb(log_beta_y, log_beta_z, maximum_beta=10.):
    angle = np.arctan2(log_beta_y, log_beta_z)
    distance_from_origin = np.sqrt(log_beta_y**2 + log_beta_z**2)
    maximum_distance = np.sqrt(2 * (np.log(maximum_beta) ** 2))
    return colorsys.hsv_to_rgb(
        angle / (2 * np.pi) + 0.5,
        distance_from_origin / maximum_distance,
        0.8)

def betas_to_rgb(beta_ys, beta_zs, maximum_beta=10.):
    num_beta_y = len(beta_ys)
    num_beta_z = len(beta_zs)

    betas_rgb_M = np.full((num_beta_y, num_beta_z, 3), 0.)
    beta_ys_M = np.full((num_beta_y, num_beta_z), 0.)
    beta_zs_M = np.full((num_beta_y, num_beta_z), 0.)

    for beta_y in beta_ys:
        for beta_z in beta_zs:
            beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
            beta_z_idx = beta_zs.index(beta_z)
            rgb = beta_pair_to_rgb(np.log(beta_y), np.log(beta_z), maximum_beta=maximum_beta)
            betas_rgb_M[beta_y_idx][beta_z_idx][0] = rgb[0]
            betas_rgb_M[beta_y_idx][beta_z_idx][1] = rgb[1]
            betas_rgb_M[beta_y_idx][beta_z_idx][2] = rgb[2]
            beta_ys_M[beta_y_idx][beta_z_idx] = beta_y
            beta_zs_M[beta_y_idx][beta_z_idx] = beta_z

    return betas_rgb_M, beta_ys_M, beta_zs_M


def mi_bound_given_acc(acc, labels_dist):
    N_C = len(labels_dist)
    L = np.array(labels_dist) / N_C
    acc_M = np.stack((acc, 1-acc), axis=1)
    H_L = entropy(L, base=2)

    H_A = entropy(acc_M, base=2, axis=-1)

    return (H_L - H_A - (1-acc) * np.log2(N_C-1)) * np.log(2)


def mi_acc_diff(acc, target, labels_dist):
    yt = mi_bound_given_acc(acc, labels_dist)
    return (yt - target)**2

def get_acc_bound_given_mi(mi_values, labels_dist=[1,1,1,1,1,1,1,1,1,1]):
    shape = mi_values.shape
    acc_solved = np.zeros(mi_values.flatten().shape)
    for idx, mi in enumerate(mi_values.flatten()):
        res = minimize(mi_acc_diff, 1.0, args=(mi, labels_dist), method='Nelder-Mead', tol=1e-6)
        acc_solved[idx] = res.x[0]
    return acc_solved.reshape(shape)


def split_into_same_betas_and_one_betas_and_rest(M_in, beta_ys, beta_zs):
    # Only make sense when betas are in the same range
    assert beta_ys == beta_zs

    num_beta_y = len(beta_ys)

    arr_same_betas = []
    arr_one_betas = []
    M = M_in.copy()
    for beta_y in beta_ys:
        beta_z = beta_y
        beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
        beta_z_idx = beta_zs.index(beta_z)
        if beta_y == 1.:
            arr_one_betas.append(M[beta_y_idx][beta_z_idx].item())
        else:
            arr_same_betas.append(M[beta_y_idx][beta_z_idx].item())

    return np.array(arr_same_betas), np.array(arr_one_betas), M_in.flatten()


def split_rates_and_value(R_z, R_yz, values, beta_ys, beta_zs):
    # Only make sense when betas are in the same range
    assert beta_ys == beta_zs

    R_z_sb, R_z_ob, R_z_all = split_into_same_betas_and_one_betas_and_rest(R_z, beta_ys, beta_zs)
    R_yz_sb, R_yz_ob, R_yz_all = split_into_same_betas_and_one_betas_and_rest(R_yz, beta_ys, beta_zs)
    values_sb, values_ob, values_all = split_into_same_betas_and_one_betas_and_rest(values, beta_ys, beta_zs)

    return R_z_sb, R_z_ob, R_z_all, R_yz_sb, R_yz_ob, R_yz_all, values_sb, values_ob, values_all

def get_entries_with_given_betas(M, target_beta_pairs, beta_ys, beta_zs):
    num_beta_y = len(beta_ys)

    arr_items = []
    for beta_y, beta_z in target_beta_pairs:
        beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
        beta_z_idx = beta_zs.index(beta_z)
        arr_items.append(M[beta_y_idx][beta_z_idx].item())
        # M[beta_y_idx][beta_z_idx] = np.NaN

    # M_flat = M.flatten()
    # return np.delete(M_flat, np.where(M_flat == None))
    return np.array(arr_items)

def get_points_given_betas(R_z, R_yz, ll, Iq, Ip, target_beta_pairs, beta_ys, beta_zs):
    R_z_got = get_entries_with_given_betas(R_z, target_beta_pairs, beta_ys, beta_zs)
    R_yz_got = get_entries_with_given_betas(R_yz, target_beta_pairs, beta_ys, beta_zs)
    ll_got = get_entries_with_given_betas(ll, target_beta_pairs, beta_ys, beta_zs)
    Iq_got = get_entries_with_given_betas(Iq, target_beta_pairs, beta_ys, beta_zs)
    Ip_got = get_entries_with_given_betas(Ip, target_beta_pairs, beta_ys, beta_zs)

    print("R_z:")
    print(R_z_got)
    print("R_yz:")
    print(R_yz_got)
    print("ll:")
    print(ll_got)
    print("Iq:")
    print(Iq_got)
    print("Ip:")
    print(Ip_got)


def get_support_from_convexHall(R_all, values_all, CH_Top):
    num_all_points = len(R_all)
    # Ignore NaN
    for i in range(len(R_all)):
        if np.isnan(R_all[i]):
            R_all[i] = R_all[i-1]
        if np.isnan(values_all[i]):
            values_all[i] = values_all[i-1]
    # Add view point
    R_all_view = np.append(R_all, 0.0)
    if CH_Top:
        values_all_view = np.append(values_all, values_all.max()*2)
    else:
        values_all_view = np.append(values_all, values_all.max()*-2)
    
    points = np.column_stack((R_all_view, values_all_view))
    qhull_options = 'QG' + str(num_all_points)
    hull = ConvexHull(points, qhull_options=qhull_options)

    support_idx = set()
    for visible_facet in hull.simplices[hull.good]:
        support_idx.add(visible_facet[0])
        support_idx.add(visible_facet[1])

    return list(support_idx)


def plot_scatter_3d(R_z, R_yz, values, title, z_label, xy_same_scale=True, fig_in=None, ax_in=None,
                    beta_ys=None, beta_zs=None,
                    highlight_points_betas=None,
                    dot_size=70,
                    cmap='viridis',
                    CH_plot=False,
                    CH_Top=True):
    # Creating figure
    if fig_in is None:
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
    else:
        fig = fig_in
        ax = ax_in

    # Find highlighted points
    if highlight_points_betas is not None and beta_ys is not None:
        R_z_got = get_entries_with_given_betas(R_z, highlight_points_betas, beta_ys, beta_zs)
        R_yz_got = get_entries_with_given_betas(R_yz, highlight_points_betas, beta_ys, beta_zs)
        v_got = get_entries_with_given_betas(values, highlight_points_betas, beta_ys, beta_zs)
    
    # Creating plot
    if beta_ys is None and beta_zs is None:
        sctt = ax.scatter3D(R_z, R_yz, values, c=values, cmap=plt.get_cmap(cmap))
    else:
        R_z_sb, R_z_ob, R_z_all, R_yz_sb, R_yz_ob, R_yz_all, values_sb, values_ob, values_all = split_rates_and_value(R_z, R_yz, values, beta_ys, beta_zs)
        sctt = ax.scatter3D(R_z_all, R_yz_all, values_all, s=dot_size, c=values_all, alpha=0.8, cmap=plt.get_cmap(cmap))
        # sctt = ax.plot_surface(R_z, R_yz, values, alpha=0.8, cmap=plt.get_cmap("viridis")) #cm.coolwarm
        # ax.scatter3D(R_z_sb, R_yz_sb, values_sb, edgecolors='r')
        # ax.scatter3D(R_z_ob, R_yz_ob, values_ob, edgecolors='k')
        ax.scatter3D(R_z_sb, R_yz_sb, values_sb, s=150, facecolors='none', edgecolors='r', linewidth=2)
        ax.scatter3D(R_z_ob, R_yz_ob, values_ob, s=150, facecolors='none', edgecolors='k', linewidth=2)
        if highlight_points_betas is not None:
            ax.scatter3D(R_z_got, R_yz_got, v_got, s=350, facecolors='none', edgecolors='m', linewidth=2.5)
    # plt.title(title)

    # Convex Hull
    if CH_plot:
        R_all = R_z_all+R_yz_all

        support_idx = get_support_from_convexHall(R_all, values_all, CH_Top)
        ax.scatter3D(R_z_all[support_idx], R_yz_all[support_idx], values_all[support_idx], s=100, facecolors='none', edgecolors='purple', linewidth=2)


    ax.set_xlabel(r'$\mathrm{R}(\mathbf{z}_2)$')
    ax.set_ylabel(r'$\mathrm{R}(\mathbf{z}_1|\mathbf{z}_2)$')
    ax.set_zlabel(z_label, fontweight ='bold')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 15)
    if xy_same_scale:
        xy_min = min(R_z.min(), R_yz.min())
        xy_max = max(R_z.max(), R_yz.max())
        v_min = values.min()
        v_max = values.max()
        ax.auto_scale_xyz([xy_min, xy_max], [xy_min, xy_max], [v_min, v_max])

    if fig_in is None:
        # show plot
        plt.show()


def plot_surface_3d(R_z, R_yz, values, title, z_label, xy_same_scale=True, fig_in=None, ax_in=None,
                    beta_ys=None, beta_zs=None,
                    dot_size=70,
                    cmap='viridis'):
    # Creating figure
    if fig_in is None:
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
    else:
        fig = fig_in
        ax = ax_in
    
    # Creating plot
    # if beta_ys is None and beta_zs is None:
    sctt = ax.plot_surface(R_z, R_yz, values, cmap=cm.coolwarm)

    ax.set_xlabel(r'$\mathrm{R}(\mathbf{z}_2)$')
    ax.set_ylabel(r'$\mathrm{R}(\mathbf{z}_1|\mathbf{z}_2)$')
    ax.set_zlabel(z_label, fontweight ='bold')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 15)

    if fig_in is None:
        # show plot
        plt.show()


def plot_two_scatter_3d(R_z, R_yz, ll, Iq, xy_same_scale=True,
                        beta_ys=None, beta_zs=None):
    # Creating figure
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_scatter_3d(R_z, R_yz, ll, "Likelihood vs Rates", "LL", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax,
                    beta_ys=beta_ys, beta_zs=beta_zs)

    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_scatter_3d(R_z, R_yz, Iq, "MI_q vs Rates", "I_q(z;l)", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax,
                    beta_ys=beta_ys, beta_zs=beta_zs)
    
    # show plot
    plt.show()


def plot_n_scatter_3d(metrics_to_plot, xy_same_scale=True,
                        beta_ys=None, beta_zs=None):
    # Creating figure
    fig = plt.figure(figsize=plt.figaspect(0.3))

    R_z = metrics_to_plot['R_z']
    R_yz = metrics_to_plot['R_yz']
    metrics = metrics_to_plot['metrics']
    num_metrics = len(metrics)

    num_col = 4
    num_row = math.ceil(num_metrics / num_col)

    i = 1
    for k, metric in metrics.items():
        ax = fig.add_subplot(num_row, num_col, i, projection='3d')
        plot_scatter_3d(R_z, R_yz, metric['values'], metric['title'], metric['z_label'], xy_same_scale=xy_same_scale,
                        fig_in=fig, ax_in=ax,
                        beta_ys=beta_ys, beta_zs=beta_zs,
                        dot_size=20,
                        cmap=metric['cmap'],
                        CH_plot=metric['CH_plot'],
                        CH_Top=metric['CH_Top'])
        i += 1

    # show plot
    plt.show()


def plot_scatter_3metrics_3d(metrics_to_plot, beta_ys, beta_zs):

    psnr = metrics_to_plot['metrics']['psnr_M']['values']
    is_mean = metrics_to_plot['metrics']['is_mean_M']['values']
    accs = metrics_to_plot['metrics']['svm_rbf_acc_M']['values']

    opt_idx = []
    opt_idx.append(np.nanargmax(psnr, keepdims=True))
    opt_idx.append(np.nanargmax(is_mean, keepdims=True))
    opt_idx.append(np.nanargmax(accs, keepdims=True))
    opt_idx = np.array(opt_idx)

    psnr_label = metrics_to_plot['metrics']['psnr_M']['z_label']
    is_mean_label = metrics_to_plot['metrics']['is_mean_M']['z_label']
    accs_label = metrics_to_plot['metrics']['svm_rbf_acc_M']['z_label']

    maximum_beta = 12.

    grid_lw = 0.3
    grid_lc = 'grey'

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # Set plot limits:
    # toplot_cond = np.full_like(psnr, True, dtype=bool)
    # toplot_cond = np.logical_and(psnr >= 20, is_mean >= 3.6)

    color_rgb_values, beta_ys_M, beta_zs_M = betas_to_rgb(beta_ys, beta_zs, maximum_beta)
    betas_eq_cond = (beta_ys_M == beta_zs_M)

    # proj_colors = ['hotpink','darkviolet','mediumblue']
    proj_colors = ['grey','grey','grey']

    accs_minlimit = 0.
    psnr_maxlimit = np.nanmax(psnr) + 1
    is_maxlimit = np.nanmax(is_mean)+1

    ax.scatter(psnr, is_mean, np.ones_like(accs)*accs_minlimit, c=proj_colors[0], marker='.', lw=0, alpha=0.2)
    ax.scatter(np.ones_like(psnr) * psnr_maxlimit, is_mean, accs, c=proj_colors[1], marker='.', lw=0, alpha=0.2)
    ax.scatter(psnr, np.ones_like(is_mean)*(is_maxlimit), accs, c=proj_colors[2], marker='.', lw=0, alpha=0.2)

    opt_markers = ['^', 'p', 'D']
    for i, opt in enumerate(opt_idx):
        psnr_v = psnr.flatten()[opt][0][0]
        is_v = is_mean.flatten()[opt][0][0]
        accs_v = accs.flatten()[opt][0][0]
        ax.scatter(psnr_v, is_v, accs_minlimit, s=30, edgecolors=proj_colors[0], facecolors='none', marker=opt_markers[i], lw=1.5, alpha=0.9)
        ax.scatter(psnr_maxlimit, is_v, accs_v, s=30, edgecolors=proj_colors[1], facecolors='none', marker=opt_markers[i], lw=1.5, alpha=0.9)
        ax.scatter(psnr_v, is_maxlimit, accs_v, s=30, edgecolors=proj_colors[2], facecolors='none', marker=opt_markers[i], lw=1.5, alpha=0.9)

        ax.plot([psnr_v,psnr_v], [is_v,is_v], [accs_v,accs_minlimit], 'k--', lw=1, alpha=0.6)
        ax.plot([psnr_v,psnr_maxlimit], [is_v,is_v], [accs_v,accs_v], 'k--', lw=1, alpha=0.6)
        ax.plot([psnr_v,psnr_v], [is_v,is_maxlimit], [accs_v,accs_v], 'k--', lw=1, alpha=0.6)

    surf_col = color_rgb_values
    ax.plot_surface(psnr, is_mean, accs, alpha=0.7, facecolors=surf_col, edgecolor='none')


    ax.set_xlabel(psnr_label, fontweight ='bold')
    ax.set_ylabel(is_mean_label, fontweight ='bold')
    ax.set_zlabel(accs_label, fontweight ='bold')
    ax.set_zlim(0., 0.9)

    # Highlight opt
    for i, opt in enumerate(opt_idx):
        ax.scatter3D(psnr.flatten()[opt], is_mean.flatten()[opt], accs.flatten()[opt], s=50, facecolors='none', edgecolors='k', marker=opt_markers[i], linewidth=1.5)
    


    cbar_ax = fig.add_subplot(1, 2, 2, adjustable='box', aspect=0.1)
    cbar_img = [
        [beta_pair_to_rgb(log_beta_y, log_beta_z, maximum_beta) for log_beta_z in np.linspace(-np.log(maximum_beta), np.log(maximum_beta), 101)]
        for log_beta_y in np.linspace(-np.log(maximum_beta), np.log(maximum_beta), 101)
    ]
    cbar_ax.imshow(
        cbar_img,
        extent = (
            -np.log(maximum_beta), np.log(maximum_beta),
            -np.log(maximum_beta), np.log(maximum_beta)
        ),
        aspect = 'equal',
        origin = 'lower',
    )
    cbar_ax.set_xticks(np.log(beta_zs), minor=True)
    cbar_ax.set_yticks(np.log(beta_ys), minor=True)
    cbar_ax.grid(color=grid_lc, linestyle='-', linewidth=grid_lw, which='minor')
    cbar_ax.hlines(y=0, xmin=np.log(beta_zs).min(), xmax=np.log(beta_zs).max(), linewidth=grid_lw, color=grid_lc)
    cbar_ax.vlines(x=0, ymin=np.log(beta_ys).min(), ymax=np.log(beta_ys).max(), linewidth=grid_lw, color=grid_lc)
    cbar_ax.scatter(np.log(beta_zs_M), np.log(beta_ys_M), s=30, facecolors='none', edgecolors='k', linewidth=0.5)
    cbar_ax.scatter(0.0, 0.0, s=30, c='k')#, facecolors='none', edgecolors='k', linewidth=1.5)
    cbar_ax.plot(np.log(beta_zs_M)[betas_eq_cond], np.log(beta_ys_M)[betas_eq_cond], 'r--', lw=1, alpha=0.9)
    # Highlight opt
    for i, opt in enumerate(opt_idx):
        cbar_ax.scatter(np.log(beta_zs_M).flatten()[opt], np.log(beta_ys_M).flatten()[opt], s=60, facecolors='none', edgecolors='k', marker=opt_markers[i], linewidth=1.5)

    cbar_ax.set_xlabel(r'$\mathrm{log}(\beta_z)$')
    cbar_ax.set_ylabel(r'$\mathrm{log}(\beta_y)$')



    plt.show()


def plot_n_surface(metrics_to_plot):

    R_z = metrics_to_plot['R_z']
    R_yz = metrics_to_plot['R_yz']
    metrics = metrics_to_plot['metrics']
    num_metrics = len(metrics)

    num_col = 4
    num_row = math.ceil(num_metrics / num_col)

    fig = make_subplots(
        rows=num_row, cols=num_col,
        specs=[[{'type': 'surface'}]*num_col for i in range(num_row)])

    col_i = 1
    row_i = 1
    for k, metric in metrics.items():
        fig.add_trace(
            go.Surface(x=R_z, y=R_yz, z=metric['values'], colorscale='Viridis', showscale=False),
            row=row_i, col=col_i)

        fig.update_scenes(  xaxis_title_text=r'$\mathrm{R}(\mathbf{z}_2)$',  
                            yaxis_title_text=r'$\mathrm{R}(\mathbf{z}_1|\mathbf{z}_2)$',  
                            zaxis_title_text=metric['z_label'])
        
        col_i = col_i + 1 if col_i + 1 <= num_col else 1
        row_i = row_i + 1 if col_i == 1 else row_i

    fig.show()
    # fig.write_image("test_fig1.pdf")


def plot_2d_total_rate(R_z, R_yz, values, title, y_label,
                    beta_ys, beta_zs,
                    xy_same_scale=True, fig_in=None, ax_in=None,
                    cmap='viridis',
                    save_name=None,
                    upper_bound=None,
                    CH_plot=False,
                    CH_Top=True):
    # Creating figure
    if fig_in is None:
        fig = plt.figure(figsize = (7, 10))
        ax = plt.axes()
    else:
        fig = fig_in
        ax = ax_in
    
    # Creating plot
    R_z_sb, R_z_ob, R_z_all, R_yz_sb, R_yz_ob, R_yz_all, values_sb, values_ob, values_all = split_rates_and_value(R_z, R_yz, values, beta_ys, beta_zs)
    
    R = np.concatenate((R_z_sb, R_z_ob)) + np.concatenate((R_yz_sb, R_yz_ob))
    values = np.concatenate((values_sb, values_ob))
    
    sctt = ax.scatter(R, values, s=10, c=values, alpha=1., cmap=plt.get_cmap(cmap))
    ax.scatter(R_z_all+R_yz_all, values_all, s=10, c=values_all, alpha=1., cmap=plt.get_cmap(cmap))
    ax.scatter(R_z_sb+R_yz_sb, values_sb, s=50, facecolors='none', edgecolors='r', linewidth=1.5)
    ax.scatter(R_z_ob+R_yz_ob, values_ob, s=50, facecolors='none', edgecolors='k', linewidth=1.5)

    if upper_bound is not None:
        ub_sb, ub_ob, ub_all = split_into_same_betas_and_one_betas_and_rest(upper_bound, beta_ys, beta_zs)
        ax.plot(np.sort(R), np.sort(np.concatenate((ub_sb, ub_ob))), 'k--')
    # plt.title(title)
    plt.grid(visible=True, alpha=0.5, linestyle='--')

    ax.set_xlabel(r'$\mathrm{R}$')
    ax.set_ylabel(y_label)
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=15)
    if xy_same_scale:
        ax.set_aspect('equal','box')

    # Convex Hull
    if CH_plot:
        R_all = R_z_all+R_yz_all
        support_idx = get_support_from_convexHall(R_all, values_all, CH_Top)
        ax.scatter(R_all[support_idx], values_all[support_idx], s=30, facecolors='none', edgecolors='purple', linewidth=1.5)


    if fig_in is None:
        # show plot
        if save_name is None:
            plt.show()
        else:
            fig.savefig(save_name, bbox_inches='tight')


def plot_betas_equal_2d(R_z, R_yz, values, title, y_label,
                    beta_ys, beta_zs,
                    xy_same_scale=True, fig_in=None, ax_in=None,
                    cmap='viridis',
                    save_name=None,
                    upper_bound=None):
    # Creating figure
    if fig_in is None:
        fig = plt.figure(figsize = (7, 10))
        ax = plt.axes()
    else:
        fig = fig_in
        ax = ax_in
    
    # Creating plot
    R_z_sb, R_z_ob, R_z_all, R_yz_sb, R_yz_ob, R_yz_all, values_sb, values_ob, values_all = split_rates_and_value(R_z, R_yz, values, beta_ys, beta_zs)
    
    R = np.concatenate((R_z_sb, R_z_ob)) + np.concatenate((R_yz_sb, R_yz_ob))
    values = np.concatenate((values_sb, values_ob))
    
    sctt = ax.scatter(R, values, s=70, c=values, alpha=1., cmap=plt.get_cmap(cmap))
    ax.scatter(R_z_sb+R_yz_sb, values_sb, s=150, facecolors='none', edgecolors='r', linewidth=2)
    ax.scatter(R_z_ob+R_yz_ob, values_ob, s=150, facecolors='none', edgecolors='k', linewidth=2)
    if upper_bound is not None:
        ub_sb, ub_ob, ub_all = split_into_same_betas_and_one_betas_and_rest(upper_bound, beta_ys, beta_zs)
        ax.plot(np.sort(R), np.sort(np.concatenate((ub_sb, ub_ob))), 'k--')
    # plt.title(title)
    plt.grid(visible=True, alpha=0.5, linestyle='--')

    ax.set_xlabel(r'$\mathrm{R}$')
    ax.set_ylabel(y_label)
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=15)
    if xy_same_scale:
        ax.set_aspect('equal','box')

    if fig_in is None:
        # show plot
        if save_name is None:
            plt.show()
        else:
            fig.savefig(save_name, bbox_inches='tight')


def plot_scatter_2d(R_z, R_yz, values, title, xy_same_scale=True, fig_in=None, ax_in=None, select_diag=None,
                    beta_ys=None, beta_zs=None,
                    cmap='viridis',
                    highlight_points_betas=None,
                    save_name=None,
                    CH_plot=False,
                    CH_Top=True):
    # Creating figure
    if fig_in is None:
        fig = plt.figure(figsize = (7, 10))
        ax = plt.axes()
    else:
        fig = fig_in
        ax = ax_in

    # Find highlighted points
    if highlight_points_betas is not None and beta_ys is not None:
        R_z_got = get_entries_with_given_betas(R_z, highlight_points_betas, beta_ys, beta_zs)
        R_yz_got = get_entries_with_given_betas(R_yz, highlight_points_betas, beta_ys, beta_zs)
    
    # Creating plot
    if beta_ys is None and beta_zs is None:
        sctt = ax.scatter(R_z, R_yz, c=values, alpha=0.8, cmap=plt.get_cmap(cmap))
    else:
        R_z_sb, R_z_ob, R_z_all, R_yz_sb, R_yz_ob, R_yz_all, values_sb, values_ob, values_all = split_rates_and_value(R_z, R_yz, values, beta_ys, beta_zs)
        sctt = ax.scatter(R_z_all, R_yz_all, s=10, c=values_all, alpha=1., cmap=plt.get_cmap(cmap))
        ax.scatter(R_z_sb, R_yz_sb, s=80, facecolors='none', edgecolors='r', linewidth=1.5)
        ax.scatter(R_z_ob, R_yz_ob, s=80, facecolors='none', edgecolors='k', linewidth=1.5)
        if highlight_points_betas is not None:
            ax.scatter(R_z_got, R_yz_got, s=350, facecolors='none', edgecolors='m', linewidth=2.5)
    plt.title(title)
    plt.grid(visible=True, alpha=0.5, linestyle='--')

    # Convex Hull
    if CH_plot:
        R_all = R_z_all+R_yz_all
        support_idx = get_support_from_convexHall(R_all, values_all, CH_Top)
        ax.scatter(R_z_all[support_idx], R_yz_all[support_idx], s=30, facecolors='none', edgecolors='purple', linewidth=1.5)
        

    y_pos = np.linspace(0, 1.5, 15)
    # if select_diag is not None:
    #     plt.axline((0, y_pos[select_diag]), slope=-1, alpha=0.8, linestyle='--', color='r', transform=plt.gca().transAxes)
    # else:
    for idx, pos in enumerate(y_pos):
        if select_diag is not None and idx == select_diag:
            plt.axline((0, pos), slope=-1, alpha=0.8, linestyle='--', linewidth=0.5, color='r', transform=plt.gca().transAxes)
        else:
            plt.axline((0, pos), slope=-1, alpha=0.8, linestyle='--', linewidth=0.5, color='k', transform=plt.gca().transAxes)

    ax.set_xlabel(r'$\mathrm{R}(\mathbf{z}_2)$')
    ax.set_ylabel(r'$\mathrm{R}(\mathbf{z}_1|\mathbf{z}_2)$')
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=15)
    if xy_same_scale:
        ax.set_aspect('equal','box')

    if fig_in is None:
        # show plot
        if save_name is None:
            plt.show()
        else:
            fig.savefig(save_name, bbox_inches='tight')


def plot_two_scatter_2d(R_z, R_yz, ll, Iq, xy_same_scale=True, select_diag=None,
                        beta_ys=None, beta_zs=None):
    # Creating figure
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1)
    plot_scatter_2d(R_z, R_yz, ll, "Likelihood vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)

    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2)
    plot_scatter_2d(R_z, R_yz, Iq, "I_q(z;l) vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)
    
    # show plot
    plt.show()
    # fig.savefig("example.pdf", bbox_inches='tight')


def plot_three_scatter_2d(R_z, R_yz, ll, Iq, Ip, xy_same_scale=True, select_diag=None,
                        beta_ys=None, beta_zs=None):
    # Creating figure
    fig = plt.figure(figsize=plt.figaspect(0.3))

    # set up the axes for the first plot
    ax = fig.add_subplot(1, 3, 1)
    plot_scatter_2d(R_z, R_yz, ll, "Likelihood vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)

    # set up the axes for the second plot
    ax = fig.add_subplot(1, 3, 2)
    plot_scatter_2d(R_z, R_yz, Iq, "I_q(z;l) vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)
    
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 3, 3)
    plot_scatter_2d(R_z, R_yz, Ip, "I_p(z;l) vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)
    
    # show plot
    plt.show()
    # fig.savefig("svhn_2dx3.pdf", bbox_inches='tight')


def plot_four_scatter_2d(R_z, R_yz, ll, Iq, Ip, fid, xy_same_scale=True, select_diag=None,
                        beta_ys=None, beta_zs=None):
    # Creating figure
    fig = plt.figure(figsize=plt.figaspect(0.3))

    # set up the axes for the first plot
    ax = fig.add_subplot(2, 2, 1)
    plot_scatter_2d(R_z, R_yz, ll, "Likelihood vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)

    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 2)
    plot_scatter_2d(R_z, R_yz, Iq, "I_q(z;l) vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)
    
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 3)
    plot_scatter_2d(R_z, R_yz, Ip, "I_p(z;l) vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)
    
    # set up the axes for the fid plot
    ax = fig.add_subplot(2, 2, 4)
    plot_scatter_2d(R_z, R_yz, fid, "FID vs Rates", xy_same_scale=xy_same_scale,
                    fig_in=fig, ax_in=ax, select_diag=select_diag,
                    beta_ys=beta_ys, beta_zs=beta_zs)

    # show plot
    plt.show()
    # fig.savefig("svhn_2dx3.pdf", bbox_inches='tight')


def plot_n_scatter_2d(metrics_to_plot, 
                        plot_betas_eq=True,
                        plot_convexhull=False,
                        xy_same_scale=True,
                        beta_ys=None, beta_zs=None):
    # Creating figure
    fig = plt.figure(figsize=plt.figaspect(0.3))

    R_z = metrics_to_plot['R_z']
    R_yz = metrics_to_plot['R_yz']
    metrics = metrics_to_plot['metrics']
    num_metrics = len(metrics)

    num_col = 4
    num_row = math.ceil(num_metrics / num_col)

    i = 1
    for k, metric in metrics.items():
        ax = fig.add_subplot(num_row, num_col, i)
        if plot_betas_eq and plot_convexhull:
            plot_2d_total_rate(R_z, R_yz, metric['values'], metric['title'], metric['z_label'], 
                            beta_ys, beta_zs,
                            xy_same_scale=xy_same_scale,
                            fig_in=fig, ax_in=ax,
                            cmap=metric['cmap'],
                            upper_bound=metric['acc_ub'],
                            CH_plot=metric['CH_plot'],
                            CH_Top=metric['CH_Top'])
        elif plot_betas_eq and not plot_convexhull:
            plot_betas_equal_2d(R_z, R_yz, metric['values'], metric['title'], metric['z_label'], 
                            beta_ys, beta_zs,
                            xy_same_scale=xy_same_scale,
                            fig_in=fig, ax_in=ax,
                            cmap=metric['cmap'],
                            upper_bound=metric['acc_ub'])
        else:
            plot_scatter_2d(R_z, R_yz, metric['values'], metric['title'], xy_same_scale=xy_same_scale,
                            fig_in=fig, ax_in=ax,
                            beta_ys=beta_ys, beta_zs=beta_zs,
                            cmap=metric['cmap'],
                            CH_plot=metric['CH_plot'],
                            CH_Top=metric['CH_Top'])
        i += 1

    # show plot
    plt.show()
    # fig.savefig("svhn_2dx3.pdf", bbox_inches='tight')


def flag_entries_with_given_betas(M, target_beta_pairs, beta_ys, beta_zs):
    num_beta_y = len(beta_ys)

    for beta_y, beta_z in target_beta_pairs:
        beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
        beta_z_idx = beta_zs.index(beta_z)
        M[beta_y_idx][beta_z_idx] = np.NaN

    # M_flat = M.flatten()
    # return np.delete(M_flat, np.where(M_flat == None))
    return M


def flag_results_with_given_betas(results, metric_list, target_beta_pairs, beta_ys, beta_zs):
    num_beta_y = len(beta_ys)

    for beta_y, beta_z in target_beta_pairs:
        beta_y_idx = num_beta_y - beta_ys.index(beta_y) - 1
        beta_z_idx = beta_zs.index(beta_z)
        results['kl_z_M'][beta_y_idx][beta_z_idx] = np.NaN
        results['kl_y_M'][beta_y_idx][beta_z_idx] = np.NaN

        for metric_name in metric_list:
            if metric_name in results:
                results[metric_name][beta_y_idx][beta_z_idx] = np.NaN

    # M_flat = M.flatten()
    # return np.delete(M_flat, np.where(M_flat == None))
    return results


def organise_results_for_plotting(results, metric_list, plot_specs, testdata_dist):
    metrics_to_plot = {}
    
    if 'kl_z_M' in results:
        metrics_to_plot['R_z'] = results['kl_z_M']
    else:
        metrics_to_plot['R_z'] = np.zeros_like(results['kl_y_M'])
    metrics_to_plot['R_yz'] = results['kl_y_M']
    metrics_to_plot['metrics'] = {}

    acc_ub = None
    if 'mi_q_mc_M' in metric_list and 'mi_q_mc_M' in results:
        acc_ub = get_acc_bound_given_mi(results['mi_q_mc_M'], testdata_dist)

    for metric_name in metric_list:
        if metric_name in results:
            if plot_specs[metric_name]['plot_acc_ub']:
                upper_bound = acc_ub
            else:
                upper_bound = None

            metrics_to_plot['metrics'][metric_name] = {
                'values':   results[metric_name],
                'title':    plot_specs[metric_name]['title'],
                'z_label':    plot_specs[metric_name]['z_label'],
                'cmap':     plot_specs[metric_name]['cmap'],
                'acc_ub':   upper_bound,
                'CH_plot':  plot_specs[metric_name]['CH_plot'],
                'CH_Top':   plot_specs[metric_name]['CH_Top']
            }

    if 'mi_zy_diff' in metric_list and 'mi_q_mc_M' in results and 'mi_q_mc_y_M' in results:
        diff = results['mi_q_mc_y_M'] - results['mi_q_mc_M']
        print('assert (diff < 0).sum() == 0')
        print((diff < 0).sum())
        print(diff[diff < 0])
        
        metrics_to_plot['metrics']['mi_zy_diff'] = {
            'values':   diff,
            'title':    plot_specs['mi_zy_diff']['title'],
            'z_label':    plot_specs['mi_zy_diff']['z_label'],
            'cmap':     plot_specs['mi_zy_diff']['cmap'],
            'acc_ub':   None
        }

    if 'recons_acc_M' in metric_list and 'recons_acc_M' in results and 'clsfr_acc' in results:
        metrics_to_plot['metrics']['recons_acc_M']['clsfr_acc'] = results['clsfr_acc']
        print(f"Classifier test set accuracy: {results['clsfr_acc']}")


    # if 'is_std_M' in results:
    #     print(results['is_std_M'])

    return metrics_to_plot



testdata_dist = [1744., 5099., 4149., 2882., 2523., 2384., 1977., 2019., 1660., 1595.] # SVHN
plot_specs = {
    'll_M': {
        'title':    'Likelihood vs Rates',
        'z_label':  'LL',
        'cmap':     'viridis',
        'CH_plot':  False,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'mi_q_mksg_M': {
        'title':    'MI_q vs Rates',
        'z_label':  'I_q(z;l)',
        'cmap':     'viridis',
        'CH_plot':  False,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'mi_p_mksg_M': {
        'title':    'MI_p vs Rates',
        'z_label':  'I_p(z;l)',
        'cmap':     'viridis',
        'CH_plot':  False,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'fid_M': {
        'title':    'FID vs Rates',
        'z_label':  'FID',
        'cmap':     'viridis_r',
        'CH_plot':  True,
        'CH_Top':   False,
        'plot_acc_ub': False
    },
    'mi_q_mc_M': {
        'title':    'MI_q_mc vs Rates',
        'z_label':  'I_q(z;l)_mc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'lr_acc_mean_M': {
        'title':    'LR_acc vs Rates',
        'z_label':  'LR_Acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': True
    },
    'psnr_M': {
        'title':    'PSNR vs Rates',
        'z_label':  'PSNR',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'svm_rbf_acc_M': {
        'title':    'svm_rbf_acc vs Rates',
        'z_label':  'svm_rbf_acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': True
    },
    'svm_linear_acc_M': {
        'title':    'svm_linear_acc vs Rates',
        'z_label':  'svm_linear_acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': True
    },
    'kNN_acc_M': {
        'title':    'kNN_acc vs Rates',
        'z_label':  'kNN_acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': True
    },
    'mi_q_mc_y_M': {
        'title':    'MI_q_y_mc vs Rates',
        'z_label':  'I_q(y;l)_mc',
        'cmap':     'viridis',
        'CH_plot':  False,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'lr_acc_mean_y_M': {
        'title':    'LR_y_acc vs Rates',
        'z_label':  'LR_y_Acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'svm_rbf_acc_y_M': {
        'title':    'svm_rbf_y_acc vs Rates',
        'z_label':  'svm_rbf_y_acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'svm_linear_acc_y_M': {
        'title':    'svm_linear_y_acc vs Rates',
        'z_label':  'svm_linear_y_acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'kNN_acc_y_M': {
        'title':    'kNN_y_acc vs Rates',
        'z_label':  'kNN_y_acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'is_mean_M': {
        'title':    'IS vs Rates',
        'z_label':  'IS',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'is_div_M': {
        'title':    'IS-diversity vs Rates',
        'z_label':  'IS-diversity',
        'cmap':     'viridis',
        'CH_plot':  False,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'is_sharp_M': {
        'title':    'IS-sharpness vs Rates',
        'z_label':  'IS-sharpness',
        'cmap':     'viridis',
        'CH_plot':  False,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'mi_zy_diff': {
        'title':    'mi_zy_diff vs Rates',
        'z_label':  'Iq(z1;l)-Iq(z2;l)',
        'cmap':     'viridis',
        'CH_plot':  False,
        'CH_Top':   False,
        'plot_acc_ub': False
    },
    'recons_acc_M': {
        'title':    'recons_acc vs Rates',
        'z_label':  'recons_acc',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
    'recons_pred_entropy_M': {
        'title':    'recons_pred_entropy vs Rates',
        'z_label':  'recons_pred_entropy',
        'cmap':     'viridis',
        'CH_plot':  True,
        'CH_Top':   True,
        'plot_acc_ub': False
    },
}




# run_batch_name = 'betas_grid_svhn_hvae_c32_fSigZ_grid21_z32'
# run_batch_name = 'betas_grid_svhn_lvae_c32_grid21_z32'
run_batch_name = 'betas_grid_cifar_hvae_c32_grid21_z32'
# run_batch_name = 'betas_grid_bMNIST_DG_hvae_c16_grid21_z20'





metric_list = [ 'fid_M', 'is_mean_M', 'is_div_M', 'is_sharp_M', 'psnr_M', 'recons_acc_M', 'recons_pred_entropy_M',
                'mi_q_mc_M', 'lr_acc_mean_M', 'svm_rbf_acc_M', 'svm_linear_acc_M', 'kNN_acc_M', 
                'lr_acc_mean_y_M', 'svm_rbf_acc_y_M', 'svm_linear_acc_y_M', 'kNN_acc_y_M'
                ]


failed_beta_yz_pairs_svhn_lvae = [(3.98107171, 0.63095734), (0.1, 0.31622777)]


result_dir = './runs/' + run_batch_name
result_file_name = "eval_results.pkl"
results = utils_eval.load_eval_results(result_dir, result_file_name)

# For betas_grid_svhn_lvae_c32_grid21_z32
if run_batch_name == 'betas_grid_svhn_lvae_c32_grid21_z32':
    results = flag_results_with_given_betas(results, metric_list, failed_beta_yz_pairs_svhn_lvae, results['beta_ys'], results['beta_zs'])

metrics_to_plot = organise_results_for_plotting(results, metric_list, plot_specs, testdata_dist)




# ---------------
# 2D
# ---------------



# -------------

# Convex Hull


# plot_n_scatter_2d(metrics_to_plot, plot_betas_eq=False, plot_convexhull=True,
#                     beta_ys=results['beta_ys'], beta_zs=results['beta_zs'],
#                     xy_same_scale=False)

# -------------


# ---------------
# 3D
# ---------------


plot_n_scatter_3d(metrics_to_plot, xy_same_scale=False,
                        beta_ys=results['beta_ys'], beta_zs=results['beta_zs'])



# ---------------

# plot_scatter_3metrics_3d(metrics_to_plot, results['beta_ys'], results['beta_zs'])