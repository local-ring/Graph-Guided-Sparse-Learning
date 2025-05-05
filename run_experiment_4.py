import os
import sys
import time
import pickle
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from itertools import product
import pickle as pkl
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from collections import ChainMap, defaultdict
import scipy.sparse as sp

import pickle as pkl
from scipy.sparse import lil_matrix
import logging

# ensure repo root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))


try:
    import sparse_module

    try:
        from sparse_module import wrap_head_tail_bisearch
    except ImportError:
        print('cannot find wrap_head_tail_bisearch method in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print('\n'.join([
        'cannot find the module: sparse_module',
        'try run: \'python setup.py build_ext --inplace\' first! ']))

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

os.system("export OMP_NUM_THREADS=1")
os.system("export OPENBLAS_NUM_THREADS=1")
os.system("export MKL_NUM_THREADS=1")
os.system("export VECLIB_MAXIMUM_THREADS=1")
os.system("export NUMEXPR_NUM_THREADS=1")

# Detect available CPUs dynamically - AL
NUM_CPUS = multiprocessing.cpu_count()


np.random.seed(17)
root_p = 'results/'
if not os.path.exists(root_p):
    os.mkdir(root_p)


def simu_grid_graph(width, height):
    width, height = int(width), int(height)
    edges, weights = [], []
    index = 0
    for i in range(height):
        for j in range(width):
            if (index % width) != (width - 1):
                edges.append((index, index + 1))
                if index + width < int(width * height):
                    edges.append((index, index + width))
            else:
                if index + width < int(width * height):
                    edges.append((index, index + width))
            index += 1
    edges = np.asarray(edges, dtype=int)
    weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def get_img_data(root_p):
    img_name_list = ['background', 'angio', 'icml']
    re_height, re_width = 100, 100
    resized_data = dict()
    s_list = []
    for img_ind, _ in enumerate(img_name_list):
        img = sio.loadmat(root_p + 'grid_%s.mat' % _)['x_gray']
        im = Image.fromarray(img).resize((re_height, re_width), Image.BILINEAR)
        im = np.asarray(im.getdata()).reshape((re_height, re_width))
        resized_data[_] = im
        s_list.append(len(np.nonzero(resized_data[_])[0]))
    img_data = {
        'img_list': img_name_list,
        'background': np.asarray(resized_data['background']).flatten(),
        'angio': np.asarray(resized_data['angio']).flatten(),
        'icml': np.asarray(resized_data['icml']).flatten(),
        'height': re_height,
        'width': re_width,
        'p': re_height * re_width,
        's': {_: s_list[ind] for ind, _ in enumerate(img_name_list)},
        's_list': s_list,
        'g_dict': {'background': 3, 'angio': 1, 'icml': 4},
        'graph': simu_grid_graph(height=re_height, width=re_width)
    }
    return img_data


def algo_head_tail_bisearch(
        edges, x, costs, g, root, s_low, s_high, max_num_iter, verbose):
    """ This is the wrapper of head/tail-projection proposed in [2].
    :param edges:           edges in the graph.
    :param x:               projection vector x.
    :param costs:           edge costs in the graph.
    :param g:               the number of connected components.
    :param root:            root of subgraph. Usually, set to -1: no root.
    :param s_low:           the lower bound of the sparsity.
    :param s_high:          the upper bound of the sparsity.
    :param max_num_iter:    the maximum number of iterations used in
                            binary search procedure.
    :param verbose: print out some information.
    :return:            1.  the support of the projected vector
                        2.  the projected vector
    """
    prizes = x * x
    # length of prizes is length of x
    # to avoid too large upper bound problem.
    if s_high >= len(prizes) - 1:
        s_high = len(prizes) - 1
    re_nodes = wrap_head_tail_bisearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    proj_w = np.zeros_like(x)
    proj_w[re_nodes[0]] = x[re_nodes[0]]
    return re_nodes[0], proj_w


def algo_graph_iht(
        x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s,
        root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    """
    :param x_mat: design matrix.
    :param y: response vector.
    :param max_epochs: maximal number of iterations for outer layer algorithm
    :param x_star: ground truth vector
    :param edges: edges in the graph
    :param g: connected component
    :param s: sparsity level
    :param gamma: to control the range of the sparsity since it cannot be the exact value
    :return:
    1. x_hat: the estimator of the vector
    """
    start_time = time.time()
    x_hat = np.copy(x0)
    xtx = np.dot(np.transpose(x_mat), x_mat)
    xty = np.dot(np.transpose(x_mat), y)

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    p = len(x0)
    beta = eigh(xtx, eigvals_only=True, subset_by_index=[p - 1, p - 1])[0]
    lr = 1. / beta
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -1. * (xty - np.dot(xtx, x_hat))
        head_nodes, proj_gradient = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high,
            proj_max_num_iter, verbose)
        bt = x_hat - lr * proj_gradient
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, g, root, t_low, t_high,
            proj_max_num_iter, verbose)
        x_hat = proj_bt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_graph_cosamp(
        x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        h_g, t_g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    h_low, h_high = int(2 * s), int(2 * s * (1.0 + gamma))
    t_low, t_high = int(s), int(s * (1.0 + gamma))

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -2. * (np.dot(xtx, x_hat) - xty)  # proxy
        head_nodes, proj_grad = algo_head_tail_bisearch(
            edges, grad, costs, h_g, root,
            h_low, h_high, proj_max_num_iter, verbose)
        gamma = np.union1d(x_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y)
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, t_g, root,
            t_low, t_high, proj_max_num_iter, verbose)
        x_hat = proj_bt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_gen_mp(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)
    p = len(x0)
    beta = eigh(xtx, eigvals_only=True, subset_by_index=[p - 1, p - 1])[0]
    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])
        vt = (-c / norm_vt) * proj_vec
        x_hat = x_hat - (np.dot(vt, grad) / beta) * vt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
            print(tt, loss, list_est_err[-1])
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_cosamp(x_mat, y, max_epochs, x_star, x0, tol_algo, step, s):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    m, p = x_mat.shape

    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -(2. / float(m)) * (np.dot(xtx, x_hat) - xty)  # proxy
        gamma = np.argsort(abs(grad))[-2 * s:]  # identify
        gamma = np.union1d(x_hat.nonzero()[0], gamma)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y)
        gamma = np.argsort(abs(bt))[-s:]
        x_hat = np.zeros_like(x_hat)
        x_hat[gamma] = bt[gamma]
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_dmo_acc_fw(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)
    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        eta_t = 2. / (tt + 2.)
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, -x_hat + grad / eta_t, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])

        vt = (-c / norm_vt) * proj_vec
        x_hat += eta_t * (vt - x_hat)
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_dmo_fw(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        eta_t = 2. / (tt + 2.)
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])
        vt = (-c / norm_vt) * proj_vec
        x_hat += eta_t * (vt - x_hat)
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def run_single_test(para):
    method, img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c, gamma = para
    n, p = x_mat.shape
    x0 = np.zeros(p, dtype=np.float64)
    noise = np.random.normal(0, gamma, size=n) # AL: introduce noise to response here 
    y = np.dot(x_mat, x_star) + noise

    if method == 'graph-iht':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_graph_iht(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'graph-cosamp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_graph_cosamp(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, h_g=g, t_g=g, s=s)
    elif method == 'dmo-fw':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_dmo_fw(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'dmo-acc-fw':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_dmo_acc_fw(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'cosamp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_cosamp(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, s)
    elif method == 'gen-mp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_gen_mp(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    else:
        print('something must wrong.')
        exit()
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = 0.0, x_star, [0.0], [0.0], [0.0]
    print('%-13s trial_%03d n: %03d w_error: %.3e num_epochs: %03d run_time: %.3e' %
          (method, trial_i, n, list_est_err[-1], num_epochs, list_run_time[-1]))
    return method, img_name, trial_i, list_est_err[-1], num_epochs, x_hat, list_run_time, list_loss, list_est_err




def recovery_accuracy(x_hat, k=50):
    # take the top k elements of x_hat
    x_hat_top_k = np.argsort(np.abs(x_hat))[-k:]
    ground_truth_top_k = np.arange(k)

    correct_predictions = np.intersect1d(x_hat_top_k, ground_truth_top_k)
    accuracy = len(correct_predictions) / k
    return accuracy

def sparse_learning_recovery(para):
    """
    What we need:
    read from the para;  
    
    run_single_test( ('gen-mp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    """
    # we need pass by the x_mat, s, g, x_star, edges, costs
    trial_i, x_mat, x_star, edges, costs, s, g, max_epochs, tol_algo, step, gamma = para
    np.random.seed(trial_i)

    # img_name, s, g, l1_norm, l2_norm, x_star = pkl.load(open(f'data/grid_img_angio.npz', 'rb'))
    img_name = 'dummy'

    # p = len(x_star)
    # n, p = x_mat.shape
    # x_star = x_star / np.linalg.norm(x_star, ord=1)
    c = np.linalg.norm(x_star, ord=2)
    # x_mat = np.random.normal(0.0, 1.0, (n, p)) / np.sqrt(n)

    results = {}

    re = run_single_test(
        ('gen-mp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c, gamma))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['gen-mp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('dmo-acc-fw', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c, gamma))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['dmo-acc-fw'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('graph-cosamp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c, gamma))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['graph-cosamp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('cosamp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c, gamma))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['cosamp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('graph-iht', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c, gamma))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['graph-iht'] = [x_hat, list_run_time, list_loss, list_est_err]

    return trial_i, results

def generate_data(n, d, k, p=0.95, q=0.01, type='regular'):
    """
    Generate synthetic data where the first k features form one cluster with non-zero weights,
    and the remaining d-k form another (zero-weighted) cluster.
    
    Parameters:
    - n: number of samples
    - d: number of features
    - k: number of non-zero features
    - p: intra-cluster connection probability
    - q: inter-cluster connection probability
    - type: type of data ('regular', 'weights', 'correlation', 'correlation_weights')

    Returns:
    A tuple: (X, w, edges, costs, k) where
        X: design matrix (n x d)
        w: true weight vector (d,)
        edges: array of graph edges (m x 2)
        costs: edge weights array (m,)
        k: number of non-zero features
    """
    from scipy import sparse as sp

    assert 0 < k < d, "k must be between 0 and d"

    # Define feature clusters
    selected_cluster = np.arange(k)
    rest_cluster = np.arange(k, d)
    clusters = [selected_cluster, rest_cluster]

    # Build adjacency matrix
    A = sp.lil_matrix((d, d))
    for cluster in clusters:
        sz = len(cluster)
        block = (np.random.rand(sz, sz) < p).astype(int)
        np.fill_diagonal(block, 0)
        block = np.triu(block) + np.triu(block, 1).T
        for i, u in enumerate(cluster):
            for j, v in enumerate(cluster):
                A[u, v] = block[i, j]
    # Inter-cluster edges
    inter = (np.random.rand(len(selected_cluster), len(rest_cluster)) < q).astype(int)
    for i, u in enumerate(selected_cluster):
        for j, v in enumerate(rest_cluster):
            A[u, v] = inter[i, j]
            A[v, u] = inter[i, j]

    # Convert to edge list
    A_coo = A.tocoo()
    edges = np.vstack((A_coo.row, A_coo.col)).T
    costs = A_coo.data.astype(np.float64)

    # Generate true weight vector
    w = np.zeros(d)
    if type == 'regular':
        w[:k] = 1 / np.sqrt(k)
    elif type in ['weights', 'correlation_weights']:
        sign = np.random.choice([-1, 1], size=k)
        w[:k] = sign / np.sqrt(k)
        # add small variation
        w += np.random.normal(0, 0.1 * np.abs(w), size=d)

    # Generate design matrix
    if type in ['correlation', 'correlation_weights']:
        mean, cov = np.zeros(d), np.eye(d)
        correlated_ratio = 0.3
        # correlate selected and rest features
        sel_corr = np.random.choice(selected_cluster, int(correlated_ratio * k), replace=False)
        rest_corr = np.random.choice(rest_cluster, int(correlated_ratio * k), replace=False)
        for i in sel_corr:
            for j in rest_corr:
                cov[i, j] = cov[j, i] = 0.9
        # ensure positive semidefiniteness
        eigvals, eigvecs = np.linalg.eigh(cov)
        cov = eigvecs @ np.diag(np.maximum(eigvals, 0)) @ eigvecs.T
        X = np.random.multivariate_normal(mean, cov, size=n)
    else:
        X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)

    return X, w, edges, costs, k


def run_experiment5(n, type='regular', p=0.95, q=0.01, gamma=0.05):
    """
    Run trials for a single sample size and configuration.
    """
    step = 1
    num_trials = 2
    max_epochs = 50
    tol_algo = 1e-20
    g = 1
    d, k = 1000, 50

    para_space = []
    for trial_i in range(num_trials):
        X, w, edges, costs, s = generate_data(n, d, k, p, q, type)
        para_space.append((trial_i, X, w, edges, costs, s, g, max_epochs, tol_algo, step, gamma))

    # Use threads for inner parallelism
    num_workers = min(num_trials, NUM_CPUS)
    with ThreadPool(processes=num_workers) as pool:
        results = pool.map(sparse_learning_recovery, para_space)
    # results is list of (trial_i, res)
    return results

def analyze_results(results):
    support_recovery_rates = defaultdict(list)
    for trail_i, res in results:
        for method, [x_hat, _, _, _] in res.items():
            support_recovery_rates[method].append(recovery_accuracy(x_hat))
    return support_recovery_rates


def visualize_results(sample_sizes, support_recovery_rates, suffix=''):
    plt.figure(figsize=(10, 5))
    for method, rates in support_recovery_rates.items():
        plt.plot(sample_sizes, rates, label=method)
    plt.legend()
    plt.xlabel('Sample Size')
    plt.ylabel('Support Recovery Rate')
    plt.title(f'Support Recovery Rate vs. Sample Size ({suffix})')
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/recovery_curve_{suffix}.png')
    plt.close()


def run_configuration(config):
    data_type, p, q, gamma = config
    sample_sizes = np.arange(50, 100, 100)
    suffix = f"{data_type}_p{p}_q{q}_g{gamma}"
    # ensure dirs
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    # setup logging
    log_file = os.path.join('logs', f"experiment_{suffix}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Config start: {suffix}")

    # Containers for mean and raw accuracies
    mean_results = []  # list of dicts: method, n, mean_accuracy
    raw_results = { }

    for n in sample_sizes:
        logging.info(f"Sample size: {n}")
        try:
            results = run_experiment5(n, data_type, p, q, gamma)
            rates = analyze_results(results)
            # rates: method -> list of per-trial accuracies
            for method, vals in rates.items():
                # record raw list
                raw_results.setdefault(method, {})[n] = vals.copy()
                # record mean
                mean_rate = np.mean(vals)
                mean_results.append({'method': method, 'n': n, 'mean_accuracy': mean_rate})
                logging.info(f"{method} @ n={n}: mean accuracy={mean_rate:.4f}")
        except Exception:
            logging.exception(f"Error at sample size {n}")

    # save mean results to CSV
    import csv
    csv_file = os.path.join('results', f"mean_accuracy_{suffix}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'n', 'mean_accuracy'])
        writer.writeheader()
        writer.writerows(mean_results)
    logging.info(f"Saved mean CSV: {csv_file}")

    # save raw accuracy dict as pickle
    pkl_file = os.path.join('results', f"raw_accuracy_{suffix}.pkl")
    with open(pkl_file, 'wb') as pf:
        pkl.dump(raw_results, pf)
    logging.info(f"Saved raw pickle: {pkl_file}")
    logging.info(f"Config done: {suffix}")
    
def main():
    types = ['regular', 'weights', 'correlation', 'correlation_weights']
    # p_values = [0.9, 0.7, 0.5, 0.3]
    # q_values = [0.05, 0.1, 0.2, 0.3]
    # gamma_values = [0.5, 1.0]
    p_values = [0.25]
    q_values = [0.01]
    gamma_values = [1.0]

    configs = list(product(types, p_values, q_values, gamma_values))
    num_workers = min(len(configs), NUM_CPUS)
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_configuration, configs)

if __name__ == '__main__':
    main()
