import numpy as np
import scipy.sparse as sp
from bokeh.palettes import Category20_20, Category20b_20, Accent8
from matplotlib import collections  as mc
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq

args = parse_args()

def plot_graph_embedding(y_emb, labels, adj, line_alpha=0.2, s=7, title=""):
    """
    Plots the visualization of graph-structured data
    Args:
        y_emb (np.array): low dimensional map of data points, matrix of size n x 2
        labels (np.array): underlying class labels, matrix of size n x 1
        adj (scipy csr matrix): adjacency matrix
    """
    labels = np.array([int(l) for l in labels])
    # adj = sp.coo_matrix(adj)
    adj = adj.tocoo()
    colormap = np.array(Category20_20 + Category20b_20 + Accent8)

    f, ax = plt.subplots(1, sharex='col', figsize=(6, 4), dpi=800)

    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)

    # Plot edges
    p0 = y_emb[adj.row, :]
    p1 = y_emb[adj.col, :]
    p_0 = [tuple(row) for row in p0]
    p_1 = [tuple(row) for row in p1]

    classA = labels[adj.row]
    classB = labels[adj.col]
    mask = classA == classB
    edge_colormask = mask * (classA + 1) - 1

    lines = list(zip(p_0, p_1))
    lc = mc.LineCollection(lines, linewidths=0.5, colors=colormap[edge_colormask])
    lc.set_alpha(line_alpha)
    ax.add_collection(lc)

    ax.scatter(y_emb[:, 0], y_emb[:, 1], s=s, c=colormap[labels])

    ax.margins(0.1, 0.1)
    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig('weights/' + args.dataset + '_' + args.alg_type + '_' + args.adj_type + '_' + str(args.sub_version) + '_vis.png')
    plt.show()


# args.dataset = 'yelp'
# args.alg_type = 'cgan'
# args.adj_type = 'appnp-ns'
# args.sub_version = 4.3

# args.dataset = 'gowalla'
# args.alg_type = 'ngcf'
# args.adj_type = 'appnp-ns'
# args.sub_version = 1.2271

args.dataset = 'gowalla'
args.alg_type = 'ngcf'
args.adj_type = 'appnp-ns'
args.sub_version = 1.224

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, num_epochs=args.epoch, dataset=args.dataset)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
adj_mat, _, _ = data_generator.get_adj_mat()
user_emb = np.load('weights/' + args.dataset + '_' + args.alg_type + '_' + args.adj_type + '_' + str(args.sub_version) + '_user_emb.npy')[10000:15000]
item_emb = np.load('weights/' + args.dataset + '_' + args.alg_type + '_' + args.adj_type + '_' + str(args.sub_version) + '_item_emb.npy')[10000:13000]
X = np.concatenate([user_emb, item_emb], axis=0)
print(user_emb.shape, item_emb.shape, X.shape)
# labels = np.array([0] * USR_NUM + [1] * ITEM_NUM)
tsne = TSNE(n_components=2, perplexity=100, n_iter=5000)
print('start transforming')
tsne.fit_transform(X)
y_emb = tsne.embedding_
# y_emb = X
print('start plotting')
# plot_graph_embedding(y_emb, labels, adj_mat)
np.save('weights/tsne.npy', y_emb)
plt.figure(figsize=(24, 16))
plt.scatter(y_emb[:5000, 0], y_emb[:5000, 1], marker='o', c='b')
plt.scatter(y_emb[5000:, 0], y_emb[5000:, 1], marker='x', c='r')
plt.show()
