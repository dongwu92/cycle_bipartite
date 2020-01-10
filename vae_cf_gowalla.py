import os
import shutil
import sys
import time
import numpy as np
from scipy import sparse
# import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sn

sn.set()
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
import bottleneck as bn
from utility.batch_test import *


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)
    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


class MultiDAE(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]

        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self):

        self.construct_weights()

        saver, logits = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        # per-user average negative log-likelihood
        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        loss = neg_ll + 2 * reg_var

        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h

    def construct_weights(self):
        self.weights = []
        self.biases = []

        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i + 1)
            bias_key = "bias_{}".format(i + 1)
            self.weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            self.biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            # add summary stats
            tf.summary.histogram(weight_key, self.weights[-1])
            tf.summary.histogram(bias_key, self.biases[-1])


class MultiVAE(MultiDAE):

    def construct_placeholders(self):
        super(MultiVAE, self).construct_placeholders()

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph,
            axis=-1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)

        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var

        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('KL', KL)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.summary.merge_all()

        return saver, logits, neg_ELBO, train_op, merged

    def q_graph(self):
        mu_q, std_q, KL = None, None, None

        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph * \
                    epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return tf.train.Saver(), logits, KL

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            bias_key = "bias_q_{}".format(i + 1)
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])
        self.weights_p, self.biases_p = [], []
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i + 1)
            bias_key = "bias_p_{}".format(i + 1)
            self.weights_p.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            self.biases_p.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])


def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


if __name__ == "__main__":
    DATA_DIR = './Data/ml-20m/'
    pro_dir = os.path.join(DATA_DIR, 'pro_sg')

    '''t0 = time.time()
    if not os.path.exists(pro_dir):
        t10 = time.time()
        os.makedirs(pro_dir)

        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
        raw_data = raw_data[raw_data['rating'] > 3.5]
        raw_data.head()
        raw_data, user_activity, item_popularity = filter_triplets(raw_data)
        sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
        print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
              (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
        unique_uid = user_activity.index
        np.random.seed(98765)
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]
        with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
            for uid in unique_uid:
                f.write('%s\n' % uid)
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        n_users = unique_uid.size
        n_heldout_users = 10000
        tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
        vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
        te_users = unique_uid[(n_users - n_heldout_users):]
        train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
        unique_sid = pd.unique(train_plays['movieId'])
        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)
        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

        vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
        vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
        test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
        test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
        test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

        def numerize(tp):
            uid = list(map(lambda x: profile2id[x], tp['userId']))
            sid = list(map(lambda x: show2id[x], tp['movieId']))
            return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

        train_data = numerize(train_plays)
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
        vad_data_tr = numerize(vad_plays_tr)
        vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
        vad_data_te = numerize(vad_plays_te)
        vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
        test_data_tr = numerize(test_plays_tr)
        test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
        test_data_te = numerize(test_plays_te)
        test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
        print("preprocessing...", time.time() - t10)'''

    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    n_items = len(unique_sid)
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    unique_uid = list()
    with open(os.path.join(pro_dir, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                               os.path.join(pro_dir, 'validation_te.csv'))
    N = train_data.shape[0]
    idxlist = list(range(N))

    # training batch size
    batch_size = 8192
    batches_per_epoch = int(np.ceil(float(N) / batch_size))

    N_vad = vad_data_tr.shape[0]
    idxlist_vad = range(N_vad)

    # validation batch size (since the entire validation set might not fit into GPU memory)
    batch_size_vad = 10000

    # the total number of gradient updates for annealing
    total_anneal_steps = 200000
    # largest annealing parameter
    anneal_cap = 0.2

    p_dims = [200, 600, n_items]
    '''tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam=0.0, random_seed=98765)
    saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()
    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))

    log_dir = '../log/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps / 1000, anneal_cap, arch_str)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    print("log directory: %s" % log_dir)
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
    chkpt_dir = '../chkpt/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps / 1000, anneal_cap, arch_str)
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
    print("chkpt directory: %s" % chkpt_dir)
    print("Initializing...", time.time() - t0)

    n_epochs = 200
    ndcgs_vad = []
    t2 = time.time()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        best_ndcg = -np.inf

        update_count = 0.0

        for epoch in range(n_epochs):
            t31 = time.time()
            np.random.shuffle(idxlist)
            t32 = time.time()
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                t21 = time.time()
                end_idx = min(st_idx + batch_size, N)
                X = train_data[idxlist[st_idx:end_idx]]
                t211 = time.time()

                if sparse.isspmatrix(X):
                    X = X.toarray()
                t212 = time.time()
                X = X.astype('float32')
                print(X.shape, '././././', vae.dims)
                t213 = time.time()

                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_cap
                t214 = time.time()

                feed_dict = {vae.input_ph: X,
                             vae.keep_prob_ph: 0.5,
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1}
                t22 = time.time()
                print('...', t211 - t21, t212 - t211, t213 - t212, t214 - t213, t22 - t214)
                sess.run(train_op_var, feed_dict=feed_dict)
                t23 = time.time()

                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_train,
                                               global_step=epoch * batches_per_epoch + bnum)
                t24 = time.time()
                print("session running...", t22 - t21, t23 - t22, t24 - t23)

                update_count += 1

            t33 = time.time()
            # compute validation NDCG
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))

            t34 = time.time()
            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_vad.append(ndcg_)
            merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_val, epoch)
            t35 = time.time()
            print("validating___", ndcg_, t32 - t31, t33 - t32, t34 - t33, t35 - t34)

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                saver.save(sess, '{}/model'.format(chkpt_dir))
                best_ndcg = ndcg_'''

    test_data_tr, test_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'),
        os.path.join(pro_dir, 'test_te.csv'))
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)
    batch_size_test = 2000
    tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam=0.0)
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    saver, logits_var, _, _, _ = vae.build_graph()
    chkpt_dir = '../chkpt/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps / 1000, anneal_cap, arch_str)
    print("chkpt directory: %s" % chkpt_dir)
    n20_list, n100_list, r20_list, r50_list = [], [], [], []

    t50 = time.time()
    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data_tr[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            t41 = time.time()
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
            print(pred_val.shape, test_data_te.shape)
            t42 = time.time()
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            t43 = time.time()
            n20_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
            t44 = time.time()
            n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
            t45 = time.time()
            r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
            t46 = time.time()
            r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))
            t47 = time.time()
            print(">>>>>>", t42 - t41, t44 - t43, t45 - t44, t46 - t45, t47 - t46)
    t51 = time.time()
    print("======", t51 - t50)
    n20_list = np.concatenate(n20_list)
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@20=%.5f (%.5f)" % (np.mean(n20_list), np.std(n20_list) / np.sqrt(len(n20_list))))
    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

    '''p_dims = [200, n_items]
    tf.reset_default_graph()
    dae = MultiDAE(p_dims, lam=0.01 / batch_size, random_seed=98765)
    saver, logits_var, loss_var, train_op_var, merged_var = dae.build_graph()
    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

    arch_str = "I-%s-I" % ('-'.join([str(d) for d in dae.dims[1:-1]]))
    log_dir = '../log/ml-20m/DAE/{}'.format(arch_str)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    print("log directory: %s" % log_dir)
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
    n_epochs = 200
    ndcgs_vad = []

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        best_ndcg = -np.inf

        for epoch in range(n_epochs):
            np.random.shuffle(idxlist)
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
                X = train_data[idxlist[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                feed_dict = {dae.input_ph: X,
                            dae.keep_prob_ph: 0.5}
                sess.run(train_op_var, feed_dict=feed_dict)

                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_train, global_step=epoch * batches_per_epoch + bnum)

                    # compute validation NDCG
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                pred_val = sess.run(logits_var, feed_dict={dae.input_ph: X})
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))

            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_vad.append(ndcg_)
            merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_val, epoch)

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                saver.save(sess, '{}/model'.format(chkpt_dir))
                best_ndcg = ndcg_

    tf.reset_default_graph()
    dae = MultiDAE(p_dims, lam=0.01 / batch_size)
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in dae.dims[1:-1]]))
    saver, logits_var, _, _, _ = dae.build_graph()
    chkpt_dir = '../chkpt/ml-20m/DAE/{}'.format(arch_str)
    print("chkpt directory: %s" % chkpt_dir)
    n100_list, r20_list, r50_list = [], [], []

    t0 = time.time()
    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data_tr[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            t01 = time.time()
            pred_val = sess.run(logits_var, feed_dict={dae.input_ph: X})
            t02 = time.time()
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            t03 = time.time()
            n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
            t04 = time.time()
            r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
            t05 = time.time()
            r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))
            t06 = time.time()
            print(">>>>>>", t02 - t01, t04 - t03, t05 - t04, t06 - t05)

    t1 = time.time()
    print("======", t1 - t0)
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))'''