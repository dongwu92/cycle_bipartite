import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *


def _init_weights(pretrain_data, n_users, n_items, n_layers):
    all_weights = dict()

    initializer = tf.contrib.layers.xavier_initializer()

    if pretrain_data is None:
        all_weights['user_embedding'] = tf.Variable(initializer([n_users, args.embed_size]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([n_items, args.embed_size]), name='item_embedding')
        print('using xavier initialization')
    else:
        all_weights['user_embedding'] = tf.Variable(initial_value=pretrain_data['user_embed'], trainable=True,
                                                    name='user_embedding', dtype=tf.float32)
        all_weights['item_embedding'] = tf.Variable(initial_value=pretrain_data['item_embed'], trainable=True,
                                                    name='item_embedding', dtype=tf.float32)
        print('using pretrained initialization')

    if args.alg_type == 'ngcf':
        for k in range(n_layers):
            weight_size_list = [args.embed_size] + eval(args.layer_size)
            all_weights['W_gc_%d' % k] = tf.Variable(
                    initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                    initializer([1, weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                    initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                    initializer([1, weight_size_list[k + 1]]), name='b_bi_%d' % k)
    elif args.alg_type == 'appnp':
        # funits = [args.embed_size] + eval(args.appnp_f_units) + [args.appnp_c_units]
        # for k in range(len(funits) - 1):
        #     all_weights['W_fappnp_%d' % k] = tf.Variable(
        #             initializer([funits[k], funits[k + 1]]), name='W_fappnp_%d' % k)
        #     all_weights['b_fappnp_%d' % k] = tf.Variable(
        #             initializer([1, funits[k + 1]]), name='b_fappnp_%d' % k)
        if args.sub_version == 1.7:
            for k in range(args.n_layers_generator):
                all_weights['gW_uv_%d' % k] = tf.Variable(initializer([args.embed_size, args.embed_size]),
                                                          name='gW_uv_%d' % k)
                all_weights['gb_uv_%d' % k] = tf.Variable(initializer([args.embed_size]), name='gb_uv_%d' % k)
                all_weights['gW_vu_%d' % k] = tf.Variable(initializer([args.embed_size, args.embed_size]),
                                                          name='gW_vu_%d' % k)
                all_weights['gb_vu_%d' % k] = tf.Variable(initializer([args.embed_size]), name='gb_vu_%d' % k)
        elif args.sub_version == 1.8:
            for k in range(args.n_layers_generator):
                all_weights['gW_%d' % k] = tf.Variable(initializer([args.embed_size, args.embed_size]),
                                                          name='gW_%d' % k)
                all_weights['gb_%d' % k] = tf.Variable(initializer([args.embed_size]), name='gb_%d' % k)
    elif args.alg_type == 'appnpuv' or args.alg_type == 'appnpcgan':
        for k in range(args.n_layers_generator):
            all_weights['gW_uv_%d' % k] = tf.Variable(initializer([args.embed_size, args.embed_size]), name='gW_uv_%d' % k)
            all_weights['gb_uv_%d' % k] = tf.Variable(initializer([args.embed_size]), name='gb_uv_%d' % k)
            all_weights['gW_vu_%d' % k] = tf.Variable(initializer([args.embed_size, args.embed_size]), name='gW_vu_%d' % k)
            all_weights['gb_vu_%d' % k] = tf.Variable(initializer([args.embed_size]), name='gb_vu_%d' % k)
        for k in range(args.n_layers_discriminator):
            all_weights['dW_uv_%d' % k] = tf.Variable(initializer([args.embed_size, args.embed_size]), name='dW_uv_%d' % k)
            all_weights['db_uv_%d' % k] = tf.Variable(initializer([args.embed_size]), name='db_uv_%d' % k)
            all_weights['dW_vu_%d' % k] = tf.Variable(initializer([args.embed_size, args.embed_size]), name='dW_vu_%d' % k)
            all_weights['db_vu_%d' % k] = tf.Variable(initializer([args.embed_size]), name='db_vu_%d' % k)

    return all_weights

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def _dropout_sparse(X, keep_prob, n_nonzero_elems):
    noise_shape = [n_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(X, dropout_mask)

    return pre_out * tf.div(1., keep_prob)

def _split_A_hat_node_dropout(X, node_dropout, n_fold, n_users, n_items):
    A_fold_hat = []

    fold_len = (n_users + n_items) // n_fold
    for i_fold in range(n_fold):
        start = i_fold * fold_len
        if i_fold == n_fold - 1:
            end = n_users + n_items
        else:
            end = (i_fold + 1) * fold_len

        # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        temp = _convert_sp_mat_to_sp_tensor(X[start:end])
        n_nonzero_temp = X[start:end].count_nonzero()
        A_fold_hat.append(_dropout_sparse(temp, 1 - node_dropout[0], n_nonzero_temp))

    return A_fold_hat

def _split_A_hat(X, n_fold, n_users, n_items):
    A_fold_hat = []

    fold_len = (n_users + n_items) // n_fold
    for i_fold in range(n_fold):
        start = i_fold * fold_len
        if i_fold == n_fold - 1:
            end = n_users + n_items
        else:
            end = (i_fold + 1) * fold_len

        A_fold_hat.append(_convert_sp_mat_to_sp_tensor(X[start:end]))
    return A_fold_hat

def _create_ngcf_embed(norm_adj, weights, mess_dropout, node_dropout, n_layers, n_fold, n_users, n_items):
    # Generate a set of adjacency sub-matrix.
    if args.node_dropout_flag:
        # node dropout.
        A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
    else:
        A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)

    ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
    all_embeddings = [ego_embeddings]
    z0_embeddings = ego_embeddings

    if args.sub_version == 0:
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, weights['W_gc_%d' % k]) + weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, weights['W_bi_%d' % k]) + weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 0.1:
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = tf.matmul(side_embeddings, weights['W_gc_0'])  # + weights['b_gc_%d' % k]
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 0.2:
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = (1 - args.appnp_alpha) * tf.matmul(side_embeddings, weights['W_gc_0']) + \
                            args.appnp_alpha * z0_embeddings # + weights['b_gc_%d' % k]
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    return None, None

#     # Generate a set of adjacency sub-matrix.
#     if args.node_dropout_flag:
#         # node dropout.
#         A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
#     else:
#         A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
#
#     ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
#     all_embeddings = [ego_embeddings]
#
#     for k in range(0, n_layers):
#         temp_embed = []
#         for f in range(n_fold):
#             temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
#
#         # sum messages of neighbors.
#         side_embeddings = tf.concat(temp_embed, 0)
#         # transformed sum messages of neighbors.
#         ego_embeddings = tf.matmul(side_embeddings, weights['W_gc_0']) # + weights['b_gc_%d' % k]
#         # message dropout.
#         # ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])
#
#         # normalize the distribution of embeddings.
#         norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
#         all_embeddings += [norm_embeddings]
#     all_embeddings = tf.concat(all_embeddings, 1)
#     u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
#     return u_g_embeddings, i_g_embeddings

    
# version 1.0
# def _create_appnp_embed(norm_adj, weights, n_users, n_items, keep_prob=0.5):
#     ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
#     all_embeddings = [ego_embeddings]
#     local_logits = ego_embeddings
#     num_funits = len(eval(args.appnp_f_units)) + 1
#     for k in range(num_funits):
#         local_logits = tf.nn.relu(tf.matmul(local_logits, weights['W_fappnp_%d' % k]) + weights['b_fappnp_%d' % k])
#         local_logits = tf.nn.dropout(local_logits, keep_prob)  # ckpt: keep prob for appnp mixed_dropout
#     coo = norm_adj.tocoo()
#     coo_indices = np.mat([coo.row, coo.col]).transpose()
#     A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
#     all_embeddings.append(local_logits)
#     Zs_prop = local_logits
#     for _ in range(args.appnp_niter):
#         A_drop_val = tf.nn.dropout(A_hat_tf.values, keep_prob)
#         A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
#         Zs_prop = tf.sparse_tensor_dense_matmul(A_drop, Zs_prop) + args.appnp_alpha * local_logits
#     all_embeddings.append(Zs_prop)
#     all_embeddings = tf.concat(all_embeddings, 1)
#     u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
#     return u_g_embeddings, i_g_embeddings

# version 1.1
# def _create_appnp_embed(norm_adj, weights, n_users, n_items, keep_prob=0.5):
#     ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
#     all_embeddings = [ego_embeddings]
#     local_logits = ego_embeddings
#     num_funits = len(eval(args.appnp_f_units)) + 1
#     for k in range(num_funits):
#         local_logits = tf.nn.relu(tf.matmul(local_logits, weights['W_fappnp_%d' % k]) + weights['b_fappnp_%d' % k])
#         local_logits = tf.nn.dropout(local_logits, keep_prob)  # ckpt: keep prob for appnp mixed_dropout
#     coo = norm_adj.tocoo()
#     coo_indices = np.mat([coo.row, coo.col]).transpose()
#     A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
#     all_embeddings.append(local_logits)
#     Zs_prop = local_logits
#     for _ in range(args.appnp_niter):
#         A_drop_val = tf.nn.dropout(A_hat_tf.values, keep_prob)
#         A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
#         Zs_prop = tf.sparse_tensor_dense_matmul(A_drop, Zs_prop) + args.appnp_alpha * local_logits
#     all_embeddings.append(Zs_prop)
#     all_embeddings = tf.concat(all_embeddings, 1)
#     u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
#     return u_g_embeddings, i_g_embeddings

# version 1.2
# def _create_appnp_embed(norm_adj, weights, n_users, n_items, keep_prob=0.5):
    # ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
    # all_embeddings = [ego_embeddings]
    # coo = norm_adj.tocoo()
    # coo_indices = np.mat([coo.row, coo.col]).transpose()
    # A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
    # Zs_prop = ego_embeddings
    # for _ in range(args.appnp_niter):
    #     A_drop_val = tf.nn.dropout(A_hat_tf.values, keep_prob)
    #     A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
    #     Zs_prop = tf.sparse_tensor_dense_matmul(A_drop, Zs_prop) + args.appnp_alpha * ego_embeddings
    #     all_embeddings.append(Zs_prop)
    # all_embeddings = tf.concat(all_embeddings, 1)
    # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
    # return u_g_embeddings, i_g_embeddings

# version 1.2.1
# def _create_appnp_embed(norm_adj, weights, n_users, n_items, keep_prob=0.8):
#     ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
#     all_embeddings = [ego_embeddings]
#     coo = norm_adj.tocoo()
#     coo_indices = np.mat([coo.row, coo.col]).transpose()
#     A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
#     Zs_prop = ego_embeddings
#     for _ in range(args.appnp_niter):
#         A_drop_val = tf.nn.dropout(A_hat_tf.values, keep_prob)
#         A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
#         Zs_prop = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop, Zs_prop) + args.appnp_alpha * ego_embeddings
#         all_embeddings.append(Zs_prop)
#     all_embeddings = tf.concat(all_embeddings, 1)
#     u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
#     return u_g_embeddings, i_g_embeddings

# version 1.3
# def _create_appnp_embed(norm_adj, weights, n_users, n_items, keep_prob=0.5):
#     ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
#     all_embeddings = [ego_embeddings]
#     coo = norm_adj.tocoo()
#     coo_indices = np.mat([coo.row, coo.col]).transpose()
#     A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
#     Zs_prop = ego_embeddings
#     for _ in range(args.appnp_niter):
#         A_drop_val = tf.nn.dropout(A_hat_tf.values, keep_prob)
#         A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
#         # Zs_prop = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop, Zs_prop) + args.appnp_alpha * ego_embeddings
#         Zs_prop = tf.sparse_tensor_dense_matmul(A_drop, Zs_prop)
#         all_embeddings.append(Zs_prop)
#     all_embeddings = tf.concat(all_embeddings, 1)
#     u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
#     return u_g_embeddings, i_g_embeddings

# version 1.5
# def _create_appnp_embed(norm_adj, weights, n_users, n_items, keep_prob=0.5):
#     initializer = tf.contrib.layers.xavier_initializer()
#     adptive_weight0 = tf.Variable(initializer([1]), name='aw0')
#     ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
#     all_embeddings = [adptive_weight0 * ego_embeddings]
#     coo = norm_adj.tocoo()
#     coo_indices = np.mat([coo.row, coo.col]).transpose()
#     A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
#     Zs_prop = ego_embeddings
#     for k in range(args.appnp_niter):
#         A_drop_val = tf.nn.dropout(A_hat_tf.values, keep_prob)
#         A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
#         Zs_prop = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop, Zs_prop) + args.appnp_alpha * ego_embeddings
#         # Zs_prop = tf.sparse_tensor_dense_matmul(A_drop, Zs_prop)
#         # all_embeddings.append(Zs_prop)
#         adptive_weight = tf.Variable(initializer([1]), name='aw%d' % k)
#         all_embeddings.append(adptive_weight * Zs_prop)
#     all_embeddings = tf.concat(all_embeddings, 1)
#     u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
#     return u_g_embeddings, i_g_embeddings

# version 1.5.1 depressed
#         ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
#         all_embeddings = [ego_embeddings]
#         coo = norm_adj.tocoo()
#         coo_indices = np.mat([coo.row, coo.col]).transpose()
#         A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
#         A_drop_val = tf.nn.dropout(A_hat_tf.values, args.appnp_keepprob)
#         A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
#         Zs_prop = ego_embeddings
#         for _ in range(args.appnp_niter):
#             Zs_prop = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop,
#                                                                              Zs_prop) + args.appnp_alpha * ego_embeddings
#             all_embeddings.append(Zs_prop)
#         all_embeddings = tf.concat(all_embeddings, 1)
#         u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
#         return u_g_embeddings, i_g_embeddings

# version 1.2.1
def _create_appnp_embed(norm_adj, weights, n_users, n_items):
    if args.sub_version == 1.21:
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        coo = norm_adj.tocoo()
        coo_indices = np.mat([coo.row, coo.col]).transpose()
        A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
        Zs_prop = ego_embeddings
        for _ in range(args.appnp_niter):
            A_drop_val = tf.nn.dropout(A_hat_tf.values, args.appnp_keepprob)
            A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
            Zs_prop = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop,
                                                                             Zs_prop) + args.appnp_alpha * ego_embeddings
            all_embeddings.append(Zs_prop)
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.6:
        # ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [[weights['user_embedding']], [weights['item_embedding']]]
        coo = norm_adj.tocoo()
        coo_indices = np.mat([coo.row, coo.col]).transpose()
        A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
        Zs_prop_user, Zs_prop_item = weights['user_embedding'], weights['item_embedding']
        A_hat_tf_user_user = tf.sparse.slice(A_hat_tf, [0, 0], [n_users, n_users])
        A_hat_tf_user_item = tf.sparse.slice(A_hat_tf, [0, n_users], [n_users, n_items])
        A_hat_tf_item_user = tf.sparse.slice(A_hat_tf, [n_users, 0], [n_items, n_users])
        A_hat_tf_item_item = tf.sparse.slice(A_hat_tf, [n_users, n_users], [n_items, n_items])
        for _ in range(args.appnp_niter):
            A_drop_val_user_user = tf.nn.dropout(A_hat_tf_user_user.values, args.appnp_keepprob)
            A_drop_val_user_item = tf.nn.dropout(A_hat_tf_user_item.values, args.appnp_keepprob)
            A_drop_val_item_user = tf.nn.dropout(A_hat_tf_item_user.values, args.appnp_keepprob)
            A_drop_val_item_item = tf.nn.dropout(A_hat_tf_item_item.values, args.appnp_keepprob)
            A_drop_user_user = tf.SparseTensor(A_hat_tf_user_user.indices, A_drop_val_user_user, A_hat_tf_user_user.dense_shape)
            A_drop_user_item = tf.SparseTensor(A_hat_tf_user_item.indices, A_drop_val_user_item, A_hat_tf_user_item.dense_shape)
            A_drop_item_user = tf.SparseTensor(A_hat_tf_item_user.indices, A_drop_val_item_user, A_hat_tf_item_user.dense_shape)
            A_drop_item_item = tf.SparseTensor(A_hat_tf_item_item.indices, A_drop_val_item_item, A_hat_tf_item_item.dense_shape)
            Zs_prop_user = (1 - args.appnp_alpha) * (tf.sparse_tensor_dense_matmul(A_drop_user_user, Zs_prop_user) + \
                            tf.sparse_tensor_dense_matmul(A_drop_user_item, Zs_prop_item)) + \
                            args.appnp_alpha * weights['user_embedding']
            Zs_prop_item = (1 - args.appnp_alpha) * (tf.sparse_tensor_dense_matmul(A_drop_item_user, Zs_prop_user) + \
                            tf.sparse_tensor_dense_matmul(A_drop_item_item, Zs_prop_item)) + \
                            args.appnp_alpha * weights['item_embedding']
            all_embeddings[0].append(Zs_prop_user)
            all_embeddings[1].append(Zs_prop_item)
        # all_embeddings = tf.concat(all_embeddings, 1)
        # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        u_g_embeddings = tf.concat(all_embeddings[0], 1)
        i_g_embeddings = tf.concat(all_embeddings[1], 1)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.61:
        # ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [[weights['user_embedding']], [weights['item_embedding']]]
        coo = norm_adj.tocoo()
        coo_indices = np.mat([coo.row, coo.col]).transpose()
        A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
        Zs_prop_user, Zs_prop_item = weights['user_embedding'], weights['item_embedding']
        # A_hat_tf_user_user = tf.sparse.slice(A_hat_tf, [0, 0], [n_users, n_users])
        A_hat_tf_user_item = tf.sparse.slice(A_hat_tf, [0, n_users], [n_users, n_items])
        A_hat_tf_item_user = tf.sparse.slice(A_hat_tf, [n_users, 0], [n_items, n_users])
        # A_hat_tf_item_item = tf.sparse.slice(A_hat_tf, [n_users, n_users], [n_items, n_items])
        for _ in range(args.appnp_niter):
            # A_drop_val_user_user = tf.nn.dropout(A_hat_tf_user_user.values, args.appnp_keepprob)
            A_drop_val_user_item = tf.nn.dropout(A_hat_tf_user_item.values, args.appnp_keepprob)
            A_drop_val_item_user = tf.nn.dropout(A_hat_tf_item_user.values, args.appnp_keepprob)
            # A_drop_val_item_item = tf.nn.dropout(A_hat_tf_item_item.values, args.appnp_keepprob)
            # A_drop_user_user = tf.SparseTensor(A_hat_tf_user_user.indices, A_drop_val_user_user, A_hat_tf_user_user.dense_shape)
            A_drop_user_item = tf.SparseTensor(A_hat_tf_user_item.indices, A_drop_val_user_item, A_hat_tf_user_item.dense_shape)
            A_drop_item_user = tf.SparseTensor(A_hat_tf_item_user.indices, A_drop_val_item_user, A_hat_tf_item_user.dense_shape)
            # A_drop_item_item = tf.SparseTensor(A_hat_tf_item_item.indices, A_drop_val_item_item, A_hat_tf_item_item.dense_shape)
            Zs_prop_user = tf.sparse_tensor_dense_matmul(A_drop_user_item, Zs_prop_item) + \
                            args.appnp_alpha * weights['user_embedding']
            Zs_prop_item = tf.sparse_tensor_dense_matmul(A_drop_item_user, Zs_prop_user) + \
                            args.appnp_alpha * weights['item_embedding']
            all_embeddings[0].append(Zs_prop_user)
            all_embeddings[1].append(Zs_prop_item)
        # all_embeddings = tf.concat(all_embeddings, 1)
        # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        u_g_embeddings = tf.concat(all_embeddings[0], 1)
        i_g_embeddings = tf.concat(all_embeddings[1], 1)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.62:
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        coo = norm_adj.tocoo()
        coo_indices = np.mat([coo.row, coo.col]).transpose()
        A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
        A_hat_tf_user = tf.sparse.slice(A_hat_tf, [0, 0], [n_users, n_users + n_items])
        A_hat_tf_item = tf.sparse.slice(A_hat_tf, [n_users, 0], [n_items, n_users + n_items])
        Zs_prop = ego_embeddings
        for _ in range(args.appnp_niter):
            A_drop_val_user = tf.nn.dropout(A_hat_tf_user.values, args.appnp_keepprob)
            A_drop_val_item = tf.nn.dropout(A_hat_tf_item.values, args.appnp_keepprob)
            A_drop_user = tf.SparseTensor(A_hat_tf_user.indices, A_drop_val_user, A_hat_tf_user.dense_shape)
            A_drop_item = tf.SparseTensor(A_hat_tf_item.indices, A_drop_val_item, A_hat_tf_item.dense_shape)
            Zs_prop_user = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_user,
                                                                             Zs_prop) + args.appnp_alpha * weights['user_embedding']
            Zs_prop_item = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_item,
                                                                             Zs_prop) + args.appnp_alpha * weights['item_embedding']
            Zs_prop = tf.concat([Zs_prop_user, Zs_prop_item], axis=0)
            all_embeddings.append(Zs_prop)
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.7:
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        coo = norm_adj.tocoo()
        coo_indices = np.mat([coo.row, coo.col]).transpose()
        A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
        Zs_prop_user, Zs_prop_item = weights['user_embedding'], weights['item_embedding']
        for _ in range(args.appnp_niter):
            A_drop_val = tf.nn.dropout(A_hat_tf.values, args.appnp_keepprob)
            A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
            fake_user = appnp_generator(Zs_prop_item, weights, direction='vu')
            fake_item = appnp_generator(Zs_prop_user, weights, direction='uv')
            Zs_prop = tf.concat([fake_item, fake_user], axis=0)
            Zs_prop = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop,
                                                    Zs_prop) + args.appnp_alpha * ego_embeddings
            all_embeddings.append(Zs_prop)
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.8:
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        coo = norm_adj.tocoo()
        coo_indices = np.mat([coo.row, coo.col]).transpose()
        A_hat_tf = tf.SparseTensor(coo_indices, np.array(coo.data, dtype=np.float32), coo.shape)
        Zs_prop = ego_embeddings
        for _ in range(args.appnp_niter):
            A_drop_val = tf.nn.dropout(A_hat_tf.values, args.appnp_keepprob)
            A_drop = tf.SparseTensor(A_hat_tf.indices, A_drop_val, A_hat_tf.dense_shape)
            Zs_prop = appnp_generator_all(Zs_prop, weights)
            Zs_prop = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop,
                                                                             Zs_prop) + args.appnp_alpha * ego_embeddings
            all_embeddings.append(Zs_prop)
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    return None, None

def appnp_generator(inp, weights, direction='uv', activation=None):
    hidden = inp
    for k in range(args.n_layers_generator):
        gw = weights['gW_uv_%d' % k] if direction == 'uv' else weights['gW_vu_%d' % k]
        gb = weights['gb_uv_%d' % k] if direction == 'vu' else weights['gb_vu_%d' % k]
        hidden = tf.matmul(hidden, gw) + gb
        if activation is not None:
            hidden = activation(hidden)
    return hidden

def appnp_generator_all(inp, weight, activation=None):
    hidden = inp
    for k in range(args.n_layers_generator):
        hidden = tf.matmul(hidden, weight['gW_%d' % k]) + weight['gb_%d' % k]
        if activation is not None:
            hidden = activation(hidden)
    return hidden

def appnp_discriminator(inp, weights, domain='u', activation=tf.nn.relu):
    hidden = inp
    for k in range(args.n_layers_generator):
        act = tf.nn.sigmoid if k == args.n_layers_discriminator - 1 else activation
        gw = weights['dW_uv_%d' % k] if domain == 'u' else weights['dW_vu_%d' % k]
        gb = weights['db_uv_%d' % k] if domain == 'v' else weights['db_vu_%d' % k]
        hidden = act(tf.matmul(hidden, gw) + gb)
    return hidden

# version 2.0
def _create_appnpuv_embed(config, weights, n_users, n_items, keep_prob=0.5):
    if args.sub_version == 2.0:
        adj_uv, adj_vu = config['adj_uv'], config['adj_vu']
        all_embeddings = [[weights['user_embedding']], [weights['item_embedding']]]
        coo_uv, coo_vu = adj_uv.tocoo(), adj_vu.tocoo()
        coo_indices_uu = np.mat([coo_uv.row, coo_uv.col]).transpose()
        coo_indices_vv = np.mat([coo_vu.row, coo_vu.col]).transpose()
        A_hat_uv = tf.SparseTensor(coo_indices_uu, np.array(coo_uv.data, dtype=np.float32), coo_uv.shape)
        A_hat_vu = tf.SparseTensor(coo_indices_vv, np.array(coo_vu.data, dtype=np.float32), coo_vu.shape)
        Zs_prop_user, Zs_prop_item = weights['user_embedding'], weights['item_embedding']
        for _ in range(args.appnp_niter):
            A_drop_val_uv = tf.nn.dropout(A_hat_uv.values, keep_prob)
            A_drop_val_vu = tf.nn.dropout(A_hat_vu.values, keep_prob)
            A_drop_uv = tf.SparseTensor(A_hat_uv.indices, A_drop_val_uv, A_hat_uv.dense_shape)
            A_drop_vu = tf.SparseTensor(A_hat_vu.indices, A_drop_val_vu, A_hat_vu.dense_shape)
            Zs_prop_user = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_uv, Zs_prop_item) + \
                    args.appnp_alpha * weights['user_embedding']
            Zs_prop_item = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_vu, Zs_prop_user) + \
                    args.appnp_alpha * weights['item_embedding']
            all_embeddings[0].append(Zs_prop_user)
            all_embeddings[1].append(Zs_prop_item)
        u_g_embeddings = tf.concat(all_embeddings[0], 1)
        i_g_embeddings = tf.concat(all_embeddings[1], 1)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 2.1: # add two mapping functions
        adj_uv, adj_vu = config['adj_uv'], config['adj_vu']
        all_embeddings = [[weights['user_embedding']], [weights['item_embedding']]]
        coo_uv, coo_vu = adj_uv.tocoo(), adj_vu.tocoo()
        coo_indices_uu = np.mat([coo_uv.row, coo_uv.col]).transpose()
        coo_indices_vv = np.mat([coo_vu.row, coo_vu.col]).transpose()
        A_hat_uv = tf.SparseTensor(coo_indices_uu, np.array(coo_uv.data, dtype=np.float32), coo_uv.shape)
        A_hat_vu = tf.SparseTensor(coo_indices_vv, np.array(coo_vu.data, dtype=np.float32), coo_vu.shape)
        Zs_prop_user, Zs_prop_item = weights['user_embedding'], weights['item_embedding']
        for _ in range(args.appnp_niter):
            A_drop_val_uv = tf.nn.dropout(A_hat_uv.values, keep_prob)
            A_drop_val_vu = tf.nn.dropout(A_hat_vu.values, keep_prob)
            A_drop_uv = tf.SparseTensor(A_hat_uv.indices, A_drop_val_uv, A_hat_uv.dense_shape)
            A_drop_vu = tf.SparseTensor(A_hat_vu.indices, A_drop_val_vu, A_hat_vu.dense_shape)
            fake_user = appnp_generator(Zs_prop_item, weights, direction='vu')
            fake_item = appnp_generator(Zs_prop_user, weights, direction='uv')
            Zs_prop_user = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_uv, fake_user) + \
                    args.appnp_alpha * weights['user_embedding']
            Zs_prop_item = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_vu, fake_item) + \
                    args.appnp_alpha * weights['item_embedding']
            all_embeddings[0].append(Zs_prop_user)
            all_embeddings[1].append(Zs_prop_item)
        u_g_embeddings = tf.concat(all_embeddings[0], 1)
        i_g_embeddings = tf.concat(all_embeddings[1], 1)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 2.2: # add two mapping functions
        adj_uv, adj_vu = config['adj_uv'], config['adj_vu']
        all_embeddings = [[weights['user_embedding']], [weights['item_embedding']]]
        coo_uv, coo_vu = adj_uv.tocoo(), adj_vu.tocoo()
        coo_indices_uu = np.mat([coo_uv.row, coo_uv.col]).transpose()
        coo_indices_vv = np.mat([coo_vu.row, coo_vu.col]).transpose()
        A_hat_uv = tf.SparseTensor(coo_indices_uu, np.array(coo_uv.data, dtype=np.float32), coo_uv.shape)
        A_hat_vu = tf.SparseTensor(coo_indices_vv, np.array(coo_vu.data, dtype=np.float32), coo_vu.shape)
        Zs_prop_user = appnp_generator(weights['user_embedding'], weights, direction='vu')
        Zs_prop_item = appnp_generator(weights['item_embedding'], weights, direction='uv')
        for _ in range(args.appnp_niter):
            A_drop_val_uv = tf.nn.dropout(A_hat_uv.values, keep_prob)
            A_drop_val_vu = tf.nn.dropout(A_hat_vu.values, keep_prob)
            A_drop_uv = tf.SparseTensor(A_hat_uv.indices, A_drop_val_uv, A_hat_uv.dense_shape)
            A_drop_vu = tf.SparseTensor(A_hat_vu.indices, A_drop_val_vu, A_hat_vu.dense_shape)
            Zs_prop_user = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_uv, Zs_prop_item) + \
                    args.appnp_alpha * weights['user_embedding']
            Zs_prop_item = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_vu, Zs_prop_user) + \
                    args.appnp_alpha * weights['item_embedding']
            all_embeddings[0].append(Zs_prop_user)
            all_embeddings[1].append(Zs_prop_item)
        u_g_embeddings = tf.concat(all_embeddings[0], 1)
        i_g_embeddings = tf.concat(all_embeddings[1], 1)
        return u_g_embeddings, i_g_embeddings
    return None, None


def _create_appnpcgan_embed(config, weights, n_users, n_items, keep_prob=0.5):
    if args.sub_version == 3.0:
        adj_uv, adj_vu = config['adj_uv'], config['adj_vu']
        all_embeddings = [[weights['user_embedding']], [weights['item_embedding']]]
        coo_uv, coo_vu = adj_uv.tocoo(), adj_vu.tocoo()
        coo_indices_uu = np.mat([coo_uv.row, coo_uv.col]).transpose()
        coo_indices_vv = np.mat([coo_vu.row, coo_vu.col]).transpose()
        A_hat_uv = tf.SparseTensor(coo_indices_uu, np.array(coo_uv.data, dtype=np.float32), coo_uv.shape)
        A_hat_vu = tf.SparseTensor(coo_indices_vv, np.array(coo_vu.data, dtype=np.float32), coo_vu.shape)
        Zs_prop_user, Zs_prop_item = weights['user_embedding'], weights['item_embedding']
        hiddens = []
        for _ in range(args.appnp_niter):
            A_drop_val_uv = tf.nn.dropout(A_hat_uv.values, keep_prob)
            A_drop_val_vu = tf.nn.dropout(A_hat_vu.values, keep_prob)
            A_drop_uv = tf.SparseTensor(A_hat_uv.indices, A_drop_val_uv, A_hat_uv.dense_shape)
            A_drop_vu = tf.SparseTensor(A_hat_vu.indices, A_drop_val_vu, A_hat_vu.dense_shape)
            fake_user = appnp_generator(Zs_prop_item, weights, direction='vu')
            fake_item = appnp_generator(Zs_prop_user, weights, direction='uv')
            hiddens.append([Zs_prop_user, Zs_prop_item, fake_user, fake_item])
            Zs_prop_user = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_uv, fake_user) + \
                    args.appnp_alpha * weights['user_embedding']
            Zs_prop_item = (1 - args.appnp_alpha) * tf.sparse_tensor_dense_matmul(A_drop_vu, fake_item) + \
                    args.appnp_alpha * weights['item_embedding']
            all_embeddings[0].append(Zs_prop_user)
            all_embeddings[1].append(Zs_prop_item)
        u_g_embeddings = tf.concat(all_embeddings[0], 1)
        i_g_embeddings = tf.concat(all_embeddings[1], 1)
        return u_g_embeddings, i_g_embeddings, hiddens
    return None, None, []


def _create_bpr_loss(users, pos_items, neg_items, decay):
    pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
    neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

    regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
    regularizer = regularizer / args.batch_size

    # In the first version, we implement the bpr loss via the following codes:
    # We report the performance in our paper using this implementation.
    maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores - neg_scores), 1e-10, 1.0))
    mf_loss = tf.negative(tf.reduce_mean(maxi))

    ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
    ## However, it will change the training performance and training performance.
    ## Please retrain the model and do a grid search for the best experimental setting.
    # mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))

    emb_loss = decay * regularizer

    reg_loss = tf.constant(0.0, tf.float32, [1])

    return mf_loss, emb_loss, reg_loss

def _create_cycle_gan_loss(users, pos_items, neg_items, hiddnes, all_weights):
    gan_loss, cycle_loss = None, None
    for rusers, ritems, fusers, fitems in hiddnes:
        real_users = tf.nn.embedding_lookup(rusers, users)
        real_poses = tf.nn.embedding_lookup(ritems, pos_items)
        real_negs = tf.nn.embedding_lookup(ritems, neg_items)
        fake_pos_users = tf.nn.embedding_lookup(fusers, pos_items)
        fake_neg_users = tf.nn.embedding_lookup(fusers, neg_items)
        fake_items = tf.nn.embedding_lookup(fitems, users)

        real_loss = tf.reduce_mean(tf.squared_difference(appnp_discriminator(real_users, all_weights, domain='u'), 0.9)) \
                    + tf.reduce_mean(tf.squared_difference(appnp_discriminator(real_poses, all_weights, domain='v'), 0.9)) \
                    + tf.reduce_mean(tf.squared_difference(appnp_discriminator(real_negs, all_weights, domain='v'), 0.9))
        fake_loss = tf.reduce_mean(tf.square(appnp_discriminator(fake_pos_users, all_weights, domain='u'))) \
                    + tf.reduce_mean(tf.square(appnp_discriminator(fake_neg_users, all_weights, domain='u'))) \
                    + tf.reduce_mean(tf.square(appnp_discriminator(fake_items, all_weights, domain='v')))
        if gan_loss is None:
            gan_loss = (real_loss + fake_loss) / 2
        else:
            gan_loss += (real_loss + fake_loss) / 2

        cycle_users_loss = tf.reduce_mean(tf.abs(appnp_generator(fake_items, all_weights, direction='vu') - real_users))
        cycle_poses_loss = tf.reduce_mean(tf.abs(appnp_generator(fake_pos_users, all_weights, direction='uv') - real_poses))
        cycle_negs_loss = tf.reduce_mean(tf.abs(appnp_generator(fake_neg_users, all_weights, direction='uv') - real_negs))
        if cycle_loss is None:
            cycle_loss = cycle_users_loss + cycle_poses_loss + cycle_negs_loss
        else:
            cycle_loss += cycle_users_loss + cycle_poses_loss + cycle_negs_loss
    return gan_loss, cycle_loss

def build_model(data_config, pretrain_data):
    n_users = data_config['n_users']
    n_items = data_config['n_items']
    n_fold = 10
    norm_adj = data_config['norm_adj']
    # n_nonzero_elems = norm_adj.count_nonzero()
    n_layers = len(eval(args.layer_size))
    regs = eval(args.regs)
    decay = regs[0]

    users = tf.placeholder(tf.int32, shape=(None,))
    pos_items = tf.placeholder(tf.int32, shape=(None,))
    neg_items = tf.placeholder(tf.int32, shape=(None,))

    node_dropout = tf.placeholder(tf.float32, shape=[None])
    mess_dropout = tf.placeholder(tf.float32, shape=[None])

    weights = _init_weights(pretrain_data, n_users, n_items, n_layers)

    if args.alg_type in ['ngcf']:
        ua_embeddings, ia_embeddings = _create_ngcf_embed(norm_adj, weights, mess_dropout,
                                                          node_dropout, n_layers, n_fold, n_users, n_items)
    elif args.alg_type in ['appnp']:
        ua_embeddings, ia_embeddings = _create_appnp_embed(norm_adj, weights, n_users, n_items)
    elif args.alg_type in ['appnpuv']:
        ua_embeddings, ia_embeddings = _create_appnpuv_embed(data_config, weights,
                                                          n_users, n_items)
    elif args.alg_type in ['appnpcgan']:
        ua_embeddings, ia_embeddings, hiddens = _create_appnpcgan_embed(data_config, weights,
                                                          n_users, n_items)
    else:
        raise NotImplementedError("alg_type %s not supported!" % args.alg_type)

    u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, users)
    pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, pos_items)
    neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, neg_items)
    batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False, transpose_b=True)

    mf_loss, emb_loss, reg_loss = _create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, decay)
    if args.alg_type == 'appnpcgan':
        gan_loss, cycle_loss = _create_cycle_gan_loss(users, pos_items, neg_items, hiddens, weights)
        loss = mf_loss + emb_loss + reg_loss + 1e-3 * (gan_loss + 10 * cycle_loss)
    else:
        loss = mf_loss + emb_loss + reg_loss
    return [users, pos_items, neg_items, node_dropout, mess_dropout], batch_ratings, [loss, mf_loss, emb_loss, reg_loss]

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    print(args)

    model_type = 'ngcf' + '_%s_%s_l%d' % (args.adj_type, args.alg_type, len(eval(args.layer_size)))
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    if args.adj_type == 'appnp':
        norm_adj = data_generator.get_appnp_mat()
        config['norm_adj'] = norm_adj
        print('use the appnp normed adjacency matrix')
    elif args.adj_type == 'appnpuv' or args.adj_type == 'appnpcgan':
        # adj_user, adj_item, adj_uu, adj_vv = data_generator.get_split_adj_mat()
        norm_adj = data_generator.get_appnp_mat()
        norm_uv, norm_vu = data_generator.get_appnp_split_mat(norm_adj)
        config['adj_uv'] = norm_uv
        config['adj_vu'] = norm_vu
        config['norm_adj'] = norm_adj
    else:
        plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
        if args.adj_type == 'plain':
            config['norm_adj'] = plain_adj
            print('use the plain adjacency matrix')
        elif args.adj_type == 'norm':
            config['norm_adj'] = norm_adj
            print('use the normalized adjacency matrix')
        elif args.adj_type == 'gcmc':
            config['norm_adj'] = mean_adj
            print('use the gcmc adjacency matrix')
        else:
            config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    placeholders, batch_ratings, losses = build_model(config, pretrain_data)
    users, pos_items, neg_items, node_dropout, mess_dropout = placeholders
    opt_loss, opt_mf_loss, opt_emb_loss, opt_reg_loss = losses
    opt = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(opt_loss)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, placeholders, batch_ratings, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')


    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        train_dataset = data_generator.load_train_temp(epoch)

        for idx in range(n_batch):
            td_users, td_pos_items, td_neg_items = train_dataset[idx]
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([opt, opt_loss, opt_mf_loss, opt_emb_loss, opt_reg_loss],
                               feed_dict={users: td_users, pos_items: td_pos_items,
                                          node_dropout: eval(args.node_dropout),
                                          mess_dropout: eval(args.mess_dropout),
                                          neg_items: td_neg_items})
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0 or epoch < 100:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        if args.dataset == 'amazon-book':
            ret = test(sess, placeholders, batch_ratings, users_to_test, drop_flag=True, batch_test_flag=True)
        else:
            ret = test(sess, placeholders, batch_ratings, users_to_test, drop_flag=True)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        # pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        # hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            # perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
            #            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
            #            (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
            #             ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
            #             ret['ndcg'][0], ret['ndcg'][-1])
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=%.5f, ndcg=%.5f' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['ndcg'][0])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        # if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
        #     save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
        #     print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    # final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
    #              (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
    #               '\t'.join(['%.5f' % r for r in pres[idx]]),
    #               '\t'.join(['%.5f' % r for r in hit[idx]]),
    #               '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  # '\t'.join(['%.5f' % r for r in pres[idx]]),
                  # '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
