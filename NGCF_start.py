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

def _A_hat(X, node_dropout):
    temp = _convert_sp_mat_to_sp_tensor(X)
    n_nonzero_temp = X.count_nonzero()
    A_hat = _dropout_sparse(temp, 1 - node_dropout[0], n_nonzero_temp)
    return A_hat

def _create_ngcf_embed(norm_adj, weights, mess_dropout, node_dropout, n_layers, n_fold, n_users, n_items):
    if args.sub_version == 1.0:
        if args.node_dropout_flag:
            # node dropout.
            A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
        else:
            A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
        # Generate a set of adjacency sub-matrix.
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
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
    elif args.sub_version == 1.1:
        # Generate a set of adjacency sub-matrix.
        if args.node_dropout_flag:
            # node dropout.
            A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
        else:
            A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.matmul(side_embeddings, weights['W_gc_%d' % k]) + weights['b_gc_%d' % k]
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.matmul(bi_embeddings, weights['W_bi_%d' % k]) + weights['b_bi_%d' % k]
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.11:
        # Generate a set of adjacency sub-matrix.
        if args.node_dropout_flag:
            # node dropout.
            A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
        else:
            A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.matmul(side_embeddings, weights['W_gc_%d' % k]) + weights['b_gc_%d' % k]
            # bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # bi_embeddings = tf.matmul(bi_embeddings, weights['W_bi_%d' % k]) + weights['b_bi_%d' % k]
            ego_embeddings = sum_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.12:
        # Generate a set of adjacency sub-matrix.
        if args.node_dropout_flag:
            # node dropout.
            A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
        else:
            A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.matmul(side_embeddings, weights['W_gc_0']) + weights['b_gc_0']
            # bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # bi_embeddings = tf.matmul(bi_embeddings, weights['W_bi_%d' % k]) + weights['b_bi_%d' % k]
            ego_embeddings = sum_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.13:
        # Generate a set of adjacency sub-matrix.
        if args.node_dropout_flag:
            # node dropout.
            A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
        else:
            A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # sum_embeddings = tf.matmul(side_embeddings, weights['W_gc_%d' % k]) + weights['b_gc_%d' % k]
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.matmul(bi_embeddings, weights['W_bi_%d' % k]) + weights['b_bi_%d' % k]
            ego_embeddings = bi_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.14:
        # Generate a set of adjacency sub-matrix.
        if args.node_dropout_flag:
            # node dropout.
            A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
        else:
            A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # sum_embeddings = tf.matmul(side_embeddings, weights['W_gc_%d' % k]) + weights['b_gc_%d' % k]
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.matmul(bi_embeddings, weights['W_bi_0']) + weights['b_bi_0']
            ego_embeddings = bi_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    elif args.sub_version == 1.2:  # multi-head
        n_head = 2
        if args.node_dropout_flag:
            A_fold_hat = _split_A_hat_node_dropout(norm_adj, node_dropout, n_fold, n_users, n_items)
        else:
            A_fold_hat = _split_A_hat(norm_adj, n_fold, n_users, n_items)
        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            for j in range(n_head):
                ego_embeddings = tf.matmul(side_embeddings, weights['W_gc_%d' % j]) + weights['b_gc_%d' % j]
                ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - mess_dropout[k])
                norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
                all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings
    return None, None

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

def build_model(data_config, pretrain_data):
    n_users = data_config['n_users']
    n_items = data_config['n_items']
    n_fold = 10
    norm_adj = data_config['norm_adj']
    # n_nonzero_elems = norm_adj.count_nonzero()
    weight_size = eval(args.layer_size)
    n_layers = len(weight_size)
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
    else:
        raise NotImplementedError("alg_type %s not supported!" % args.alg_type)

    u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, users)
    pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, pos_items)
    neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, neg_items)
    batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False, transpose_b=True)

    mf_loss, emb_loss, reg_loss = _create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, decay)
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
