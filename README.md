# cycle_bipartite
Bipartite Graph Convolutional Networks with Cycle Adversial Generators

# How to run

## Amazon-book

BGCN_base: 

`python BGCN.py --dataset amazon-book --regs [1e-5] --embed_size 160 --layer_size [160,160,160] --lr 0.00045 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 0 --sub_version 1.224 --n_head 4 --adj_type appnp`

BGCN_bi:

`python BGCN.py --dataset amazon-book --regs [1e-5] --embed_size 160 --layer_size [160,160,160] --lr 0.00045 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 0 --sub_version 1.224 --n_head 4 --adj_type appnp-ns`

BGCN_map:

`python BGCN.py --dataset amazon-book --regs [1e-5] --embed_size 160 --layer_size [160,160,160] --lr 0.00045 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 1 --sub_version 1.2271 --n_head 5 --adj_type appnp-ns`

BGCN_gan:

`python BGCN.py --dataset amazon-book --regs [1e-5] --embed_size 160 --layer_size [160,160,160] --lr 0.00045 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 0 --sub_version 4.3 --n_head 4 --adj_type appnp-ns --alg_type cgan --cgan_weight 1e-5`

## Gowalla

BGCN_base:

`python BGCN.py --dataset gowalla --regs [1e-5] --embed_size 250 --layer_size [250,250,250] --lr 0.0002 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 0 --sub_version 1.224 --n_head 2 --adj_type appnp`

BGCN_bi:

`python BGCN.py --dataset gowalla --regs [1e-5] --embed_size 250 --layer_size [250,250,250] --lr 0.0002 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 2 --sub_version 1.224 --n_head 2 --adj_type appnp-ns`

BGCN_map:

`python BGCN.py --dataset gowalla --regs [1e-5] --embed_size 250 --layer_size [250,250,250] --lr 0.0002 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 1 --sub_version 1.2271 --n_head 2 --adj_type appnp-ns`

BGCN_gan:

`python BGCN.py --dataset gowalla --regs [2e-6] --embed_size 250 --layer_size [250,250,250] --lr 0.0002 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 0 --sub_version 4.3 --n_head 2 --adj_type appnp-ns --alg_type cgan`

## Yelp2018

BGCN_base:

`python BGCN.py --dataset yelp --regs [1e-5] --embed_size 280 --layer_size [280,280,280] --lr 0.0002 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 0 --sub_version 1.224 --n_head 2 --adj_type appnp`

BGCN_bi:

`python BGCN.py --dataset yelp --regs [1e-5] --embed_size 280 --layer_size [280,280,280] --lr 0.0002 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 2 --sub_version 1.224 --n_head 2 --adj_type appnp-ns`

BGCN_map:

`python BGCN.py --dataset yelp --regs [1e-5] --embed_size 280 --layer_size [280,280,280] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 1 --sub_version 1.2271 --n_head 2 --adj_type appnp-ns`

BGCN_gan:

`batch_size 16384 --epoch 600 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_ids 3 --sub_version 4.3 --n_head 2 --adj_type appnp-ns --alg_type cgan --cgan_weight 1e-4`