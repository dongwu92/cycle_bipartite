python NGCF_start.py --dataset gowalla --regs [1e-5] --embed_size 96 --layer_size [96,96,96] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 400 --verbose 1 --node_dropout [0.15] --mess_dropout [0.15,0.15,0.15] --gpu_ids 2 --sub_version 1.22 --n_head 3
python NGCF_start.py --dataset gowalla --regs [1e-5] --embed_size 96 --layer_size [96,96,96] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 400 --verbose 1 --node_dropout [0.2] --mess_dropout [0.2,0.2,0.2] --gpu_ids 2 --sub_version 1.22 --n_head 3
python NGCF_start.py --dataset gowalla --regs [1e-5] --embed_size 96 --layer_size [96,96,96] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 400 --verbose 1 --node_dropout [0.3] --mess_dropout [0.3,0.3,0.3] --gpu_ids 2 --sub_version 1.22 --n_head 3
python NGCF_start.py --dataset gowalla --regs [1e-5] --embed_size 96 --layer_size [96,96,96] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 400 --verbose 1 --node_dropout [0.05] --mess_dropout [0.05,0.05,0.05] --gpu_ids 2 --sub_version 1.22 --n_head 3
python NGCF_start.py --dataset gowalla --regs [1e-5] --embed_size 96 --layer_size [96,96,96] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 400 --verbose 1 --node_dropout [0.2] --mess_dropout [0.2,0.2,0.2] --gpu_ids 2 --sub_version 1.22 --n_head 3 --adj_type gcmc
python NGCF_start.py --dataset gowalla --regs [1e-5] --embed_size 96 --layer_size [96,96,96] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 400 --verbose 1 --node_dropout [0.2] --mess_dropout [0.2,0.2,0.2] --gpu_ids 2 --sub_version 1.22 --n_head 3 --adj_type appnp
python NGCF_start.py --dataset gowalla --regs [1e-5] --embed_size 96 --layer_size [96,96,96] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 16384 --epoch 400 --verbose 1 --node_dropout [0.2] --mess_dropout [0.2,0.2,0.2] --gpu_ids 2 --sub_version 1.22 --n_head 3 --adj_type appnp-ns
