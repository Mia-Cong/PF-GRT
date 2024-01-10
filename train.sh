# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 5399 \
# train.py --config configs/pose_free_transfomer.yaml --N_rand 2048 --distributed \
# --lrate_gnt 1e-3 --i_weights 25000 --i_img 25000


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 5399 \
train.py --config configs/pose_free_transfomer_same.yaml --N_rand 1024 --distributed \
--lrate_gnt 1e-3 --i_weights 25000 --i_img 25000