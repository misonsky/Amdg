CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 runner.py \
    --corpus $1 \
    --model $2 \
    --do_train \
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 4 
