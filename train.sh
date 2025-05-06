export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((15000 + RANDOM % 10000))


torchrun --nproc_per_node=4 --master-port $MASTER_PORT train.py \
    --model_name_or_path qwen2.5-7b\
    --data_path llama3.1-8b \
    --output_dir outputs/distillations \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --eval_steps 40 \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --tf32 True \
    --harmful_size 1.0
