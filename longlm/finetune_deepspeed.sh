export NCCL_DEBUG=INFO
deepspeed  \
    finetune_trainer.py \
    --data_dir=./data \
    --train_name=train \
    --output_dir=./save_model \
    --save_total_limit=10 \
    --per_device_train_batch_size=3 \
    --per_device_eval_batch_size=3 \
    --num_train_epochs=1 \
    --logging_steps=5 \
    --model_name_or_path=./LongLM-large \
    --learning_rate=1e-4 \
    --n_val=100 \
    --evaluation_strategy=steps \
    --eval_steps=100 \
    --do_train --do_eval \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --gradient_accumulation_steps 40 \
    --deepspeed ./ds_zero2_config_st.json
