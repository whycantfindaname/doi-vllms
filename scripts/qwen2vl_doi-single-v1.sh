#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=42
#SBATCH --partition=gpu
#SBATCH --exclude=gpu19,gpu14
#SBATCH --job-name=lmms-finetune
#SBATCH --output=/home/u2114040/lmms-finetune/output/%j.no_bbox_out.txt
#SBATCH --error=/home/u2114040/lmms-finetune/output/%j.no_bbox_err.txt
GPUS=${GPUS:-6}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRAD_ACCUM=1

DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=qwen2-vl-7b-instruct                                # model id; pick on by running `python supported_models.py`
MODEL_LOCAL_PATH=../models/qwen2-vl-7b-instruct
TRAIN_DATA_PATH=../datasets/train_qwen2_vl/train_qwen2vl_all.json  # path to the training data json file
EVAL_DATA_PATH=../datasets/q-bench/q_bench_eval.json  # path to the evaluation data json file (optional)
IMAGE_FOLDER=../datasets/images                     # path to the image root folder; if provided, the image paths in the json should be relative
VIDEO_FOLDER=../example_data/videos                      # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=8                                            # how many frames are sampled from each video

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=True                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=128                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=256                                            # the lora alpha (both llm and vision encoder)

RUN_ID=${MODEL_ID}_lora-${USE_LORA}_qlora-${Q_LORA}-doi-single-v1    # a custom run id that determines the checkpoint folder and wandb run name

DS_STAGE=zero3                                          # deepspeed stage; < zero2 | zero3 >
NUM_EPOCHS=1                                            # number of training epochs


LR=0.0003                                                 # learning rate
LR=$(echo "scale=10; $LR / ($BATCH_SIZE / $PER_DEVICE_BATCH_SIZE / $GPUS)" | bc)
echo "Used LR: $LR"
MODEL_MAX_LEN=2560                                        # maximum input length of the model


torchrun $DISTRIBUTED_ARGS train.py \
    --model_id $MODEL_ID \
    --model_local_path $MODEL_LOCAL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir ./checkpoints/$MODEL_ID/$RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "steps" \
    --eval_steps 10000 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 7 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --group_by_length True