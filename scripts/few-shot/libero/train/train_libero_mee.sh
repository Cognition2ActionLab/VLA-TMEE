#!/bin/bash

# cd /share/project/baishuanghao/code/VLA-TMEE && ./scripts/few-shot/libero/train/train_libero_mee.sh dual libero_goal

# ✅ Activate conda env
source /share/project/baishuanghao/miniconda3/etc/profile.d/conda.sh
conda activate vla_tmee

# ✅ Set env variable
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export action_input_dim=2
export WANDB_MODE=disabled
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/project/baishuanghao/code/decord/build
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

# ✅ Set user variable
original_train_steps=30000
para_type=2B
base_vlm_dir=/share/project/baishuanghao/code/HLM-VLA/models/Qwen3-VL-${para_type}-Instruct
data_root_dir=/share/project/baishuanghao/data/libero_lerobot

few_shot_ratio=0.5
max_train_steps=$(awk "BEGIN { printf \"%d\", $original_train_steps * $few_shot_ratio }")

mee_type=tmee   # cw-tmee, ew-tmee
mee_weight=1e-2

# ====== Parse model_name ======
model_name=${1:-}
run_id="starvla_qwen_${model_name}"
if [[ "$model_name" == "gr00t" ]]; then
    model_type="QwenGR00T"
elif [[ "$model_name" == "pi" ]]; then
    model_type="QwenPI"
elif [[ "$model_name" == "oft" ]]; then
    model_type="QwenOFT"
elif [[ "$model_name" == "dual" ]]; then
    model_type="Qwen-Dual"
else
    echo "Unknown model_name: $model_name"
    exit 1
fi

# ====== Parse libero_suite ======
libero_suite_in=${2:-}
if [[ -z "${libero_suite_in}" ]]; then
  echo "Usage: $0 <libero_suite | libero_all_split>"
  exit 1
fi

if [[ "${libero_suite_in}" == "libero_all_split" ]]; then
  suites=(libero_goal libero_object libero_spatial libero_10)
else
  suites=("${libero_suite_in}")
fi

# ====== Training loop ======
for libero_suite in "${suites[@]}"; do
  echo "=============================================="
  echo "🚀 Start training on suite: ${libero_suite}"
  echo "=============================================="
  
  /share/project/baishuanghao/miniconda3/envs/vla_tmee/bin/accelerate launch \
    --config_file vla_tmee/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 8 \
    vla_tmee/training/train_starvla.py \
    --config_yaml ./vla_tmee/config/training/libero.yaml \
    --framework.name ${model_type} \
    --framework.qwenvl.base_vlm ${base_vlm_dir} \
    --framework.action_model.action_hidden_dim 2 \
    --framework.action_model.action_model_type DiT-B \
    --datasets.vla_data.data_root_dir ${data_root_dir} \
    --datasets.vla_data.data_mix ${libero_suite} \
    --datasets.vla_data.per_device_batch_size 16 \
    --datasets.vla_data.use_all false \
    --datasets.vla_data.use_few true \
    --datasets.vla_data.few_shot_ratio ${few_shot_ratio} \
    --trainer.freeze_modules "" \
    --trainer.max_train_steps ${max_train_steps} \
    --trainer.save_interval 60000 \
    --trainer.logging_frequency 100 \
    --trainer.eval_interval 30000 \
    --trainer.learning_rate.qwen_vl_interface 1e-5 \
    --trainer.learning_rate.action_model 1e-4 \
    --run_root_dir ./outputs_libero_few_shot/${libero_suite}_${para_type}_${mee_type}_${mee_weight} \
    --run_id ${run_id} \
    --wandb_project mee-vla \
    --enable_mee true \
    --mee_type ${mee_type} \
    --mee_weight ${mee_weight}
done