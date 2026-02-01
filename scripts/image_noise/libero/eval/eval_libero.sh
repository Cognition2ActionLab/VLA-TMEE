#!/bin/bash

# cd /share/project/baishuanghao/code/VLA-TMEE && ./scripts/image_noise/libero/eval/eval_libero.sh

source /share/project/baishuanghao/miniconda3/etc/profile.d/conda.sh
conda activate vla_tmee

export CUDA_VISIBLE_DEVICES=0
or_types=(degradation_blur_motion_1)        # degradation_noise_sAp
ckpt_root=/share/project/baishuanghao/code/starVLA/pretrained_models_noise_image
suite_paths=(
  "libero_10_2B"
  "libero_10_2B_tmee_1e-2"
)
or_type_dir=degradation_blur_motion     # degradation_noise_sAp
ckpt_path=starvla_qwen_gr00t/final_model/pytorch_model.pt

export LIBERO_HOME=/share/project/baishuanghao/code/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

num_trials_per_task=50
host="127.0.0.1"
base_port=10093

for suite_path in "${suite_paths[@]}"; do
    run_id=$(echo "$ckpt_path" | cut -d'/' -f1)
    your_ckpt="${ckpt_root}/${suite_path}/${or_type_dir}/${ckpt_path}"
    folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
    log_path="${ckpt_root}/${suite_path}/${or_type_dir}/${run_id}"

    for or_type in "${or_types[@]}"; do
        task_suite_name=$(basename "$suite_path" | sed 's/_2B.*//')
        echo "Task suite: ${task_suite_name}"
        python ./examples/LIBERO/eval_libero_noise_direct.py \
            --args.pretrained-path ${your_ckpt} \
            --args.host "$host" \
            --args.port $base_port \
            --args.task-suite-name "$task_suite_name" \
            --args.num-trials-per-task "$num_trials_per_task" \
            --args.video-out-path "results/${task_suite_name}/${folder_name}" \
            --args.log_path ${log_path} \
            --args.OR_type ${or_type}
        
    done    
done