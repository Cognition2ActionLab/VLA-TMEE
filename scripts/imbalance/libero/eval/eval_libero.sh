#!/bin/bash

# cd /share/project/baishuanghao/code/VLA-TMEE && ./scripts/few-shot/libero/eval/eval_libero.sh

source /share/project/baishuanghao/miniconda3/etc/profile.d/conda.sh
conda activate vla_tmee

export CUDA_VISIBLE_DEVICES=0
ckpt_root=/share/project/baishuanghao/code/starVLA/pretrained_models_few_shot/libero_10_2B_0.5
ckpt_path=starvla_qwen_pi/final_model/pytorch_model.pt
run_id=$(echo "$ckpt_path" | cut -d'/' -f1)
your_ckpt="$ckpt_root/$ckpt_path"
log_path="$ckpt_root/$run_id"
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

task_suite_names=$(basename "$ckpt_root" | sed 's/_2B.*//')
echo $task_suite_names
if [[ "$task_suite_names" == "libero_all" ]]; then
    task_suite_names=("libero_goal" "libero_spatial" "libero_object" "libero_10")
else
    task_suite_names=("$task_suite_names")
fi

export LIBERO_HOME=/share/project/baishuanghao/code/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

num_trials_per_task=50

for task_suite_name in "${task_suite_names[@]}"; do
    python ./examples/LIBERO/eval_libero_direct.py \
        --args.pretrained-path ${your_ckpt} \
        # --args.host "127.0.0.1" \
        # --args.port 10093 \
        --args.task-suite-name ${task_suite_name} \
        --args.num-trials-per-task ${num_trials_per_task} \
        --args.video-out-path "results/${task_suite_name}/${folder_name}" \
        --args.log_path ${log_path}
done
