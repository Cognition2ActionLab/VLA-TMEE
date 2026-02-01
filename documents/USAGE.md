# Usage

## 1. Nearly Balance Setting

Following [DOWNLOAD_DATASET](../documents/DOWNLOAD_DATASET.md) and [DOWNLOAD_MODEL](../documents/DOWNLOAD_MODEL.md), we have prepared the required datasets and pretrained models.

### 1.1 Training for LIBERO

Before training, we need to modify in each bash file:

- The **path of base vlm**: `base_vlm_dir`
- The **path of dataset**: `data_root_dir`

Run the following script to start training:

```bash
# train baseline
./scripts/all/libero/train/train_libero.sh [model_name] [libero_suite]

# train baseline with mee-based objects
./scripts/all/libero/train/train_libero_mee.sh [model_name] [libero_suite]
```

Please choose:
- `model_name` from `['gr00t', 'pi', 'oft', 'dual']`
- `libero_suite` from `['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_all', 'libero_all_split']`

Here, libero_all co-trains a single model on all four LIBERO suites, while libero_all_split trains four separate models, one for each suite.
You can also modify `mee_type` and `mee_weight` in the MEE-related bash scripts to configure different MEE settings.

### 1.2 Evaluation for LIBERO

Before training, we need to modify in each bash file:

- The **path of LIBERO**: `LIBERO_HOME`
- The **root path of checkpoints**: `ckpt_root`
- The **related path of checkpoints**: `ckpt_path`

Run the following scripts to start evaluation:

```bash
# launch the evaluation job (with port forwarding)
./scripts/all/libero/eval/eval_libero_job.sh

# run evaluation directly
./scripts/all/libero/eval/eval_libero.sh
```

### 1.3 Training for SimplerEnv

Run the following script to start training:

```bash
# train baseline
./scripts/all/simpler/train/train_simpler.sh [model_name]

# train baseline with mee-based objects
./scripts/all/simpler/train/train_simpler_mee.sh [model_name]
```

### 1.4 Evaluation for SimplerEnv

Before evaluation, we need to modify in each bash file:

- The **path of conda env of vla-tmee**: `vla_tmee_python`
- The **path of conda env of simulation**: `sim_python`
- The **path of SimplerEnv Code**: `SimplerEnv_PATH`
- The **path of checkpoint**: `MODEL_PATH`

Run the following scripts to start evaluation:

```bash
# launch the evaluation job (with port forwarding)
./scripts/all/simpler/eval/eval_simpler.sh
```


## 2. Few-shot, Imbalance, and Noise Setting

### 2.1 Training for LIBERO

The overall setup largely follows the balanced setting, with the following additional configurations:

- The **ratio of few-shot** in [few-shot setting](../scripts/few-shot/libero/train): `few_shot_ratio`
- The **ratio of imbalance** in [imbalance setting](../scripts/imbalance/libero/train): `imbalance_ratio`
- The **image noise type** in [image noise setting](../scripts/image_noise/libero/train): `image_noise_type`
- The **action noise type** in [action noise setting](../scripts/action_noise/libero/train): `action_noise_type`

Run the following script to start training:

```bash
# train baseline
./scripts/imbalance/few-shot/train/train_libero.sh [model_name] [libero_suite]      # few-shot
./scripts/imbalance/libero/train/train_libero.sh [model_name] [libero_suite]        # imbalance
./scripts/image_noise/libero/train/train_libero.sh [model_name] [libero_suite]      # image noise
./scripts/action_noise/libero/train/train_libero.sh [model_name] [libero_suite]     # action noise

# train baseline with mee-based objects
./scripts/imbalance/few-shot/train/train_libero_mee.sh [model_name] [libero_suite]      # few-shot
./scripts/imbalance/libero/train/train_libero_mee.sh [model_name] [libero_suite]        # imbalance
./scripts/image_noise/libero/train/train_libero_mee.sh [model_name] [libero_suite]      # image noise
./scripts/action_noise/libero/train/train_libero_mee.sh [model_name] [libero_suite]     # action noise
```

### 2.1 Evaluation for LIBERO

Run the following scripts to start evaluation:

```bash
# few-shot
./scripts/few-shot/libero/eval/eval_libero.sh              # launch the evaluation job
./scripts/few-shot/libero/eval/eval_libero_job.sh          # run evaluation directly

# imbalance
./scripts/imbalance/libero/eval/eval_libero.sh              # launch the evaluation job
./scripts/imbalance/libero/eval/eval_libero_job.sh          # run evaluation directly

# image noise
./scripts/image_noise/libero/eval/eval_libero.sh

# action
./scripts/action_noise/libero/eval/eval_libero.sh
```
