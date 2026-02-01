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


## 2. 

### 2.1 Training for LIBERO

大体设置与balance基本一致，多加了

- The **ratio of few-shot** in [few-shot setting](../scripts/few-shot/libero/train): `few_shot_ratio`
- The **ratio of imbalance** in [imbalance setting](../scripts/imbalance/libero/train): `few_shot_ratio`

```bash
# train baseline
./scripts/imbalance/few-shot/train/train_libero.sh [model_name] [libero_suite]      # few-shot
./scripts/imbalance/libero/train/train_libero.sh [model_name] [libero_suite]        # imbalance

# train baseline with mee-based objects
./scripts/imbalance/few-shot/train/train_libero_mee.sh [model_name] [libero_suite]  # few-shot
./scripts/imbalance/libero/train/train_libero_mee.sh [model_name] [libero_suite]    # imbalance
```