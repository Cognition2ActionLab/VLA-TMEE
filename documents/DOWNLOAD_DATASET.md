## Dataset Downloading

### 1. LIBERO Datasets for LIBERO Simulation

For large-scale models, we adopt the **LeRobot version of the LIBERO dataset**, which differs from the original LIBERO HDF5 datasets. Compared to the original version, the LeRobot datasets store image observations at a higher resolution of **256×256** (instead of **128×128**) and apply additional filtering, including the removal of **no-op (zero) actions** and **unsuccessful demonstrations**.

Run the following code to download the datasets:

```bash
python vla_tmee/utils/download_libero_datasets.py
```

Alternatively, you can download the datasets locally using the following links:

- [LIBERO-Spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-Goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-Object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

Finally, move `modality.json` in [examples/LIBERO](../examples/LIBERO) to each `$LIBERO_LEROBOT_DATA/$subset/meta/modality.json`.


### 2. Bridge Dataset for SimplerEnv Simulation

Run the following code to download the datasets:

```bash
python vla_tmee/utils/download_bridge_datasets.py
```

Alternatively, you can download the datasets locally using the following links:

- [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot)

Finally, move `modality.json` in [examples/SimplerEnv](../examples/SimplerEnv) to each `bridge_orig_1.0.0_lerobot/meta/modality.json`.
