## Pretrained Qwen3-VL Model Downloading

Before starting the experiments, we recommend downloading the pretrained Qwen-VL models to a designated directory.
Run the following script:

```bash
python vla_tmee/utils/download_qwen3_vl_models.py
```

Alternatively, you can download the models locally using the following links:

- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)


## VLA-TMEE Model Downloading

我们展示了在LIBERO上near balance设置下的结果及其权重：

| Method        | LIBERO-Spatial | LIBERO-Goal | LIBERO-Object | LIBERO-Long | Avg  | 🤗 HF Checkpoint |
|---------------|----------------|-------------|---------------|-------------|------|-----------------|
| GR00T         | 98.4 | 95.4 | 98.8 | 92.8 | 96.4 | [GR00T checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-GR00T) |
| + T-MEE       | **98.8** | 95.6 | 99.0 | 93.4 | 96.7 | [GR00T-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-GR00T-TMEE) |
| + Cw-TMEE     | 98.4 | 96.6 | 99.4 | 94.6 | 97.3 | [GR00T-Cw-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-GR00T-Cw-TMEE) |
| + Ew-TMEE     | 98.6 | **97.0** | **100.0** | 94.0 | **97.4** | [GR00T-Ew-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-GR00T-Ew-TMEE) |
| OFT           | 98.8 | 93.6 | 98.4 | 91.2 | 95.5 | [OFT checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-OFT) |
| + T-MEE       | **99.0** | 96.8 | **99.2** | 92.8 | 97.0 | [OFT-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-OFT-TMEE) |
| + Cw-TMEE     | **99.0** | **97.6** | 98.8 | **93.0** | **97.1** | [OFT-Cw-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-OFT-Cw-TMEE) |
| + Ew-TMEE     | 98.8 | 95.6 | 99.0 | 91.8 | 96.3 | [OFT-Ew-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-OFT-Ew-TMEE) |
| π<sub>0</sub>            | 99.4 | 96.4 | 98.6 | 92.8 | 96.8 | [π<sub>0</sub> checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-PI) |
| + T-MEE       | **99.8** | **98.2** | **100.0** | **95.6** | **98.4** | [π<sub>0</sub>-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-PI-TMEE) |
| + Cw-TMEE     | 99.4 | 97.6 | 99.8 | 95.2 | 98.0 | [π<sub>0</sub>-Cw-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-PI-Cw-TMEE) |
| + Ew-TMEE     | 99.4 | 97.0 | 99.8 | 95.2 | 97.9 | [π<sub>0</sub>-Ew-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-PI-Ew-TMEE) |
| DS-VLA        | 98.2 | 96.4 | 99.2 | 89.2 | 95.8 | [DS-VLA checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-Dual) |
| + T-MEE       | 98.6 | **97.6** | **99.8** | **96.8** | **98.2** | [DS-VLA-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-Dual-TMEE) |
| + Cw-TMEE     | **99.4** | 96.6 | **99.8** | 94.4 | 97.6 | [DS-VLA-Cw-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-Dual-Cw-TMEE) |
| + Ew-TMEE     | 99.2 | 96.2 | 99.6 | 95.0 | 97.5 | [DS-VLA-Ew-TMEE checkpoint](https://huggingface.co/Cognition2ActionLab/MEE-VLA-LIBERO-Dual-Ew-TMEE) |