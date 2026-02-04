# <img src="assets/logo.png" width="60" style="vertical-align: middle;"> Reshaping Action Error Distributions for Reliable Vision-Language-Action Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)]() [![Website](https://img.shields.io/badge/Website-GitHub%20Pages-blue.svg)]() [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


[Shuanghao Bai*](https://baishuanghao.github.io/), [Dakai Wang*](), [Cheng Chi*](https://chicheng123.github.io/), [Wanqi Zhou](https://ellezwq.github.io/), [Jing Lyu](), [Xiaoguang Zhao](https://people.ucas.ac.cn/~zhaoxiaoguang?language=en), [Pengwei Wang](https://scholar.google.com/citations?hl=zh-CN&user=2xR6P5AAAAAJ), [Zhongyuan Wang](https://www.wangzhongyuan.com/), [Lei Xing](https://scholar.google.com/citations?user=TlfrTOkAAAAJ&hl=en), [Shanghang Zhang](https://pku-hmi-lab.github.io/HMI-Web/leader.html), [Badong Chen](https://gr.xjtu.edu.cn/web/chenbd/home)


## 🔥 Key Features

1. Trajectory-level MEE reshapes action error distributions in VLA models.

2. Enhances the accuracy and robustness of action generation under standard, few-shot, noisy, and imbalanced settings within a characterized range.

3. A plug-and-play training objective with no inference-time overhead.

4. Provides theoretical analysis that characterizes its optimization properties, robustness, and range of applicability.


## 🛠️ Installation

Refer to [INSTALL.md](documents/INSTALL.md) for installation instructions.


## 💾 Data Preparation

Refer to [DOWNLOAD_DATASET.md](documents/DOWNLOAD_DATASET.md) for instructions on downloading datasets.


## 💻 Model Preparation

Refer to [DOWNLOAD_MODEL.md](documents/DOWNLOAD_MODEL.md) for instructions on downloading pre-trained models.


## 📈 Usage

Refer to [USAGE.md](documents/USAGE.md) for instructions on training and evaluation.


## 📨 Contact

If you have any questions, please create an issue on this repository or contact us at baishuanghao@stu.xjtu.edu.cn.


## 📝 Citation

If you find our work useful, please consider citing:
```
@article{bai2026reshaping,
  title={Reshaping Action Error Distributions for Reliable Vision-Language-Action Models},
  author={Bai, Shuanghao and Wang, DaKai and Chi, Cheng and Zhou, Wanqi and Lyu, Jing and Zhao, Xiaoguang and Wang, Pengwei and Wang, Zhongyuan and Xing, Lei and Zhang, Shanghang and Chen, Badong},
  journal={arXiv preprint arXiv:},
  year={2026}
}

@article{bai2025rethinking,
  title={Rethinking Latent Redundancy in Behavior Cloning: An Information Bottleneck Approach for Robot Manipulation},
  author={Bai, Shuanghao and Zhou, Wanqi and Ding, Pengxiang and Zhao, Wei and Wang, Donglin and Chen, Badong},
  journal={arXiv preprint arXiv:2502.02853},
  year={2025}
}
```


## 🙏 Acknowledgements

This project is primarily built upon [starVLA](https://github.com/starVLA/starVLA), with the image noise components adapted from [CronusVLA](https://github.com/LiHaoHN/SimX-OR).



