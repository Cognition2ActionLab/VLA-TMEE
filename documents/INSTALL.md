## Installing Conda Environment

The following guidance works well for a machine with H100 GPU, cuda 11.8/12.4, torch 2.6.0, driver 535.161.08.

First, git clone this repo and `cd` into it.

```bash
# clone project
git clone https://github.com/Cognition2ActionLab/VLA-TMEE.git
cd VLA-TMEE
```

#### 1. create python/pytorch env

```bash
# crerate conda environment
conda create -n vla_tmee python=3.10 -y
conda activate vla_tmee
```

#### 2. install dependencies

```bash
# Install env dependencies
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation

# Install VLA-TMEE
pip install -e .
```

If `flash-attn` fails to install correctly, you can run

```bash
python vla_tmee/utils/test_flash_attn.py
```

to check the versions of PyTorch, CUDA, and the libstdc++ ABI.
Then, manually download a compatible wheel from the [flash-attn release](https://github.com/Dao-AILab/flash-attention/releases).
We recommend using version 2.7.3 in most cases. However, for newer GPUs (e.g., NVIDIA RTX 5090), you should install the latest available release (e.g., version 2.8.3) to ensure compatibility.
Example:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 2. install LIBERO simulation dependencies

1. Clone the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) in a separate directory and `cd` into it

```bash
# clone project
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
```

2. Install the dependencies required by LIBERO

```bash
pip install -r requirements.txt
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 transformers==4.57.1
```


#### 3. install SimplerEnv simulation dependencies

You can refer to [starVLA](https://github.com/starVLA/starVLA/tree/starVLA/examples/SimplerEnv) to set up the SimplerEnv environment.


A minimal setup guide is also provided below:

<details>
<summary>Step-by-step SimplerEnv setup</summary>

1. Clone the [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) in a separate directory and `cd` into it

```bash
# clone project
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
cd SimplerEnv
```

2. Install the dependencies required by SimplerEnv

<!-- pip install tyro matplotlib mediapy websockets msgpack rich omegaconf accelerate transformers torchvision diffusers numpy==1.24.4  -->

```bash
pip install tyro matplotlib mediapy websockets msgpack numpy==1.24.4

# install maniSkill2
cd ManiSkill2_real2sim
pip install -e .

# install simpler_env
cd ..
pip install -e .
```
</details>
