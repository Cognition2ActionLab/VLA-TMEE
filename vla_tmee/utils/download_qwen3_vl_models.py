from huggingface_hub import snapshot_download
import os

os.environ["HF_ENDPOINT"] = "https://huggingface.co/"   # or "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

# os.environ["https_proxy"] = ""
# os.environ["http_proxy"] = ""

qwen_models = {
    "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
}

base_dir = "your/model/path" 

os.makedirs(base_dir, exist_ok=True)

for name, repo_id in qwen_models.items():
    local_dir = os.path.join(base_dir, name)
    print(f"Downloading {name} to {local_dir} ...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",      
        local_dir=local_dir,
        max_workers=8,
        resume_download=True   
    )
