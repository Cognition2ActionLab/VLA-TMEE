from huggingface_hub import snapshot_download
import os

os.environ["HF_ENDPOINT"] = "https://huggingface.co/"   # or "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

# os.environ["https_proxy"] = ""
# os.environ["http_proxy"] = ""


simpler_datasets = {
    "bridge_orig_1.0.0_lerobot":      "IPEC-COMMUNITY/bridge_orig_lerobot",
    # "fractal20220817_data_0.1.0_lerobot":      "IPEC-COMMUNITY/fractal20220817_data_lerobot",
}

base_dir = "your/dataset/path"

os.makedirs(base_dir, exist_ok=True)

for name, repo_id in simpler_datasets.items():
    local_dir = os.path.join(base_dir, name)
    print(f"Downloading {name} to {local_dir} ...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",       
        local_dir=local_dir,
        max_workers=8,                
        resume_download=True           
    )
