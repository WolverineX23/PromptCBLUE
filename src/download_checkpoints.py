from huggingface_hub import snapshot_download

# repo_id = "THUDM/visualglm-6b"        # 原代码 - 为啥是 visualglm ? 就离谱!!!
repo_id = "THUDM/chatglm-6b"
downloaded = snapshot_download(
    repo_id,
    # cache_dir="./",                   # 原代码
    cache_dir="./base/",
)
