import os
from huggingface_hub import hf_hub_download
from datasets import load_dataset

# 1. Configuration
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "zhouzhoujy/EgoTouch"
local_dir = "/remote-home/luojr/US.T/EgoTouch"

# 2. Clean Local Quantity Check & Finding the missing 1 file
print("Step 1: Running deep structure check to find the missing file...")

# Let's call HF hub directly to get the ground truth file list without downloading them
from huggingface_hub import list_repo_files
try:
    print("Fetching official file list from mirror...")
    remote_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    remote_files = [f for f in remote_files if not f.startswith(".")] # Filter hidden files
    
    # Check what is missing locally
    missing_files = []
    for f in remote_files:
        local_path = os.path.join(local_dir, f)
        if not os.path.exists(local_path):
            missing_files.append(f)
            
    print(f"Official active files count: {len(remote_files)}")
    print(f"Locally missing files count: {len(missing_files)}")
    
    if missing_files:
        print(f"The missing file is: {missing_files[0]}")
        print("Downloading the missing file right now...")
        hf_hub_download(
            repo_id=repo_id,
            filename=missing_files[0],
            repo_type="dataset",
            local_dir=local_dir,
            force_download=True
        )
    else:
        print("? All active data files are already present on your disk!")
except Exception as e:
    print(f"Remote file verification skipped due to: {e}")

# 3. Smart Dataset Loading (Fixing the "No supported data files" error)
print("\n" + "="*50 + "\n")
print("Step 2: Loading dataset using standard structured mapping...")

try:
    # Method A: Force loading using the repository's online definition but directing data to local path
    # This prevents 429 because it only reads 1 small loading script online, and reads all heavy files locally
    dataset = load_dataset(repo_id, data_dir=local_dir)
    print("?? Success! Dataset loaded completely using Mode A:")
    print(dataset)
except Exception as e_a:
    print(f"Mode A fell back: {e_a}")
    print("Attempting Mode B (Pure local offline image/json parsing)...")
    try:
        # Method B: If Mode A fails, parse it as a local generic dataset path
        dataset = load_dataset("json", data_files=os.path.join(local_dir, "**/*.json"))
        print("?? Success! Dataset loaded completely using Mode B (JSON mapping):")
        print(dataset)
    except Exception as e_b:
        print(f"?? Mode B also failed: {e_b}")
        print("Please check your file layout by running: ls -l /remote-home/luojr/US.T/EgoTouch")