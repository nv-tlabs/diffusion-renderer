import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download

parser = argparse.ArgumentParser(description="Download a model checkpoint from Hugging Face Hub")
parser.add_argument(
    "--repo_id",
    type=str,
    required=True,
    help="The repository ID on Hugging Face Hub (e.g., 'username/repo_name')",
)
parser.add_argument(
    "--local_dir",
    type=str,
    default=None,
)

args = parser.parse_args()

def download_weights(repo_id, local_dir):
    """
    Downloads a model checkpoint from Hugging Face Hub and saves it to the specified cache directory.
    
    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        local_dir (str): The directory where the checkpoint will be saved.
    """
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the model snapshot
    snapshot_download(repo_id, local_dir=local_dir)
    print(f"Checkpoint downloaded to {local_dir}")

if __name__ == "__main__":
    if args.local_dir is None:
        args.local_dir = os.path.join('checkpoints', os.path.basename(args.repo_id))
    download_weights(args.repo_id, args.local_dir)