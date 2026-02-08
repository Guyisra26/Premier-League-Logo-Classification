from pathlib import Path
import shutil
import kagglehub

DATASET_NAME = "alexteboul/english-premier-league-logo-detection-20k-images"

ROOT_DIR = Path(__file__).resolve().parent
TARGET_DIR = ROOT_DIR / "data"


def download_and_reset_data() -> None:
    print("--- Starting Data Download Process ---")

    if TARGET_DIR.exists():
        print(f"Deleting old directory: {TARGET_DIR}...")
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle (this may take a moment)...")
    cache_path = kagglehub.dataset_download(DATASET_NAME)
    cache_path = Path(cache_path)
    print(f"Download finished to cache: {cache_path}")

    print(f"Copying files to: {TARGET_DIR}...")
    shutil.copytree(cache_path, TARGET_DIR, dirs_exist_ok=True)

    print(f"\nDataset is ready at: {TARGET_DIR}")


if __name__ == "__main__":
    download_and_reset_data()
