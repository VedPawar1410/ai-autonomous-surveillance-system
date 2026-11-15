"""Download the UCF-Crime I3D features (~10 GB) into data/ucf_crime/.

The zips are NOT extracted: surveillance.data reads the .npy files straight from them
(extracting would cost another 10 GB of disk).

    python -m surveillance.download
"""
import subprocess

from surveillance import config

FILES = ["ground_truth.json", "train.zip", "test.zip"]


def main():
    config.UCF_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = config.UCF_DIR / name
        print(f"downloading {name} ...")
        # curl -C - resumes a partial download instead of starting over (these files are GBs).
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(dest), f"{config.HF_DATASET_URL}/{name}"], check=True)
    print("done:", sorted(p.name for p in config.UCF_DIR.iterdir()))


if __name__ == "__main__":
    main()
