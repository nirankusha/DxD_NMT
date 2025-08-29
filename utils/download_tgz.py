# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import os
import tarfile
import urllib.request

def download_and_extract(url, save_dir):
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    tgz_path = os.path.join(save_dir, os.path.basename(url))

    print(f"📥 Downloading: {url}")
    urllib.request.urlretrieve(url, tgz_path)
    print(f"✅ Saved to: {tgz_path}")

    print("📂 Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=save_dir)
    print(f"✅ Extracted into: {save_dir}")

    # Optional: list files
    print("\n📄 Extracted XML files (first 10):")
    for i, path in enumerate(sorted([p for p in os.listdir(save_dir)])):
        if i >= 10:
            break
        print("  ", path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract a .tgz corpus.")
    parser.add_argument("url", help="URL of the .tgz file to download")
    parser.add_argument("save_path", help="Directory to save and extract contents")
    args = parser.parse_args()

    download_and_extract(args.url, args.save_path)

"""
Created on Sun Aug 10 12:32:24 2025

@author: niran
"""

