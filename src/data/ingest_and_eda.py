import requests
from PIL import Image
from io import BytesIO
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import matplotlib.pyplot as plt
import argparse

def download_image(url, timeout=15, max_retries=3):
    if not isinstance(url, str) or not url.strip():
        return False, "Empty URL"
    
    tries = 0
    last_err = None
    
    while tries < max_retries:
        tries += 1
        try:
            resp = requests.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                try:
                    img = Image.open(BytesIO(resp.content))
                    img.load()
                    return True, f"{img.format},{img.size}"
                except Exception as e:
                    last_err = f"PIL load failed: {e}"
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.3)
    
    return False, last_err

def build_manifest(train_csv, test_csv, out_manifest, sample_download=500):
    print("🔃 LOADING train/test CSVs")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"✅ TRAIN CSV: {len(train_df)} | TEST CSV: {len(test_df)}")

    assert train_df['sample_id'].is_unique, "=== ERROR: train sample_id not unique ==="
    assert test_df['sample_id'].is_unique, "=== ERROR: test sample_id not unique ==="

    print("📊 PRICE STATS")
    print("----------------------------------")
    print(train_df['price'].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))
    print("----------------------------------")

    print("📦 CREATING MANIFEST")
    manifest = train_df[['sample_id','catalog_content','image_link','price']].copy()
    manifest['text_len'] = manifest['catalog_content'].fillna("").str.len()
    manifest['has_image'] = manifest['image_link'].notnull() & manifest['image_link'].str.strip().ne("")
    manifest['image_status'] = "not_checked"

    print("🔍 DOWNLOADING SMALL SAMPLE OF IMAGES TO CHECK")
    sample = manifest[manifest['has_image']].sample(
        n = min(sample_download,len(manifest)),random_state=42
    )

    print(f"✅ CHECKING {len(sample)} SAMPLE IMAGES FOR DOWNLOAD/VALIDITY")
    result = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_image,url) : url for url in sample['image_link']}
        for fut in as_completed(futures):
            url = futures[fut]
            ok , info = fut.result()
            result[url] = (ok, info)

    update_image_status(manifest, result)

    os.makedirs(os.path.dirname(out_manifest),exist_ok=True)
    manifest.to_csv(out_manifest,index=False)
    print(f"✅ MANIFEST SAVED TO {out_manifest}")

    os.makedirs("reports",exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(train_df['price'].clip(lower=0.01),bins=100)
    plt.title("Price Distribution (Train)")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/price_distribution.png", dpi=150)
    print("✅ PRICE DISTRIBUTION REPORT SAVED TO reports/price_distribution.png")

    return manifest

def update_image_status(manifest, result):
    def status(x):
        if x in result:
            return "ok" if result[x][0] else f"fail:{result[x][1]}"
        else:
            return "not_checked"
    
    manifest.loc[manifest['has_image'], 'image_status'] = manifest.loc[
        manifest['has_image'], 'image_link'
    ].apply(status)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",required=True)
    parser.add_argument("--test",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--sample_download",type=int,default=500)
    args = parser.parse_args()

    build_manifest(train_csv=args.train,test_csv=args.test,out_manifest=args.out,sample_download=args.sample_download)