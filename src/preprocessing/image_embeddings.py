import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import os
from PIL import Image
from io import BytesIO
import torch
from transformers import ViTImageProcessor, ViTModel
import pandas as pd
import numpy as np
from tqdm import tqdm

CHECKPOINT_PATH = "data/embeddings/_checkpoint_image.txt"
MEMMAP_PATH = "data/embeddings/image_embeddings.dat"
OUTPUT_PATH = "data/embeddings/image_embeddings.npy"
TEMP_FOLDER = "data/tmp_images"

MANIFEST_PATH = "data/processed/manifest.csv"
BATCH_SIZE = 100
ASYNC_WORKERS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/vit-base-patch16-224"
EMBED_DIM = 768


os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MEMMAP_PATH), exist_ok=True)


async def fetch_image(session, url, save_path):
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.read()
                async with aiofiles.open(save_path, 'wb') as f:
                    await f.write(data)
                return True
    except:
        return False
    return False

async def download_batch(urls, folder, workers=ASYNC_WORKERS):
    semaphore = asyncio.Semaphore(workers)
    async def bound_fetch(session, url):
        async with semaphore:
            filename = Path(url).name
            save_path = os.path.join(folder, filename)
            if os.path.exists(save_path):
                return True
            return await fetch_image(session, url, save_path)

    async with aiohttp.ClientSession() as session:
        tasks = [bound_fetch(session, url) for url in urls]
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
            results.append(await f)
    return results


def pil_loader(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def process_batch_embeddings(image_paths, processor, model):
    images = []
    valid_idx = []
    for idx, path in enumerate(image_paths):
        img = pil_loader(path)
        if img is not None:
            images.append(img)
            valid_idx.append(idx)
    if not images:
        return np.zeros((len(image_paths), EMBED_DIM), dtype=np.float32)

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    final_embs = np.zeros((len(image_paths), EMBED_DIM), dtype=np.float32)
    for idx, emb in zip(valid_idx, batch_embs):
        final_embs[idx, :] = emb
    return final_embs


def generate_image_embeddings(manifest_path, temp_folder, memmap_path, output_path, checkpoint_path):
    df = pd.read_csv(manifest_path)
    n_samples = len(df)

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    if not os.path.exists(memmap_path):
        embeddings_memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(n_samples, EMBED_DIM))
    else:
        embeddings_memmap = np.memmap(memmap_path, dtype='float32', mode='r+')

    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"🔄 Resuming from index {start_idx}/{n_samples}")

    for i in range(start_idx, n_samples, BATCH_SIZE):
        batch_urls = df['image_link'].iloc[i:i+BATCH_SIZE].tolist()
        batch_filenames = [os.path.join(temp_folder, Path(url).name) for url in batch_urls]

        asyncio.run(download_batch(batch_urls, temp_folder, ASYNC_WORKERS))
        batch_embs = process_batch_embeddings(batch_filenames, processor, model)
        embeddings_memmap[i:i+len(batch_embs), :] = batch_embs


        with open(checkpoint_path, 'w') as f:
            f.write(str(i + BATCH_SIZE))

        for path in batch_filenames:
            if os.path.exists(path):
                os.remove(path)

    embeddings_memmap.flush()
    np.save(output_path, np.array(embeddings_memmap))
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"✅ Completed embeddings for {n_samples} images. Saved to {output_path}")


if __name__ == "__main__":
    generate_image_embeddings(MANIFEST_PATH, TEMP_FOLDER, MEMMAP_PATH, OUTPUT_PATH, CHECKPOINT_PATH)
