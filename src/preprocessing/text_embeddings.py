import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Paths
manifest_path = 'data/processed/manifest.csv'
output_path = 'data/embeddings/text_embeddings.npy'
memmap_path = 'data/embeddings/text_embeddings.dat'
checkpoint_path = 'data/embeddings/_checkpoint.txt'

# Parameters
BATCH_SIZE = 32
MODEL_NAME = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
EMBEDDING_DIM = 768

def generate_text_embeddings(
    input_csv,
    out_path,
    memmap_file,
    checkpoint_file,
    batch_size=BATCH_SIZE,
    model_name=MODEL_NAME,
    device=DEVICE,
    max_length=MAX_LENGTH,
):
    print("📥 LOADING DATA FOR EMBEDDING GENERATION")
    df = pd.read_csv(input_csv)
    n_samples = len(df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print("📦 MODEL LOADED ON", device)

    if not os.path.exists(memmap_file):
        embeddings_memmap = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(n_samples, EMBEDDING_DIM))
    else:
        embeddings_memmap = np.memmap(memmap_file, dtype='float32', mode='r+')
        print("🔄 RESUMING FROM CHECKPOINT")

    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"🔄 RESUMING FROM SAMPLE INDEX {start_idx}/{n_samples}")

    for i in tqdm(range(start_idx, n_samples, batch_size), desc="Generating embeddings"):
        texts = df['catalog_content'].fillna('').iloc[i:i+batch_size].tolist()
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        embeddings_memmap[i:i+embeddings.shape[0], :] = embeddings

        with open(checkpoint_file, 'w') as f:
            f.write(str(i + embeddings.shape[0]))

    embeddings_memmap.flush()

    np.save(out_path, np.array(embeddings_memmap))
    print(f"✅ ALL EMBEDDINGS SAVED TO {out_path} ({n_samples} samples)")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
    generate_text_embeddings(manifest_path, output_path, memmap_path, checkpoint_path)
