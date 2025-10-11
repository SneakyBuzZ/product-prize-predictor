import pandas as pd
import re
from transformers import pipeline
from tqdm import tqdm
import torch

output_path = 'data/processed/manifest_features.csv'
device = 0 if torch.cuda.is_available() else -1
max_len = 512

tqdm.pandas()

def extract_pack_count(text):
    match = re.search(r'(?:Pack of (\d+)|(\d+)\s*per case)', text, re.I)
    if match:
        for grp in match.groups():
            if grp:
                return int(grp)
    return 1

def extract_unit(text):
    units = ['Ounce', 'Oz', 'Fl Oz', 'Count', 'Kg', 'g', 'ml', 'L']
    for u in units:
        if re.search(r'\b' + re.escape(u) + r'\b', text, re.I):
            return u
    return 'Unknown'

def count_bullets(text):
    bullets = re.findall(r'Bullet Point \d+:', text, re.I)
    return len(bullets)

def count_digits(text):
    return len(re.findall(r'\d+', text))

def count_uppercase(text):
    total_chars = len(text)
    if total_chars == 0:
        return 0
    return sum(1 for c in text if c.isupper()) / total_chars

def keyword_flags(text):
    keywords = ['premium','luxury','organic','gluten-free','vegan']
    return {f'kw_{k}': int(bool(re.search(r'\b'+k+r'\b', text, re.I))) for k in keywords}

def truncate_text(text, max_len=max_len):
    return text[:max_len]

if __name__ == "__main__":
    print("📥 LOADING MANIFEST")
    manifest = pd.read_csv('data/processed/manifest.csv')
    manifest['catalog_content'] = manifest['catalog_content'].fillna('')

    print("🛠️ EXTRACTING FEATURES")
    manifest['pack_count'] = manifest['catalog_content'].apply(extract_pack_count)
    manifest['unit'] = manifest['catalog_content'].apply(extract_unit)
    manifest['bullet_count'] = manifest['catalog_content'].apply(count_bullets)
    manifest['digit_count'] = manifest['catalog_content'].apply(count_digits)
    manifest['caps_ratio'] = manifest['catalog_content'].apply(count_uppercase)

    kw_df = manifest['catalog_content'].apply(keyword_flags).apply(pd.Series)
    manifest = pd.concat([manifest, kw_df], axis=1)

    print("COMPUTING TRANSFORMER SENTIMENT")
    # Set device to 0 if you have a GPU, else -1 for CPU
    sentiment_model = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment', device=device)
    batch_size = 32
    sentiments = []

    for i in tqdm(range(0,len(manifest),batch_size),desc="Batches"):
        batch_texts = manifest['catalog_content'].iloc[i:i+batch_size].tolist()
        batch_texts = [truncate_text(text) for text in batch_texts]
        batch_results = sentiment_model(batch_texts)
        sentiments.extend(batch_results)

    manifest['sentiment_label'] = [r['label'] for r in sentiments]
    manifest['sentiment_score'] = [r['score'] for r in sentiments]

    manifest = pd.get_dummies(manifest, columns=['unit'], dummy_na=True)

    print("✅ FEATURES EXTRACTED")
    manifest.to_csv(output_path, index=False)
    print(f"✅ FEATURES SAVED TO {output_path}")