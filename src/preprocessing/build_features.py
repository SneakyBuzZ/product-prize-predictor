import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

MANIFEST_PATH = 'data/processed/manifest_features.csv'
OUTPUT_PATH = 'data/processed/processed_features.csv'
FEATURE_CORR_PATH = 'reports/feature_correlation.png'
PRICE_DIST_PATH = 'reports/price_distribution_log.png'


def compute_total_weight(row):
    val = row['value']
    count = row['pack_count']
    if row.get('unit_Count', 0):
        return val * count
    elif row.get('unit_Kg', 0):
        return val * 1000 * count
    elif row.get('unit_g', 0):
        return val * 1 * count
    elif row.get('unit_L', 0):
        return val * 1000 * count
    elif row.get('unit_ml', 0):
        return val * 1 * count
    elif row.get('unit_Ounce', 0) or row.get('unit_Oz', 0):
        return val * 28.3495 * count
    elif row.get('unit_Fl_Oz', 0):
        return val * 29.5735 * count
    else:
        return np.nan


def build_features(input_csv, output_csv, price_dist_path, feature_corr_path):
    print("📥 LOADING MANIFEST")
    df = pd.read_csv(input_csv)

    # Drop unnecessary columns
    df = df.drop(columns=['sample_id', 'catalog_content', 'image_link', 'image_status', 'sentiment_label'])

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Fill missing values
    df['pack_count'] = df['pack_count'].fillna(1)
    df['value'] = df['value'].fillna(0)
    df['has_image'] = df['has_image'].fillna(0)
    df['unit_nan'] = df['unit_nan'].fillna(0)

    print("🛠️ COMPUTING DERIVED FEATURES")
    df['total_weight'] = df.apply(compute_total_weight, axis=1).fillna(0)
    df['total_weight'] = np.log1p(df['total_weight'])

    df['text_len_x_digit_count'] = np.log1p(df['text_len'] * df['digit_count'])
    df['text_len_x_bullet_count'] = np.log1p(df['text_len'] * df['bullet_count'])
    df['pack_count_x_total_weight'] = np.log1p(df['pack_count'] * df['total_weight'])
    df['has_bullet'] = (df['bullet_count'] > 0).astype(int)

    # Log-transform price
    df['price'] = np.log1p(df['price'])

    # Standardize numeric features (exclude target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('price')
    print(f"⚡ STANDARDIZING NUMERIC FEATURES: {numeric_cols}")
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save final processed CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ FEATURES SAVED TO {output_csv}")

    # Generate price distribution plot
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.hist(np.expm1(df['price']), bins=50, color='skyblue')
    plt.title('Price Distribution (Original)')
    plt.subplot(1,2,2)
    plt.hist(df['price'], bins=50, color='orange')
    plt.title('Price Distribution (Log Transformed)')
    plt.tight_layout()
    plt.savefig(price_dist_path)
    plt.close()
    print(f"📌 PRICE DISTRIBUTION SAVED TO {price_dist_path}")

    # Generate correlation heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(feature_corr_path)
    plt.close()
    print(f"📌 FEATURE CORRELATION HEATMAP SAVED TO {feature_corr_path}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FEATURE_CORR_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PRICE_DIST_PATH), exist_ok=True)

    build_features(
        MANIFEST_PATH,
        OUTPUT_PATH,
        PRICE_DIST_PATH,
        FEATURE_CORR_PATH
    )
