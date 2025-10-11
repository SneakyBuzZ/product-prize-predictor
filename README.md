# Amazon ML Challenge 2025 — Product Price Prediction

This repo contains the full pipeline for the Amazon ML Challenge 2025 (Smart Product Pricing) — a multimodal regression task combining textual and visual data to predict product prices.

## Structure
- `src/` : main pipeline code (data, features, models)
- `data/` : datasets (train/test + image cache)
- `notebooks/` : quick exploratory analysis
- `checkpoints/` : trained model weights
- `reports/` : plots and analysis

## Getting Started
```bash
pip install -r requirements.txt
python src/data/ingest_and_eda.py --train data/raw/train.csv --test data/raw/test.csv
