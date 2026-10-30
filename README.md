# Laptop Price Predictor

Machine-learning project that predicts a laptop's price from its specifications — brand, processor, RAM, storage, GPU, display and more.

## What it is

The core of the project is a Google Colab training notebook that builds a preprocessing pipeline, trains and compares **8 regression models plus a stacking ensemble**, and exports the best model as a `best_model.pkl` artifact intended to be loaded by a FastAPI backend. The repo also carries the dataset, the processed data, CatBoost training logs, and the frontend build.

Dataset schema (`laptop_price_dataset.csv`): `Brand, Processor_Brand, Processor_Series, RAM, Storage_Type, Storage_Size, GPU, Display_Size, Resolution, Refresh_Rate, Operating_System, Touch_Screen, Weight, Battery, Price`.

## Contents

```
LaptopPricePredictor/
  training/
    train_colab.ipynb     Training notebook (see below)
  dataset/
    laptop_price_dataset.csv   Raw dataset (~340 KB)
  models/
    processed_data.csv         Preprocessed dataset (~1.26 MB)
  backend/                     FastAPI backend modules (main, schemas, service,
                               logging_config) — present as compiled .pyc caches
  frontend/
    src/lib/                   TypeScript helpers (constants.ts, utils.ts)
    dist/                      Built frontend output
  catboost_info/               CatBoost training metadata for one run
catboost_info/                 CatBoost logs (learn_error.tsv, time_left.tsv, ...)
```

The notebook (`train_colab.ipynb`) walks through:

1. Installing dependencies (pandas, scikit-learn, XGBoost, CatBoost, LightGBM, joblib, matplotlib, seaborn)
2. Optionally loading a real Kaggle dataset and normalising it into the project schema
3. A synthetic fallback generator (6,000 rows) if no Kaggle dataset is supplied
4. A preprocessing pipeline (`ColumnTransformer`: one-hot encoding for categorical features, scaling/imputation for numeric ones)
5. Training the models with `RandomizedSearchCV` tuning on the top tree-based models
6. A stacking ensemble combining the top 4 models with a LightGBM meta-learner

## Run it

Training runs in Google Colab:

1. Open `LaptopPricePredictor/training/train_colab.ipynb` in Colab.
2. Run the cells in order — either upload a `kaggle.json` API file to use a real Kaggle dataset, or let the notebook fall back to its synthetic data generator.
3. The notebook finishes by exporting the trained best model artifact (`best_model.pkl`).

## Tech

- **ML:** Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, LightGBM, joblib
- **Notebook:** Google Colab / Jupyter
- **App side:** FastAPI backend (compiled modules in `backend/`), TypeScript frontend helpers with a Vite build in `frontend/dist/`

## License

MIT — see [LICENSE](LICENSE).
