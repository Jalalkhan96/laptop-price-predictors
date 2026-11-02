"""Command-line price predictor for the Laptop Price Predictor project.

Usage (from the repository root)::

    python3 LaptopPricePredictor/backend/predict.py \
        --brand Dell --processor-brand Intel --processor-series i7 \
        --ram 16GB --storage-type SSD --storage-size 512 \
        --gpu "RTX 3050" --display-size 15.6 --resolution FHD \
        --refresh-rate 60 --operating-system Windows --touch-screen No \
        --weight 1.8 --battery 56

By default the CLI trains a small ridge-regression model on the bundled
dataset (``LaptopPricePredictor/dataset/laptop_price_dataset.csv``) using
the same preprocessing the training notebook uses: one-hot encoding for
the categorical specs and standard scaling for the numeric ones. If a
trained pipeline artifact exists (``--model models/best_model.pkl``, the
artifact the notebook exports), it is loaded with joblib and used instead,
so the CLI works both with and without a pre-trained model.

Run ``--list-values`` to see the accepted value for every categorical
field, and ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Paths are resolved relative to this file so the CLI works from any cwd.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_DATASET = PROJECT_DIR / "dataset" / "laptop_price_dataset.csv"
DEFAULT_MODEL = PROJECT_DIR / "models" / "best_model.pkl"

CATEGORICAL_FEATURES = [
    "Brand",
    "Processor_Brand",
    "Processor_Series",
    "RAM",
    "Storage_Type",
    "GPU",
    "Resolution",
    "Operating_System",
    "Touch_Screen",
]
NUMERIC_FEATURES = [
    "Storage_Size",
    "Display_Size",
    "Refresh_Rate",
    "Weight",
    "Battery",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Accepted values, taken from the bundled dataset (case-insensitive match).
ALLOWED_VALUES = {
    "Brand": ["Acer", "Apple", "Asus", "Dell", "HP", "Huawei", "Lenovo",
              "MSI", "Microsoft", "Samsung"],
    "Processor_Brand": ["AMD", "Apple Silicon", "Intel"],
    "Processor_Series": ["M1", "M2", "M3", "M4", "Ryzen 3", "Ryzen 5",
                         "Ryzen 7", "Ryzen 9", "i3", "i5", "i7", "i9"],
    "RAM": ["4GB", "8GB", "16GB", "24GB", "32GB", "64GB"],
    "Storage_Type": ["HDD", "NVMe SSD", "SSD"],
    "GPU": ["Integrated", "Intel Iris Xe", "RTX 2050", "RTX 3050",
            "RTX 4050", "RTX 4060", "RTX 4070", "RX 7600"],
    "Resolution": ["HD", "FHD", "2K", "4K"],
    "Operating_System": ["DOS", "Linux", "MacOS", "Windows"],
    "Touch_Screen": ["No", "Yes"],
}

# Defaults mirror the FastAPI backend's LaptopSpecs schema defaults.
DEFAULTS = {
    "brand": "Dell",
    "processor_brand": "Intel",
    "processor_series": "i7",
    "ram": "16GB",
    "storage_type": "SSD",
    "storage_size": 512.0,
    "gpu": "RTX 3050",
    "display_size": 15.6,
    "resolution": "FHD",
    "refresh_rate": 60.0,
    "operating_system": "Windows",
    "touch_screen": "No",
    "weight": None,
    "battery": None,
}


def normalise_categorical(field: str, value: str) -> str:
    """Return the dataset spelling of *value*, or raise ValueError."""
    for allowed in ALLOWED_VALUES[field]:
        if value.lower() == allowed.lower():
            return allowed
    raise ValueError(
        f"invalid {field.replace('_', ' ').lower()}: {value!r} "
        f"(allowed: {', '.join(ALLOWED_VALUES[field])})"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a laptop price from its specifications."
    )
    parser.add_argument("--brand", default=DEFAULTS["brand"])
    parser.add_argument("--processor-brand", default=DEFAULTS["processor_brand"])
    parser.add_argument("--processor-series", default=DEFAULTS["processor_series"])
    parser.add_argument("--ram", default=DEFAULTS["ram"])
    parser.add_argument("--storage-type", default=DEFAULTS["storage_type"])
    parser.add_argument("--storage-size", type=float,
                        default=DEFAULTS["storage_size"],
                        help="storage capacity in GB")
    parser.add_argument("--gpu", default=DEFAULTS["gpu"])
    parser.add_argument("--display-size", type=float,
                        default=DEFAULTS["display_size"],
                        help="screen size in inches")
    parser.add_argument("--resolution", default=DEFAULTS["resolution"])
    parser.add_argument("--refresh-rate", type=float,
                        default=DEFAULTS["refresh_rate"], help="refresh rate in Hz")
    parser.add_argument("--operating-system", default=DEFAULTS["operating_system"])
    parser.add_argument("--touch-screen", default=DEFAULTS["touch_screen"])
    parser.add_argument("--weight", type=float, default=None,
                        help="weight in kg (defaults to the dataset median)")
    parser.add_argument("--battery", type=float, default=None,
                        help="battery capacity in Wh (defaults to the dataset median)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="dataset CSV to train from when no model is given")
    parser.add_argument("--model", type=Path, default=None,
                        help="trained pipeline artifact (joblib .pkl); "
                             "when omitted, models/best_model.pkl is used if present")
    parser.add_argument("--json", action="store_true",
                        help="print the result as JSON instead of text")
    parser.add_argument("--list-values", action="store_true",
                        help="print the accepted values and exit")
    return parser


def specs_to_row(args: argparse.Namespace) -> dict:
    """Validate CLI args and return one dataset-schema row (no Price)."""
    row = {}
    for field, arg_name in (
        ("Brand", "brand"),
        ("Processor_Brand", "processor_brand"),
        ("Processor_Series", "processor_series"),
        ("RAM", "ram"),
        ("Storage_Type", "storage_type"),
        ("GPU", "gpu"),
        ("Resolution", "resolution"),
        ("Operating_System", "operating_system"),
        ("Touch_Screen", "touch_screen"),
    ):
        row[field] = normalise_categorical(field, getattr(args, arg_name))
    row["Storage_Size"] = args.storage_size
    row["Display_Size"] = args.display_size
    row["Refresh_Rate"] = args.refresh_rate
    row["Weight"] = args.weight
    row["Battery"] = args.battery
    return row


def train_from_dataset(dataset_path: Path):
    """Train the notebook-style pipeline (one-hot + scale + ridge)."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(dataset_path)
    df = df.replace("", float("nan")).dropna(subset=FEATURE_COLUMNS + ["Price"])
    if df.empty:
        raise ValueError(f"no usable rows in dataset: {dataset_path}")

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),
    ])
    model.fit(df[FEATURE_COLUMNS], df["Price"])
    return model


def load_model(args: argparse.Namespace):
    """Return a fitted pipeline: artifact first, dataset training second."""
    model_path = args.model or DEFAULT_MODEL
    if model_path is not None and Path(model_path).exists():
        import joblib

        return joblib.load(model_path), f"model artifact ({model_path})"
    return train_from_dataset(args.dataset), f"trained on {args.dataset.name}"


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list_values:
        for field in CATEGORICAL_FEATURES:
            print(f"{field}: {', '.join(ALLOWED_VALUES[field])}")
        return 0

    try:
        row = specs_to_row(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        model, source = load_model(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import pandas as pd

    prediction = float(model.predict(pd.DataFrame([row]))[0])
    result = {"predicted_price": round(prediction, 2), "currency": "PKR",
              "model_used": source, "specs": row}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Predicted price: PKR {prediction:,.0f} (via {source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
