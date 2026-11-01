"""Load the dataset and make the train/test split (same split everywhere)."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "bacterial_motility_dataset.csv")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
IMAGES_DIR = os.path.join(ROOT, "images")

TARGET = "chemical"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python scripts/generate_dataset.py")
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split. Returns X_train, X_test, y_train, y_test, df_test.

    df_test keeps the metadata columns (channel_width, hydro) for error analysis.
    Metadata is NEVER used as a model input.
    """
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    return (
        df_train[FEATURE_COLS], df_test[FEATURE_COLS],
        df_train[TARGET], df_test[TARGET], df_test,
    )
