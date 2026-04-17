"""
src/preprocessing.py
--------------------
Data loading, cleaning, and splitting for the Fake vs Real News classifier.
All functions are pure and stateless — safe to call from notebooks or scripts.
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def find_csv(name: str, search_dirs: list[str]) -> str:
    """Resolve a CSV filename across multiple candidate directories."""
    for d in search_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"'{name}' not found in: {search_dirs}\n"
        "Place Fake.csv and True.csv in data/raw/ or the project root."
    )


def load_data(base_dir: str) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, assign labels, and return a combined DataFrame.

    Labels: Fake=0, Real=1
    """
    search = [
        os.path.join(base_dir, "data", "raw"),
        base_dir,
    ]
    fake = pd.read_csv(find_csv("Fake.csv", search))
    true = pd.read_csv(find_csv("True.csv", search))

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing steps (documented choices):

    1. Fill nulls in title/text — preserves row count, avoids silent drops.
    2. Combine title + text — title carries strong stylistic signals.
    3. Lowercase — reduces vocabulary without losing signal.
    4. Deduplicate on text — BEFORE split to prevent test contamination.
    5. Drop subject/date — subject is a near-perfect label proxy (leakage risk);
       date adds no generalizable signal for fake-news detection.

    Returns a clean DataFrame with columns: [content, label]
    """
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["text"]  = df["text"].fillna("").astype(str)

    # Combine — title carries stylistic signals (clickbait vs formal)
    df["content"] = (df["title"] + " " + df["text"]).str.lower().str.strip()

    # Dedup on raw text before splitting
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)

    print(f"[preprocessing] Rows before dedup : {before:,}")
    print(f"[preprocessing] Duplicates removed : {removed:,}")
    print(f"[preprocessing] Rows after dedup  : {len(df):,}")

    return df[["content", "label"]].copy()


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = SEED,
) -> tuple:
    """
    Stratified train/test split.

    Returns: X_train, X_test, y_train, y_test
    """
    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Sanity: assert no text overlap
    overlap = set(X_train) & set(X_test)
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} overlapping texts!"

    print(f"[split] Train : {len(X_train):,} | Test : {len(X_test):,}")
    print(f"[split] Train class balance : {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"[split] Test  class balance : {y_test.value_counts(normalize=True).round(3).to_dict()}")

    return X_train, X_test, y_train, y_test
