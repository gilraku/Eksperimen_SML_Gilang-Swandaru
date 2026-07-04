"""
automate_Gilang-Swandaru.py

Modul otomatisasi preprocessing untuk dataset Adult Census Income
(https://www.kaggle.com/datasets/uciml/adult-census-income).

Fungsi utama: preprocess_data()
Mengembalikan data yang sudah bersih dan siap dilatih (X_train, X_test,
y_train, y_test) sekaligus menyimpan hasilnya ke folder output dalam
bentuk CSV (train.csv, test.csv).

Tahapan yang dilakukan (identik dengan proses manual pada notebook
Eksperimen_Gilang-Swandaru.ipynb):
    1. Load dataset
    2. Bersihkan nilai "?" -> NaN lalu hapus baris yang mengandung NaN
    3. Hapus data duplikat
    4. Encoding target (income) menjadi 0/1
    5. Encoding fitur kategorikal menggunakan Label Encoding
    6. Standarisasi fitur numerik menggunakan StandardScaler
    7. Split data menjadi train dan test (80:20, stratified)
    8. Simpan hasil preprocessing ke CSV
"""
#test push
import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

NUMERICAL_COLUMNS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

TARGET_COLUMN = "income"


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan nilai kosong ('?') dan data duplikat."""
    df = df.copy()
    df = df.replace("?", pd.NA)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def encode_features(df: pd.DataFrame):
    """Melakukan label encoding pada fitur kategorikal dan target."""
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    df[TARGET_COLUMN] = target_encoder.fit_transform(df[TARGET_COLUMN].astype(str))
    encoders[TARGET_COLUMN] = target_encoder

    return df, encoders


def scale_features(df: pd.DataFrame):
    """Melakukan standarisasi pada fitur numerik."""
    df = df.copy()
    scaler = StandardScaler()
    df[NUMERICAL_COLUMNS] = scaler.fit_transform(df[NUMERICAL_COLUMNS])
    return df, scaler


def preprocess_data(
    data_path: str,
    save_path: str = "preprocessing/adult_preprocessing",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Fungsi utama otomatisasi preprocessing.

    Parameters
    ----------
    data_path : str
        Path menuju file CSV dataset mentah (raw).
    save_path : str
        Folder tujuan untuk menyimpan train.csv dan test.csv.
    test_size : float
        Proporsi data test.
    random_state : int
        Random seed agar hasil split konsisten.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
        Data yang sudah siap dilatih.
    """
    os.makedirs(save_path, exist_ok=True)

    df = load_data(data_path)
    df = clean_data(df)
    df, _ = encode_features(df)
    df, _ = scale_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(os.path.join(save_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(save_path, "test.csv"), index=False)

    print(f"[INFO] Preprocessing selesai. Train shape: {train_df.shape}, "
          f"Test shape: {test_df.shape}")
    print(f"[INFO] Hasil disimpan di: {save_path}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate preprocessing Adult Census Income")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../adult_raw/adult.csv",
        help="Path ke dataset mentah (raw csv)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="adult_preprocessing",
        help="Folder tujuan penyimpanan hasil preprocessing",
    )
    args = parser.parse_args()

    preprocess_data(args.data_path, args.save_path)
