import os
import numpy as np
import pandas as pd


def resolve_file_path(file_path):
    candidates = [
        file_path,
        os.path.join(os.getcwd(), file_path),
        os.path.join("data", os.path.basename(file_path)),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        "Expected dataset location: data/cleaned_real_asset_returns.csv"
    )


def load_yearly_returns(file_path):
    if str(file_path).startswith("http"):
        print(f"Using data URL: {file_path}")
        df = pd.read_csv(file_path)
    else:
        file_path = resolve_file_path(file_path)
        print(f"Using data file: {file_path}")
        df = pd.read_csv(file_path)

    df.columns = df.columns.astype(str).str.strip()

    if "Year" not in df.columns:
        raise ValueError("The CSV must contain a column named 'Year'.")

    for col in df.columns:
        if col != "Year":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna().copy()
    df["Year"] = df["Year"].astype(int)
    df = df.set_index("Year").sort_index()

    returns = df / 100.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    return returns
