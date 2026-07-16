"""Step B2: normalize Pharmacy 2's already-tabular sheet.

Verified: 1,000 rows, 12 columns. Generic Name filled 10.2%, Barcode 31.3%,
Manufacture 100%, Product Category is the literal string "all" for every
row (dropped, carries no information).
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.upper()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def clean_pharmacy2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # trim whitespace on all string columns, normalize placeholder text to real NaN
    placeholder_re = re.compile(r"^\s*(none|nan|n/?a|null|-)?\s*$", re.IGNORECASE)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            df[col] = df[col].apply(
                lambda v: None if isinstance(v, str) and placeholder_re.match(v) else v
            )

    if "Product Category" in df.columns:
        df = df.drop(columns=["Product Category"])

    name_source = df["Product English Name"]
    fallback = df["Product Arabic Name"] if "Product Arabic Name" in df.columns else None
    if fallback is not None:
        n_english_missing = name_source.isna().sum()
        if n_english_missing:
            print(f"NOTE: {n_english_missing} rows missing Product English Name; "
                  f"falling back to Product Arabic Name for Name_Normalized.")
        name_source = name_source.where(name_source.notna(), fallback)

    df["Name_Normalized"] = name_source.apply(normalize_name)

    return df


def main():
    print(f"Reading {config.INPUT_PATH} ...")
    df = pd.read_excel(config.INPUT_PATH, sheet_name=config.SHEET_PHARMACY2)
    print(f"Loaded {len(df)} rows.")

    cleaned = clean_pharmacy2(df)
    print(f"Cleaned {len(cleaned)} rows, columns: {list(cleaned.columns)}")

    for col in ["Generic Name", "Barcode", "Manufacture"]:
        n = cleaned[col].notna().sum()
        print(f"  {col}: {n}/{len(cleaned)} filled ({n/len(cleaned):.1%})")

    out_path = config.CACHE_DIR / "pharmacy2_cleaned.pkl"
    cleaned.to_pickle(out_path)
    print(f"Saved cleaned table to {out_path}")
    return cleaned


if __name__ == "__main__":
    main()
