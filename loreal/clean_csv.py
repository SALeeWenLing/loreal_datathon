import pandas as pd
import ast
from pathlib import Path
import json


def parse_list_column(series: pd.Series) -> pd.Series:
    """
    Clean messy list-like strings in a Series into proper JSON array strings.
    Returns a Series of JSON strings (like '["a","b"]').
    """
    def parse(val):
        if pd.isna(val):
            return "[]"
        if isinstance(val, list):
            return json.dumps(val)
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return json.dumps(parsed)
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return json.dumps(parsed)
        except Exception:
            pass
        return json.dumps([val])  # fallback

    return series.apply(parse)


def clean_and_convert_to_parquet(input_csv, output_parquet):
    df = pd.read_csv(input_csv)

    if "tags" in df.columns:
        df["tags"] = parse_list_column(df["tags"])
    if "topicCategories" in df.columns:
        df["topicCategories"] = parse_list_column(df["topicCategories"])

    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_parquet, index=False)
    print(f"Saved cleaned Parquet: {output_parquet}")


if __name__ == "__main__":
    input_dir = Path("dataset")
    output_dir = Path("dataset_cleaned")
    output_dir.mkdir(exist_ok=True)

    for csv_file in input_dir.glob("*.csv"):
        parquet_file = output_dir / (csv_file.stem + ".parquet")
        clean_and_convert_to_parquet(csv_file, parquet_file)
