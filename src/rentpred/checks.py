import pandas as pd
TARGET = "rent_rs"

def load_data():
    train = pd.read_csv("data/raw/train.csv")
    holdout = pd.read_csv("data/raw/holdout_features.csv")
    return train, holdout

def check_leaking_column(train, holdout):
    """Columns that are in train but not in holdout, and how close they are to the target."""
    candidates = [c for c in train.columns if c not in holdout.columns and c != TARGET]
    results = {}
    for col in candidates:
        if pd.api.types.is_numeric_dtype(train[col]):
            results[col] = round(train[col].corr(train[TARGET]), 4)
    return results

def check_unit_mixup(train, threshold=3):
    """Find locality/column pairs whose median is very different from the rest."""
    numeric_cols = [c for c in train.select_dtypes("number").columns
                    if c not in ("record_id", TARGET)]
    flagged = []
    for col in numeric_cols:
        for loc in train["locality"].unique():
            inside = train.loc[train["locality"] == loc, col].median()
            outside = train.loc[train["locality"] != loc, col].median()
            if outside == 0 or inside == 0:
                continue
            ratio = outside / inside
            if ratio > threshold or ratio < 1 / threshold:
                flagged.append((col, loc, round(ratio, 2)))
    return flagged

def check_sentinel_value(train):
    """Find numeric columns with impossible negative values, and how often they repeat."""
    flagged = []
    for col in train.select_dtypes("number").columns:
        if col in ("record_id", TARGET):
            continue
        negatives = train.loc[train[col] < 0, col]
        if len(negatives) > 0:
            value = negatives.mode()[0]
            count = int((train[col] == value).sum())
            flagged.append((col, float(value), count))
    return flagged

def check_duplicate_rows(train):
    """Count rows that repeat once record_id is ignored."""
    cols = [c for c in train.columns if c != "record_id"]
    return int(train.duplicated(subset=cols).sum())

def check_stuck_sensor(train, min_unique=10):
    """Longest run of identical values (ordered by date) for columns with many distinct values."""
    data = train.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "record_id"])
    results = {}
    for col in data.select_dtypes("number").columns:
        if col in ("record_id", TARGET) or data[col].nunique() < min_unique:
            continue
        run_id = (data[col] != data[col].shift()).cumsum()
        run_lengths = data.groupby(run_id)[col].agg(["size", "first"])
        longest = run_lengths.sort_values("size", ascending=False).iloc[0]
        results[col] = (int(longest["size"]), float(longest["first"]))
    return results

def check_target_typos(train, threshold=3):
    """Rows where rent_rs is far larger than the signed rent (typo like an extra zero)."""
    ratio = train[TARGET] / train["rent_in_signed_agreement"]
    bad = train[ratio > threshold]
    return {"count": len(bad), "record_ids": bad["record_id"].tolist()[:10]}

def check_spelling_variants(train, max_expected=5):
    """Text columns with more distinct values than expected, listing the values."""
    flagged = {}
    for col in train.columns:
        if col in ("record_id", "date") or not pd.api.types.is_string_dtype(train[col]):
            continue
        values = train[col].value_counts(dropna=False)
        if len(values) > max_expected:
            flagged[col] = {"distinct": len(values), "values": values.index.tolist()}
    return flagged

if __name__ == "__main__":
    train, holdout = load_data()
    print("Leak:", check_leaking_column(train, holdout))
    print("Unit mix-up:", check_unit_mixup(train))
    print("Stuck sensor:", check_stuck_sensor(train))
    print("Sentinel:", check_sentinel_value(train))
    print("Duplicates:", check_duplicate_rows(train))
    print("Target typos:", check_target_typos(train))
    print("Spelling variants:", check_spelling_variants(train))