import pandas as pd

LEAK_COLUMN = "rent_in_signed_agreement"
SQM_TO_SQFT = 10.7639
FURNISHING_FIXES = {"ful.": "full", "sem.": "semi", "unf.": "unfurnished"}


def clean_data(df, wifi_median=None):
    """Fix the data problems found by checks.py. Returns a cleaned copy."""
    df = df.copy()

    # 1. Leaking column: not available at prediction time, so drop it
    df = df.drop(columns=[LEAK_COLUMN], errors="ignore")

    # 2. Unit mix-up: Sector East recorded area in square metres
    east = df["locality"] == "Sector East"
    df.loc[east, "room_area_sqft"] = df.loc[east, "room_area_sqft"] * SQM_TO_SQFT

    # 3. Sentinel: -1 means "missing", so replace it with the median speed
    df["wifi_mbps"] = df["wifi_mbps"].where(df["wifi_mbps"] >= 0)
    if wifi_median is None:
        wifi_median = df["wifi_mbps"].median()
    df["wifi_mbps"] = df["wifi_mbps"].fillna(wifi_median)

    # 4. Spelling variants: strip spaces, lowercase, expand abbreviations
    df["furnishing"] = (
        df["furnishing"].str.strip().str.lower().replace(FURNISHING_FIXES)
    )
    return df


if __name__ == "__main__":
    train = pd.read_csv("data/raw/train.csv")
    clean = clean_data(train)
    print(clean["furnishing"].value_counts())
    print(clean.groupby("locality")["room_area_sqft"].median())
    print("wifi min:", clean["wifi_mbps"].min(), "| columns:", list(clean.columns))