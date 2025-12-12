import sqlite3
from datetime import datetime
import pandas as pd
from ranking_fetcher import fetch_category_current
from config import DATABASE

# Same mapping you used before for rankings categories
CATEGORIES = {
    # Boys singles
    "BS12": "2068",
    "BS14": "2072",
    "BS16": "2076",  # adjust to your real IDs
    "BS18": "2080",  # adjust to your real IDs

    # Girls singles
    "GS12": "2069",  # adjust to your real IDs
    "GS14": "2073",
    "GS16": "2077",
    "GS18": "2081",
}

# DB_PATH = "rankings.db"


def update_current_week():
    frames: list[pd.DataFrame] = []

    # fetch all 8 categories
    for age_group, cat_id in CATEGORIES.items():
        print(f"Fetching {age_group} (category {cat_id})...")
        df = fetch_category_current(age_group, cat_id)

        # Safely handle None or empty
        if df is None or getattr(df, "empty", True):
            print(f"Warning: no data returned for {age_group}")
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError("No data fetched for any category. Check cookie or site (or cookie.txt).")

    combined = pd.concat(frames, ignore_index=True)

    # Remove junk rows: keep only rows where Rank is numeric
    if "Rank" in combined.columns:
        combined = combined[pd.to_numeric(combined["Rank"], errors="coerce").notna()]

    # Remove unwanted columns by name (safe)
    cols_to_drop = ["Rank.1", "Unnamed: 2", "Unnamed: 4"]
    combined.drop(
        columns=[c for c in cols_to_drop if c in combined.columns],
        inplace=True,
        errors="ignore",
    )

    # Add timestamp
    combined["UpdatedAt"] = datetime.utcnow().isoformat(timespec="seconds")

    # Ensure Week column exists
    if "Week" not in combined.columns:
        raise ValueError("Combined DataFrame has no 'Week' column. Check fetch_category_current logic.")

    week_label = combined["Week"].iloc[0]
    print(f"Updating database for Week = {week_label!r}")

    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    # Try to delete existing rows for this week; ignore if table doesn't exist yet
    try:
        cur.execute("DELETE FROM rankings WHERE Week = ?", (week_label,))
        conn.commit()
    except sqlite3.OperationalError as e:
        if "no such table: rankings" in str(e):
            print("Table 'rankings' does not exist yet; will be created by to_sql().")
        else:
            conn.close()
            raise

    # Append new rows (creates table if it doesn't exist)
    combined.to_sql("rankings", conn, if_exists="append", index=False)
    conn.close()



if __name__ == "__main__":
    update_current_week()
