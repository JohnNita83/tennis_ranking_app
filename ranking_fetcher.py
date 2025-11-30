from io import StringIO
import requests
import pandas as pd
import sqlite3
from datetime import date
from pathlib import Path

COOKIE_FILE = Path(__file__).with_name("cookie.txt")


def load_cookie() -> str:
    try:
        return COOKIE_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


# ---- CONFIG ----
BASE_URL = "https://ti.tournamentsoftware.com/ranking/category.aspx"
PAGE_SIZE = 100  # we use 100 like in Excel


def get_week_label(d: date | None = None) -> str:
    """
    Build a label like '3-2026' from a date.
    Uses ISO week number (same as ISOWEEKNUM in Excel).
    """
    if d is None:
        d = date.today()
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_week}-{iso_year}"


def fetch_category_current(age_group: str, category_id: str) -> pd.DataFrame:
    """
    Fetch ALL pages for one category for the *current* week
    using rid=169 + category=XXXX.
    Returns a pandas DataFrame with an extra AgeGroup + Week column.
    """

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": load_cookie(),
    }

    page = 1
    all_pages: list[pd.DataFrame] = []

    while True:
        params = {
            "rid": "169",                # current week
            "category": category_id,
            "ps": str(PAGE_SIZE),
            "p": str(page),
        }

        resp = requests.get(BASE_URL, params=params, headers=headers)
        resp.raise_for_status()

        try:
            tables = pd.read_html(StringIO(resp.text))
        except ValueError:
            break  # no tables → stop

        if not tables:
            break

        df = tables[0]

        if df.empty or len(df) == 0:
            break

        all_pages.append(df)

        if len(df) < PAGE_SIZE:
            break

        page += 1

    if not all_pages:
        return pd.DataFrame()

    combined = pd.concat(all_pages, ignore_index=True)

    week_label = get_week_label()
    combined["AgeGroup"] = age_group
    combined["Week"] = week_label

    return combined

def fetch_category_for_week(week_id: str, age_group: str, category_id: str) -> pd.DataFrame:
    """
    Fetch rankings table for a specific week (by ?id=WeekID&category=...)
    Handles pagination so all players are retrieved (not just the first 25).
    Returns a cleaned DataFrame with at least: Rank, Player, Week, AgeGroup, etc.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": load_cookie(),
    }

    page = 1
    all_pages: list[pd.DataFrame] = []

    while True:
        params = {
            "id": week_id,
            "category": category_id,
            "ps": str(PAGE_SIZE),   # request up to PAGE_SIZE rows per page
            "p": str(page),         # page number
        }

        resp = requests.get(BASE_URL, params=params, headers=headers)
        resp.raise_for_status()

        try:
            tables = pd.read_html(StringIO(resp.text))
        except ValueError:
            break  # no tables → stop

        if not tables:
            break

        df = tables[0]

        if df.empty:
            break

        # Drop completely empty columns
        df = df.dropna(axis=1, how="all")

        # Remove junk rows where Rank is not numeric
        if "Rank" in df.columns:
            df = df[pd.to_numeric(df["Rank"], errors="coerce").notna()]

        # Drop unwanted columns if they exist
        for col in ["Rank.1", "Unnamed: 2", "Unnamed: 4"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        all_pages.append(df)

        # If less than a full page, we’re done
        if len(df) < PAGE_SIZE:
            break

        page += 1

    if not all_pages:
        return pd.DataFrame()

    combined = pd.concat(all_pages, ignore_index=True)
    combined["AgeGroup"] = age_group
    combined["Week"] = week_id

    return combined


DB_PATH = "rankings.db"

def add_ranking_week(week_id: str, week_label: str | None = None):
    """
    Fetch and save rankings for all categories for a given TI WeekID.
    week_id: the numeric TI week ID (e.g. "49157")
    week_label: optional label (e.g. "47-2025"). If None, week_id is used.
    """
    categories = [
        ("BS12", "2068"),
        ("BS14", "2072"),
        ("BS16", "2076"),
        ("BS18", "2080"),
        ("GS12", "2069"),
        ("GS14", "2073"),
        ("GS16", "2077"),
        ("GS18", "2081"),
    ]

    label = week_label or week_id

    for age_group, category_id in categories:
        try:
            df = fetch_category_for_week(week_id, age_group, category_id)
            save_week_to_db(df, label)
            print(f"Saved {len(df)} rows for {age_group} (Week {label})")
        except Exception as e:
            print(f"Failed to fetch {age_group} for Week {week_id}: {e}")


def save_week_to_db(df: pd.DataFrame, week_id: str):
    if df is None or df.empty:
        print(f"No rows to save for week {week_id}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Rank INTEGER,
            Player TEXT,
            MemberID TEXT,
            YearOfBirth INTEGER,
            WTN REAL,
            RankingPoints INTEGER,
            TotalPoints INTEGER,
            Tournaments INTEGER,
            AvgPoints REAL,
            Province TEXT,
            AgeGroup TEXT NOT NULL,
            Week TEXT NOT NULL,
            UpdatedAt TEXT
        )
    """)

    age_group = df["AgeGroup"].iloc[0] if "AgeGroup" in df.columns else None

    # Delete only rows for this AgeGroup + Week
    if age_group:
        cur.execute("DELETE FROM rankings WHERE Week = ? AND AgeGroup = ?", (week_id, age_group))
    else:
        cur.execute("DELETE FROM rankings WHERE Week = ?", (week_id,))

    conn.commit()

    df.to_sql("rankings", conn, if_exists="append", index=False)
    conn.close()
    print(f"Saved {len(df)} rows into {DB_PATH} for Week {week_id}, AgeGroup {age_group}.")
   

if __name__ == "__main__":
    # Quick test: BS12 (2068)
    df_bs12 = fetch_category_current("BS12", "2068")
    print(df_bs12.head())
    print("Rows:", len(df_bs12))
