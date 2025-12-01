# services/rankings.py
import sqlite3
import pandas as pd
from utils.names import normalize_name

def load_rankings(db_path="rankings.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT id, name, points FROM rankings", conn)
    df["name_norm"] = df["name"].apply(normalize_name)
    return df
