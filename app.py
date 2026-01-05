import sqlite3
from update_current_week import update_current_week as run_update
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from tournament_fetcher import fetch_tournament_details
from ranking_fetcher import fetch_category_for_week
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import requests
from bs4 import BeautifulSoup
import unicodedata
import re
import numpy as np
from services.entries import load_entries
from services.rankings import load_rankings
from services.match import match_entries_to_rankings
from utils.names import normalize_name
from services.match import debug_matches
from rapidfuzz import process
import plotly.graph_objs as go
import plotly.utils
import json
from collections import defaultdict
from stringing import stringing_bp
from fitness import fitness_bp, ensure_fitness_tables
from config import DATABASE
from io import StringIO


COOKIE_FILE = Path(__file__).with_name("cookie.txt")

def load_cookie() -> str:
    try:
        return COOKIE_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""

# DB_PATH = "rankings.db"

# Age groups we support
AGE_GROUPS = ["BS12", "BS14", "BS16", "BS18", "GS12", "GS14", "GS16", "GS18"]

def get_db_connection():
    from config import DATABASE
    print("Connecting to DB:", DATABASE, flush=True)  # flush ensures it hits logs
    conn = sqlite3.connect(DATABASE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_categories_table():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            type TEXT CHECK(type IN ('domestic','international')) NOT NULL,
            display_name TEXT,
            deleted INTEGER DEFAULT 0
        );
    """)

    # Try to add 'deleted' if missing
    try:
        cur.execute("ALTER TABLE categories ADD COLUMN deleted INTEGER DEFAULT 0;")
    except sqlite3.OperationalError:
        pass

    # Seed with initial categories if table is empty
    cur.execute("SELECT COUNT(*) FROM categories")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO categories (code, type, display_name) VALUES (?, ?, ?)",
            [
                ("T100", "domestic", "T100"),
                ("T200", "domestic", "T200"),
                ("T500", "domestic", "T500"),
                ("T1000", "domestic", "T1000"),
                ("T1250", "domestic", "T1250"),
                ("T1500", "domestic", "T1500"),
                ("T2000", "domestic", "T2000"),
                ("TP500", "international", "TP500"),
                ("TP1000", "international", "TP1000"),
                ("TE3", "international", "TE3"),
                ("TE2", "international", "TE2"),
                ("TE1", "international", "TE1"),
            ]
        )

    conn.commit()
    conn.close()


def ensure_points_table():
    conn = get_db_connection()
    cur = conn.cursor()
    # Create points table with Place and any existing categories as columns
    cur.execute("""
        CREATE TABLE IF NOT EXISTS points (
            Place INTEGER PRIMARY KEY
        );
    """)
    # Seed default places 0..32 if empty
    cur.execute("SELECT COUNT(*) AS c FROM points;")
    if cur.fetchone()["c"] == 0:
        cur.executemany("INSERT INTO points (Place) VALUES (?)", [(p,) for p in range(0, 33)])
    conn.commit()
    conn.close()

def load_categories(include_deleted: bool = False):
    conn = get_db_connection()
    cur = conn.cursor()
    if include_deleted:
        cur.execute("SELECT code, display_name, type FROM categories ORDER BY type, code")
    else:
        cur.execute("SELECT code, display_name, type FROM categories WHERE deleted = 0 ORDER BY type, code")
    rows = cur.fetchall()
    conn.close()
    return rows  # now returns a list of rows with (code, display_name, type)



def load_categories_by_type(include_deleted: bool = False):
    conn = get_db_connection()
    cur = conn.cursor()
    if include_deleted:
        cur.execute("SELECT code, display_name, type FROM categories ORDER BY type, code")
    else:
        cur.execute("SELECT code, display_name, type FROM categories WHERE deleted = 0 ORDER BY type, code")
    rows = cur.fetchall()
    conn.close()

    domestic = [(r[0], r[1]) for r in rows if r[2] == "domestic"]
    international = [(r[0], r[1]) for r in rows if r[2] == "international"]

    return {"domestic": domestic, "international": international}



def load_points_map(point_columns):
    """
    Build a mapping of {place: {category_code: points}} from the points table.
    point_columns is expected to be a list of (code, display_name) tuples.
    """
    ensure_points_table()
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row   # ✅ ensure rows are dict‑like
    cur = conn.cursor()
    cur.execute("SELECT * FROM points;")
    rows = cur.fetchall()
    conn.close()

    result = {}
    for r in rows:
        place = r["Place"]  # adjust if your schema uses lowercase 'place'
        inner = {}
        for col_code, _disp in point_columns:
            # fill in points for each category code, defaulting to None if missing
            inner[col_code] = r[col_code] if col_code in r.keys() else None
        result[place] = inner
    return result

# Flask app creation
app = Flask(__name__)
app.secret_key = "IWA@1StJohns"
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['ENV'] = 'development'
app.config['DEBUG'] = True

# Register Blueprints
app.register_blueprint(stringing_bp)
ensure_fitness_tables()
app.register_blueprint(fitness_bp)


# Ensure tables exist before routes use them
# ensure_categories_table()
# ensure_points_table()


# Step 1: initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # redirect to /login if not authenticated

# Step 2: define users in backend (hardcoded or from DB)
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Example: backend-defined users (no registration, only added manually)
USERS = {
    "admin": User(id=1, username="admin", password="1830503100164"),
}

@login_manager.user_loader
def load_user(user_id):
    for user in USERS.values():
        if str(user.id) == str(user_id):
            return user
    return None

# Step 3: Global protection hook
@app.before_request
def require_login():
    # allow login/logout and static files without authentication
    if request.endpoint in ("login", "logout", "static"):
        return
    if not current_user.is_authenticated:
        return redirect(url_for("login"))

# Step 4: login/logout routes
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = USERS.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for("rankings"))
        else:
            error = "Invalid username or password"
    return render_template("login.html", error=error)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/health")
def health():
    return "ok"

_bootstrapped = False   # global flag

@app.before_request
def bootstrap_db():
    global _bootstrapped
    if not _bootstrapped:
        from config import DATABASE
        import os
        print("Using DB:", DATABASE, flush=True)
        if not os.path.exists(DATABASE):
            print("WARNING: DB file does not exist:", DATABASE, flush=True)
        else:
            try:
                with sqlite3.connect(DATABASE) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM sqlite_schema;")
                    print("Tables in DB:", cur.fetchone()[0], flush=True)
            except Exception as e:
                print("DB check failed:", e, flush=True)

        try:
            run_update()
            print("Database bootstrapped with current week rankings.")
        except Exception as e:
            print(f"Bootstrap failed: {e}")

        _bootstrapped = True

def datetimeformat(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d.%m.%y")
    except:
        return value

app.jinja_env.filters["datetimeformat"] = datetimeformat
     

def reset_categories_table():
    conn = get_db_connection()
    cur = conn.cursor()
    # Drop the old table completely
    cur.execute("DROP TABLE IF EXISTS categories;")
    conn.commit()
    conn.close()
    # Recreate and reseed with clean codes
    ensure_categories_table()


def ensure_points_table():
    conn = get_db_connection()
    cur = conn.cursor()

    # Create table if it doesn't exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS points (
            Place INTEGER PRIMARY KEY
            , T100   INTEGER
            , T200   INTEGER
            , T500   INTEGER
            , TP500  INTEGER
            , T1000  INTEGER
            , TP1000 INTEGER
            , T1250  INTEGER
            , T1500  INTEGER
            , T2000  INTEGER
            , TE3    INTEGER
            , TE2    INTEGER
            , TE1    INTEGER
        );
        """
    )

    # Ensure places 0..32 all exist
    for place in range(0, 33):
        cur.execute("SELECT COUNT(*) AS c FROM points WHERE Place = ?;", (place,))
        if cur.fetchone()["c"] == 0:
            cur.execute("INSERT INTO points (Place) VALUES (?);", (place,))

    conn.commit()
    conn.close()

def reset_rankings_table():
    """Drop the rankings table (it will be recreated on next update/add)."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS rankings;")
    conn.commit()
    conn.close()

def add_ranking_week(week_id: str, week_label: str | None = None):
    """
    Fetch rankings for a specific TI week ID (id=week_id) for all categories in CATEGORIES
    and append them to the rankings table with the given week_label.
    If week_label is None, it will try to leave Week column as is or use week_id as fallback.
    """
    from update_current_week import CATEGORIES  # reuse same mapping

    frames: list[pd.DataFrame] = []

    for age_group, cat_id in CATEGORIES.items():
        print(f"Fetching {age_group} for week ID {week_id} (category {cat_id})...")
        df = fetch_category_for_week(week_id, age_group, cat_id)
        if df.empty:
            print(f"Warning: no data returned for {age_group} (week ID {week_id})")
        else:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No data fetched for any category with WeekID={week_id}. Check cookie or site.")

    combined = pd.concat(frames, ignore_index=True)

    # Remove junk rows: keep only rows where Rank is numeric
    combined = combined[pd.to_numeric(combined["Rank"], errors="coerce").notna()]

    # Ensure a Week column
    if "Week" not in combined.columns:
        # If week_label provided, use that; else use week_id as a simple fallback label
        combined["Week"] = week_label if week_label else str(week_id)
    else:
        # If Week exists but you provide a label, override
        if week_label:
            combined["Week"] = week_label

    # Add / update UpdatedAt
    combined["UpdatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    conn = get_db_connection()
    cur = conn.cursor()

    # Delete existing rows for this Week if present (to avoid duplicates)
    week_value = combined["Week"].iloc[0]
    try:
        cur.execute("DELETE FROM rankings WHERE Week = ?", (week_value,))
        conn.commit()
    except sqlite3.OperationalError:
        # Table might not exist yet; we'll let to_sql create it
        pass

    combined.to_sql("rankings", conn, if_exists="append", index=False)
    conn.close()

    # print(f"Saved {len(combined)} rows for Week={week_value} from WeekID={week_id}.")

def ensure_tournaments_table():

    # Ensure categories table exists first
    ensure_categories_table()

    conn = get_db_connection()
    cur = conn.cursor()

    # Create table if it doesn't exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tournaments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            age_group TEXT NOT NULL,
            tournament_id TEXT,
            is_international INTEGER,
            start_date TEXT,
            end_date TEXT,
            tournament_name TEXT,
            category_code TEXT,
            opens_date TEXT,
            close_date TEXT,
            place INTEGER,
            won INTEGER,
            lost INTEGER,
            created_at TEXT
        );
        """
    )
    # Add event_id column if missing
    try:
        cur.execute("ALTER TABLE tournaments ADD COLUMN event_id INTEGER")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    # Add draw_id column if missing
    try:
        cur.execute("ALTER TABLE tournaments ADD COLUMN draw_id INTEGER")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    # Add player_id column if missing
    try:
        cur.execute("ALTER TABLE tournaments ADD COLUMN player_id INTEGER")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    # Add status column if missing
    try:
        cur.execute("ALTER TABLE tournaments ADD COLUMN status TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass

        conn.commit()
        conn.close()



def load_player_weekly_ranks(player_name: str, age_group: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT week, rank FROM rankings WHERE player = ? AND agegroup = ?",
        (player_name, age_group)
    )
    rows = cur.fetchall()
    conn.close()

    # Normalize and sort by year/week extracted from "WEEK-YEAR"
    def parse_week(w):
        try:
            w_str = str(w)
            wk, yr = w_str.split("-")
            return int(yr), int(wk)
        except Exception:
            return (0, 0)

    rows_sorted = sorted(rows, key=lambda r: parse_week(r["week"]))
    weeks = [r["week"] for r in rows_sorted]
    ranks = [int(r["rank"]) for r in rows_sorted if r["rank"] is not None]

    # Match lengths (skip rows without rank)
    weeks = [r["week"] for r in rows_sorted if r["rank"] is not None]

    return weeks, ranks


def load_player_rankings(player_name: str, age_group: str) -> dict:
    """
    Return {week_label: rank_int} for given player + age_group from rankings DB.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT Week, Rank
        FROM rankings
        WHERE Player = ?
          AND AgeGroup = ?
        """,
        (player_name, age_group),
    )
    rows = cur.fetchall()
    conn.close()

    result = {}
    for r in rows:
        try:
            wk = r["Week"]
            rk = int(r["Rank"])
            result[wk] = rk
        except Exception:
            continue
    return result

def load_rankings():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql("SELECT * FROM rankings", conn)
    conn.close()
    # Normalize column names for easier merging
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def get_week_label_for_date(d: date) -> str:
    # ISO week number
    week_num = d.isocalendar()[1]
    year = d.isocalendar()[0]
    return f"{week_num}-{year}"


def fetch_singles_stats(tournament_id: str, player_id: str):
    """Fetch Singles stats (won/lost) from tournamentsoftware page."""

    url = f"https://ti.tournamentsoftware.com/tournament/{tournament_id}/player/{player_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    stats_table = soup.select_one(".module.module--card .table--new")
    if not stats_table:
        return None, None

    for row in stats_table.select("tbody tr"):
        cells = [td.get_text(strip=True) for td in row.select("td")]
        if cells and cells[0].lower() == "singles":
            wl_text = cells[2]  # e.g. "2-3 (40%)"
            wl_part = wl_text.split()[0] if wl_text else ""
            if "-" in wl_part:
                try:
                    won, lost = wl_part.split("-")
                    return int(won), int(lost)
                except ValueError:
                    return None, None
    return None, None


def fetch_event_wl_stats(tournament_id: str, player_id: str) -> dict:
    """
    Return win/loss stats for a player in a tournament.
    Uses wl_stats cache table; scrapes TI only if missing or stale (<1 day old).
    """
   
    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure cache table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS wl_stats (
            tournament_id TEXT,
            player_id TEXT,
            singles_wins INTEGER,
            singles_losses INTEGER,
            doubles_wins INTEGER,
            doubles_losses INTEGER,
            updated_at TEXT,
            PRIMARY KEY (tournament_id, player_id)
        )
    """)
    conn.commit()


    # Check cache
    cur.execute(
        "SELECT singles_wins, singles_losses, doubles_wins, doubles_losses, updated_at "
        "FROM wl_stats WHERE tournament_id=? AND player_id=?",
        (tournament_id, player_id)
    )
    row = cur.fetchone()

    if row:
        singles_wins, singles_losses, doubles_wins, doubles_losses, updated_at = row
        try:
            if datetime.fromisoformat(updated_at) > datetime.now(timezone.utc) - timedelta(days=1):
                conn.close()
                return {
                    "singles": (singles_wins, singles_losses),
                    "doubles": (doubles_wins, doubles_losses),
                }
        except Exception:
            pass  # fall through to re-fetch if parsing fails

    # Scrape live if cache missing/stale
    url = f"https://ti.tournamentsoftware.com/tournament/{tournament_id}/player/{player_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
    result = {"singles": (0, 0), "doubles": (0, 0)}

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"[WARN] Timeout fetching stats for {tournament_id}/{player_id}", flush=True)
        conn.close()
        return result
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Request error fetching {url}: {e}", flush=True)
        conn.close()
        return result

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        stats_table = soup.select_one(".module.module--card .table--new")
        if stats_table:
            def parse_wl(text: str):
                try:
                    wl_part = text.split()[0]
                    if "-" in wl_part:
                        w, l = wl_part.split("-")
                        return int(w), int(l)
                except Exception as e:
                    print(f"[DEBUG] parse_wl error: {e} on text={text}", flush=True)
                return 0, 0

            for row in stats_table.select("tbody tr"):
                tds = [td.get_text(strip=True) for td in row.select("td")]
                if not tds or len(tds) < 3:
                    continue
                label = tds[0].lower()
                wl_text = tds[2]
                if label == "singles":
                    result["singles"] = parse_wl(wl_text)
                elif label == "doubles":
                    result["doubles"] = parse_wl(wl_text)

    except Exception as e:
        print(f"[ERROR] Parsing error for {tournament_id}/{player_id}: {e}", flush=True)

    # Update cache
    cur.execute(
        "INSERT OR REPLACE INTO wl_stats "
        "(tournament_id, player_id, singles_wins, singles_losses, doubles_wins, doubles_losses, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            tournament_id,
            player_id,
            result["singles"][0],
            result["singles"][1],
            result["doubles"][0],
            result["doubles"][1],
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    )
    conn.commit()
    conn.close()

    return result

def get_week_range(start_date_str, end_date_str):
    try:
        start = datetime.fromisoformat(start_date_str).date() if start_date_str else None
        end = datetime.fromisoformat(end_date_str).date() if end_date_str else None
    except Exception:
        return None

    if not start:
        return None

    start_week = start.isocalendar()[1]
    if end:
        end_week = end.isocalendar()[1]
        if end_week != start_week:
            return f"{start_week}/{end_week}"
    return str(start_week)


def build_tournament_view(player_name: str, age_group: str) -> dict:
    ensure_tournaments_table()
    rankings_map = load_player_rankings(player_name, age_group)
    categories = load_categories_by_type()
    raw_categories = load_categories(include_deleted=False)
    
    # ✅ Sorting helpers for tournament categories
    def _numeric_part(code: str) -> int:
        digits = ''.join(ch for ch in code if ch.isdigit())
        return int(digits) if digits else 0

    def sort_by_numeric(items):
        return sorted(items, key=lambda x: _numeric_part(x[0]))

    # keep tuples (code, display_name)
    domestic = [(r["code"], r["display_name"]) for r in raw_categories if r["type"] == "domestic"]
    international = [(r["code"], r["display_name"]) for r in raw_categories if r["type"] == "international"]

    domestic_sorted = sort_by_numeric(domestic)
    international_sorted = sort_by_numeric(international)

    point_columns = domestic_sorted + international_sorted

    points_map = load_points_map(point_columns)

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM tournaments
        WHERE player_name = ?
          AND age_group = ?
        ORDER BY start_date, id
        """,
        (player_name, age_group),
    )
    db_rows = cur.fetchall()
    
    cur.execute(
        """
        SELECT Week, wtn
        FROM rankings
        WHERE Player = ?
        ORDER BY Week
        """,
        (player_name,)
    )
    ranking_rows = cur.fetchall()
    conn.close()

    
    
    def parse_date_safe(s):
        try:
            return datetime.fromisoformat(s).date() if s else None
        except Exception:
            return None

    rows = []
    prev_rank = None

    total_won = 0
    total_lost = 0
    cat_counts = {code: 0 for code, _disp in point_columns}
    cat_points = {code: 0 for code, _disp in point_columns}

    max_ranking_points = None
    top6_tournaments = []

    # Build enriched rows and compute per-row ranking_points with 1-year cutoff
    for idx, r in enumerate(db_rows, start=1):
        row_id = r["id"]
        start_date = r["start_date"]
        end_date = r["end_date"]
        wk_label = None
        rank_value = None
        cat_code = r["category_code"] if "category_code" in r.keys() else r["cat_code"]
        place = r["place"]

        won = r["won"] if r["won"] is not None else 0
        lost = r["lost"] if r["lost"] is not None else 0

        singles_won = singles_lost = doubles_won = doubles_lost = 0

        if r["tournament_id"] and not r["is_international"] and r["player_id"]:
            # Domestic: use cached wl_stats
            wl_stats = fetch_event_wl_stats(r["tournament_id"], r["player_id"])
            singles_won, singles_lost = wl_stats.get("singles", (0, 0))
            doubles_won, doubles_lost = wl_stats.get("doubles", (0, 0))
            won, lost = singles_won, singles_lost
        else:
            # International/manual: use DB values
            singles_won = r["won"] if r["won"] is not None else 0
            singles_lost = r["lost"] if r["lost"] is not None else 0
            won, lost = singles_won, singles_lost


        # Totals: decide if you want singles only or both
        total_won += won
        total_lost += lost


        # Count only tournaments with status == "Played"
        status = r["status"] or "Interest"
        if status == "Played" and cat_code in cat_counts:
            cat_counts[cat_code] += 1

        # points by place/category
        pts = None
        if place is not None and cat_code:
            pm = points_map.get(place)
            if pm:
                pts = pm.get(cat_code)

        if pts is not None and cat_code in cat_points:
            cat_points[cat_code] += pts

        # next week's label and rank at that point
        if end_date:
            try:
                d = datetime.fromisoformat(end_date).date()
                next_week_date = d + timedelta(days=7)
                wk_label = get_week_label_for_date(next_week_date)
                rank_value = rankings_map.get(wk_label)
            except Exception:
                rank_value = None

        # diff/arrow relative to previous rank
        diff = None
        arrow = None
        if rank_value is not None and prev_rank is not None:
            diff = prev_rank - rank_value
            if diff > 0:
                arrow = "up"
            elif diff < 0:
                arrow = "down"
            else:
                arrow = "same"
        prev_rank = rank_value if rank_value is not None else prev_rank

        matches = won + lost
        week_label = get_week_range(start_date, end_date)

        # helper returns weekday + date part separately
        def format_date_parts(date_str):
            try:
                d = datetime.fromisoformat(date_str).date()
                weekday = d.strftime("%a")       # e.g. Sat
                date_part = d.strftime("%d.%m.%y")  # e.g. 03.01.26
                return weekday, date_part
            except Exception:
                return None, None

        # inside your loop
        start_weekday, start_date_part = format_date_parts(start_date)
        end_weekday, end_date_part = format_date_parts(end_date)

        row = {
            "id": row_id,
            "no": idx,
            "week": get_week_range(start_date, end_date),
            "start_date": start_date,   # raw ISO string for <input type="date">
            "end_date": end_date,
            "start_weekday": start_weekday,
            "start_date_part": start_date_part,
            "end_weekday": end_weekday,
            "end_date_part": end_date_part,
            "wk_label": wk_label,
            "tournament_name": r["tournament_name"],
            "cat_code": cat_code,
            "opens_date": r["opens_date"],
            "close_date": r["close_date"],
            "status": r["status"] or "Interest",
            "place": place,
            "points": pts,
            "ranking_points": None,  # set below
            "rank": rank_value,
            "diff": diff,
            "arrow": arrow,
            "won": won,   # singles only
            "lost": lost, # singles only
            "matches": won + lost,
            "singles_won": singles_won,
            "singles_lost": singles_lost,
            "singles_total": singles_won + singles_lost,
            "doubles_won": doubles_won,
            "doubles_lost": doubles_lost,
            "doubles_total": doubles_won + doubles_lost,
            "tournament_id": r["tournament_id"],
            "is_international": str(r["is_international"]) == "1",
            "event_id": r["event_id"],
            "draw_id": r["draw_id"],
            "player_id": r["player_id"],
            "is_top6": False,
        }
        rows.append(row)

        # per-row ranking_points: top-6 within 365 days from this row's end_date
        current_end = parse_date_safe(end_date)
        current_status = row.get("status") or "Interest"

        # Only rows with status == "Played" should have ranking_points
        if current_status != "Played":
            row["ranking_points"] = None  # or 0 if you prefer
        else:
            if current_end:
                valid_points = []
                for prev in rows:  # includes current + all previous
                    prev_end = parse_date_safe(prev["end_date"])
                    if prev_end and 0 <= (current_end - prev_end).days <= 365:
                        status_prev = prev.get("status") or "Interest"
                        p = prev["points"]
                        if status_prev == "Played" and p is not None and p > 0:
                            valid_points.append(p)
                top6 = sorted(valid_points, reverse=True)[:6]
                row["ranking_points"] = sum(top6) if top6 else 0
            else:
                row["ranking_points"] = None

    # max ranking points achieved across all rows
    if rows:
        max_ranking_points = max(
            (r["ranking_points"] for r in rows if r["ranking_points"] is not None),
            default=None
        )
        for r in rows:
            r["is_best_ranking_points"] = (r["ranking_points"] == max_ranking_points)

    # Global highlight: latest row's 12-month window top-6
    if rows:
        last_row = rows[-1]
        last_end = parse_date_safe(last_row["end_date"])
        if last_end:
            cutoff = last_end - timedelta(days=365)
            candidates = []
            for i, rr in enumerate(rows):
                end = parse_date_safe(rr["end_date"])
                if end and cutoff <= end <= last_end:
                    status_rr = rr.get("status") or "Interest"
                    p = rr["points"]

                    if status_rr == "Played" and p is not None and p > 0:
                        candidates.append((i, p))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top6_indices = {i for i, p in candidates[:6]}
            for i in top6_indices:
                rows[i]["is_top6"] = True
            top6_tournaments = [rows[i] for i in sorted(top6_indices)]

    # Category summary (computed once after rows are built)
    category_summary = defaultdict(lambda: {
        "tournaments": 0,
        "total_points": 0,
        "best_place": None,
        "won": 0,
        "lost": 0,
        "total": 0,
        "max_points": 0,
    })

    for rr in rows:
        cat = rr["cat_code"]
        if not cat:
            continue

        status = rr.get("status") or "Interest"

        # Only count Played tournaments
        if status != "Played":
            continue

        summary = category_summary[cat]
        summary["tournaments"] += 1
        summary["total_points"] += rr["points"] if rr["points"] else 0

        place_val = rr["place"]
        if place_val is not None and place_val > 0:
            if summary["best_place"] is None or place_val < summary["best_place"]:
                summary["best_place"] = place_val

        summary["won"] += rr["won"]
        summary["lost"] += rr["lost"]
        summary["total"] += rr["matches"]

        if rr["points"] and rr["points"] > summary["max_points"]:
            summary["max_points"] = rr["points"]

    # Filter categories with at least one tournament
    filtered_summary = {
        cat: data for cat, data in category_summary.items()
        if data["tournaments"] >= 1
    }

    # Grand total row (best_place as min non-zero across categories)
    best_places = [d["best_place"] for d in filtered_summary.values() if d["best_place"] not in (None, 0)]
    grand_total = {
        "tournaments": sum(d["tournaments"] for d in filtered_summary.values()),
        "total_points": sum(d["total_points"] for d in filtered_summary.values()),
        "best_place": min(best_places) if best_places else None,
        "won": sum(d["won"] for d in filtered_summary.values()),
        "lost": sum(d["lost"] for d in filtered_summary.values()),
        "total": sum(d["total"] for d in filtered_summary.values()),
        "max_points": max((d["max_points"] for d in filtered_summary.values()), default=0),
    }

    # Ranking points per category from rows marked is_top6 (latest 12-month window)
    cat_ranking_points = {code: 0 for code, _disp in point_columns}
    for rr in rows:
        if rr.get("is_top6"):
            cat = rr.get("cat_code")
            pts = rr.get("points") or 0
            if cat and pts:
                cat_ranking_points[cat] += pts

    total_matches = total_won + total_lost
    won_pct = (total_won / total_matches * 100) if total_matches > 0 else 0
    lost_pct = (total_lost / total_matches * 100) if total_matches > 0 else 0

    def parse_week_label(label: str):
        try:
            wk, yr = label.split("-")
            return int(yr), int(wk)
        except Exception:
            return (9999, 9999)

    latest_week = None
    latest_rank = None
    if rankings_map:
        latest_week = max(rankings_map.keys(), key=parse_week_label)
        latest_rank = rankings_map[latest_week]

    best_rank = None
    best_rank_week = None
    if rankings_map:
        best_rank = min(rankings_map.values())
        for wk, val in rankings_map.items():
            if val == best_rank:
                best_rank_week = wk
                break

    wtn_map = {r["Week"]: r["WTN"] for r in ranking_rows if r["WTN"] is not None}

    latest_wtn = None
    best_wtn = None
    best_wtn_week = None
    if wtn_map:
        latest_wtn_week = max(wtn_map.keys(), key=parse_week_label)
        latest_wtn = wtn_map[latest_wtn_week]

        best_wtn = min(wtn_map.values())
        for wk, val in wtn_map.items():
            if val == best_wtn:
                best_wtn_week = wk
                break


    # --- Build global stat_summary across all tournaments ---
    def pct(part, total):
        return round(part / total * 100, 1) if total > 0 else 0.0

    # For each row, prefer scraped singles_won/lost if present,
    # otherwise fall back to manually entered won/lost.
    singles_won_total = sum(
        r.get("singles_won", r.get("won", 0)) for r in rows
    )
    singles_lost_total = sum(
        r.get("singles_lost", r.get("lost", 0)) for r in rows
    )

    # Doubles only comes from scraping, so keep as-is
    doubles_won_total = sum(r.get("doubles_won", 0) for r in rows)
    doubles_lost_total = sum(r.get("doubles_lost", 0) for r in rows)

    total_won_all = singles_won_total + doubles_won_total
    total_lost_all = singles_lost_total + doubles_lost_total

    stat_summary = {
        "singles": {
            "won": singles_won_total,
            "lost": singles_lost_total,
            "won_pct": pct(singles_won_total, singles_won_total + singles_lost_total),
            "lost_pct": pct(singles_lost_total, singles_won_total + singles_lost_total),
            "total": singles_won_total + singles_lost_total,
        },
        "doubles": {
            "won": doubles_won_total,
            "lost": doubles_lost_total,
            "won_pct": pct(doubles_won_total, doubles_won_total + doubles_lost_total),
            "lost_pct": pct(doubles_lost_total, doubles_won_total + doubles_lost_total),
            "total": doubles_won_total + doubles_lost_total,
        },
        "total": {
            "won": total_won_all,
            "lost": total_lost_all,
            "won_pct": pct(total_won_all, total_won_all + total_lost_all),
            "lost_pct": pct(total_lost_all, total_won_all + total_lost_all),
            "total": total_won_all + total_lost_all,
        },
    }

    
        
    return {
        "rows": rows,
        "totals": {
            "total_won": total_won,
            "total_lost": total_lost,
            "total_matches": total_matches,
            "won_pct": won_pct,
            "lost_pct": lost_pct,
        },
        "cat_counts": cat_counts,
        "cat_points": cat_points,
        "point_columns": point_columns,
        "max_ranking_points": max_ranking_points,
        "cat_ranking_points": cat_ranking_points,
        "top6_tournaments": top6_tournaments,
        "latest_rank": latest_rank,
        "latest_week": latest_week,
        "best_rank": best_rank,
        "best_rank_week": best_rank_week,
        "latest_wtn": latest_wtn,
        "best_wtn": best_wtn,
        "best_wtn_week": best_wtn_week,
        "category_summary": filtered_summary,
        "grand_total": grand_total,
        "stat_summary": stat_summary,
        "domestic_columns": domestic_sorted,
        "international_columns": international_sorted,
    }



def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_week_for_age_group(age_group):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT DISTINCT Week FROM rankings WHERE AgeGroup = ? ORDER BY Week",
            (age_group,),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        # Table 'rankings' does not exist yet
        conn.close()
        return None

    conn.close()
    if not rows:
        return None
    return rows[-1]["Week"]


def fetch_entries(tournament_id: str) -> list[dict]:
    """
    Fetch entries for a tournament.
    - If today < start_date: scrape live and return fresh entries.
    - If today >= start_date:
        * If cache exists and updated_at >= start_date: return cache.
        * Else scrape fresh and cache.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure entries table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            tournament_id TEXT,
            data TEXT,          -- JSON string of entries
            updated_at TEXT,
            PRIMARY KEY (tournament_id)
        )
    """)
    conn.commit()

    # Try to add updated_at column if missing
    try:
        cur.execute("ALTER TABLE entries ADD COLUMN updated_at TEXT")
        conn.commit()
        print("[INFO] Added updated_at column to entries table", flush=True)
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Get tournament start date from tournaments table
    cur.execute("SELECT start_date FROM tournaments WHERE tournament_id=?", (tournament_id,))
    row = cur.fetchone()
    start_date = None
    if row and row[0]:
        try:
            start_date = datetime.fromisoformat(row[0]).date()
        except Exception:
            start_date = None

    today = datetime.now(timezone.utc).date()

    # --- If today >= start_date, check cache ---
    if start_date and today >= start_date:
        cur.execute("SELECT data, updated_at FROM entries WHERE tournament_id=?", (tournament_id,))
        cached = cur.fetchone()
        if cached and cached["updated_at"]:
            try:
                cache_date = datetime.fromisoformat(cached["updated_at"]).date()
                if cache_date >= start_date:
                    print(f"[INFO] Using cached entries for {tournament_id} (updated_at={cached['updated_at']})", flush=True)
                    conn.close()
                    return json.loads(cached["data"])
            except Exception:
                pass
        print(f"[INFO] Cache missing/stale, scraping entries for {tournament_id}", flush=True)

    # --- Otherwise (today < start_date OR cache stale) scrape fresh ---
    event_id = request.args.get("event_id", type=int)
    url = f"https://ti.tournamentsoftware.com/sport/event.aspx?id={tournament_id}&event={event_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch entries for {tournament_id}: {e}", flush=True)
        conn.close()
        return []

    entries = extract_entries(resp.text)

    # --- Cache scraped entries ---
    cur.execute(
        "INSERT OR REPLACE INTO entries (tournament_id, data, updated_at) VALUES (?, ?, ?)",
        (tournament_id, json.dumps(entries), datetime.now(timezone.utc).isoformat(timespec="seconds"))
    )
    conn.commit()
    conn.close()

    return entries


def fetch_matches(tournament_id, player_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get tournament end_date
    cur.execute("SELECT end_date FROM tournaments WHERE tournament_id=?", (tournament_id,))
    row = cur.fetchone()
    end_date = None
    if row and row["end_date"]:
        try:
            end_date = datetime.fromisoformat(row["end_date"]).date()
        except Exception:
            pass

    today = datetime.now(timezone.utc).date()

    # If tournament ended → return cached matches
    if end_date and today >= end_date:
        cur.execute("SELECT data FROM matches WHERE tournament_id=? AND player_id=?", (tournament_id, player_id))
        row = cur.fetchone()
        conn.close()
        if row:
            return json.loads(row["data"])  # matches list
        return []  # nothing cached

    conn.close()

    # Otherwise scrape fresh
    url = f"https://ti.tournamentsoftware.com/tournament/{tournament_id}/player/{player_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    matches = []
    for match_div in soup.select(".match-group__item .match"):
        draw = match_div.select_one(".match__header-title .nav-link__value")
        draw_name = draw.get_text(strip=True) if draw else ""
        players = [a.get_text(strip=True) for a in match_div.select(".match__row-title-value-content a")]
        statuses = [s.get_text(strip=True) for s in match_div.select(".match__status")]
        scores = [[li.get_text(strip=True) for li in ul.select("li")] for ul in match_div.select(".match__result ul.points")]
        date_time = match_div.select_one(".icon-clock + .nav-link__value")
        court = match_div.select_one(".icon-marker + .nav-link__value")
        h2h_link = match_div.select_one(".match__btn-h2h")
        h2h_url = h2h_link["href"] if h2h_link else ""

        matches.append({
            "draw": draw_name,
            "players": players,
            "statuses": statuses,
            "scores": scores,
            "date_time": date_time.get_text(strip=True) if date_time else "",
            "court": court.get_text(strip=True) if court else "",
            "h2h_url": h2h_url
        })

    # Cache scraped matches
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO matches (tournament_id, player_id, data) VALUES (?, ?, ?)",
        (tournament_id, player_id, json.dumps(matches))
    )
    conn.commit()
    conn.close()

    return matches



def extract_entries(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = pd.read_html(StringIO(str(soup)))
    if not tables:
        return []

    # Pick the largest table
    largest = max(tables, key=lambda t: len(t))

    # Convert all columns to text
    largest = largest.astype(str)

    # Find header row with "Player" and "Seed"
    header_index = -1
    for i, row in largest.iterrows():
        values = row.values.tolist()
        if "Player" in values and "Seed" in values:
            header_index = i
            break

    if header_index >= 0:
        largest.columns = largest.iloc[header_index]
        largest = largest.iloc[header_index + 1:]

    # Drop rows with empty Player
    if "Player" in largest.columns:
        largest = largest[largest["Player"].str.strip() != ""]

    # Rename first column to "Draw"
    first_col = largest.columns[0]
    largest = largest.rename(columns={first_col: "Draw"})

    # Normalize column names
    largest.columns = [c.strip().lower().replace(" ", "_") for c in largest.columns]

    return largest.to_dict(orient="records")

# --- Cleaning helpers ---

def clean_seed(seed) -> str:
    if pd.isna(seed):   # catches NaN
        return ""
    return str(seed).strip()

def clean_player_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    return re.sub(r"^\[[A-Z]{3}\]\s*", "", name.strip())


def sort_entries(df):
    # Clean player names
    df["player"] = df["player"].astype(str).str.strip()
    df["player"] = df["player"].apply(
        lambda name: re.sub(r"^\[[A-Z]{3}\]\s*", "", name)
    )
    df["player_norm"] = df["player"].str.lower().str.strip()

    # Normalize Draw column
    df["draw_lower"] = df["Draw"].astype(str).str.lower()

    # Assign priority
    df["draw_priority"] = np.select(
        [
            df["draw_lower"].str.contains("maindraw"),
            df["draw_lower"].str.contains("qualification"),
            df["draw_lower"].str.contains("reserve"),
            df["draw_lower"].str.contains("exclude"),
        ],
        [1, 2, 3, 4],
        default=5
    )

    # Ensure rank is numeric
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    # Reserve list number extraction
    df["reserve_num"] = df["draw_lower"].str.extract(r"(\d+)").astype(float)
    df["reserve_num"] = df["reserve_num"].fillna(9999)

    # 👉 Split into maindraw vs others
    maindraw_df = df[df["draw_lower"].str.contains("maindraw")].sort_values(
        by=["rank"], ascending=[True]
    )
    other_df = df[~df["draw_lower"].str.contains("maindraw")].sort_values(
        by=["draw_priority", "reserve_num", "draw_lower", "rank"],
        ascending=[True, True, True, True]
    )

    # Concatenate back together
    df_sorted = pd.concat([maindraw_df, other_df], ignore_index=True)

    return df_sorted.drop(columns=["draw_lower", "draw_priority", "reserve_num"])



def format_display_name(s: str) -> str:
    """Capitalize names for display, handling hyphens and apostrophes."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Normalize apostrophe-like characters to ASCII '
    s = s.replace("’", "'").replace("`", "'").replace("´", "'")
    s = s.lower().strip()

    def cap_token(tok: str) -> str:
        # Split on hyphen or apostrophe, keep separators
        parts = re.split(r"([\'-])", tok)
        return "".join(p.capitalize() if p not in ("'", "-") else p for p in parts)

    tokens = s.split()
    return " ".join(cap_token(t) for t in tokens if t)


if app.config["ENV"] == "development":
    @app.after_request
    def add_header(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
@app.route("/")
def home():
    updated = request.args.get("updated") == "1"
    return render_template("base.html", age_groups=AGE_GROUPS, updated=updated)

@app.route("/settings/cookie", methods=["GET", "POST"])
def edit_cookie():
    saved = request.args.get("saved") == "1"

    if request.method == "POST":
        new_cookie = request.form.get("cookie", "").strip()
        COOKIE_FILE.write_text(new_cookie, encoding="utf-8")
        return redirect(url_for("edit_cookie", saved=1))

    current_cookie = ""
    if COOKIE_FILE.exists():
        current_cookie = COOKIE_FILE.read_text(encoding="utf-8")

    return render_template(
        "cookie.html",
        cookie=current_cookie,
        saved=saved,
        age_groups=AGE_GROUPS,
    )

@app.route("/rankings")
def rankings():
    age_group = request.args.get("age_group", "BS14")  # default BS14
    week = request.args.get("week")
    q = request.args.get("q", "").strip()  # search query

    if age_group not in AGE_GROUPS:
        age_group = "BS14"

    rows = []
    updated_at = None

    # weeks dropdown
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT Week FROM rankings WHERE AgeGroup = ?",
        (age_group,),
    )
    weeks = [r["Week"] for r in cur.fetchall()]
    conn.close()

    # ✅ Sort weeks numerically by (year, week number), descending
    def parse_week(w):
        try:
            week_num, year = w.split("-")
            return (int(year), int(week_num))
        except Exception:
            return (0, 0)

    weeks = sorted(weeks, key=parse_week, reverse=True)

    # ✅ If no week is provided, pick the latest one
    if not week and weeks:
        week = weeks[0]

    if week:
        conn = get_db_connection()
        cur = conn.cursor()

        if q:
            cur.execute(
                """
                SELECT *
                FROM rankings
                WHERE AgeGroup = ? AND Week = ? AND Player LIKE ?
                ORDER BY CAST(Rank AS INTEGER), Player ASC
                """,
                (age_group, week, f"%{q}%"),
            )
        else:
            cur.execute(
                """
                SELECT *
                FROM rankings
                WHERE AgeGroup = ? AND Week = ?
                ORDER BY CAST(Rank AS INTEGER), Player ASC
                """,
                (age_group, week),
            )
        rows = cur.fetchall()

        try:
            cur.execute(
                """
                SELECT MAX(UpdatedAt) AS UpdatedAt
                FROM rankings
                WHERE AgeGroup = ? AND Week = ?
                """,
                (age_group, week),
            )
            row2 = cur.fetchone()
            if row2 and row2["UpdatedAt"]:
                raw_updated = row2["UpdatedAt"]
                try:
                    dt = datetime.fromisoformat(raw_updated)
                    updated_at = f"{dt.date()} Time: {dt.time()}"
                except Exception:
                    updated_at = raw_updated
        except Exception:
            updated_at = None

        conn.close()

    return render_template(
        "rankings.html",
        age_group=age_group,
        week=week,
        weeks=weeks,
        rows=rows,
        age_groups=AGE_GROUPS,
        updated_at=updated_at,
        q=q,
    )


@app.route("/database", methods=["GET", "POST"])
def database_admin():
    status = None
    error = None

    if request.method == "POST":
        action = request.form.get("action")

        try:
            if action == "update":
                run_update()
                status = "Database updated with current week's rankings."

            elif action == "erase":
                reset_rankings_table()
                status = "Rankings table erased. Run an update or add a week to recreate it."

            elif action == "add_week":
                week_id = request.form.get("week_id", "").strip()
                week_label = request.form.get("week_label", "").strip()

                if not week_id:
                    error = "Week ID is required."
                elif not week_label:
                    error = "Week label is required."
                else:
                    add_ranking_week(week_id, week_label)
                    status = f"Ranking week '{week_label}' added from WeekID {week_id}."

        except Exception as e:
            error = f"Error: {e}"

    return render_template(
        "database.html",
        status=status,
        error=error,
        age_groups=AGE_GROUPS,
    )

@app.route("/admin/update")
def admin_update():
    # Run the weekly update (this may take a few seconds)
    run_update()
    # Redirect back home with a flag so we can show a message
    return redirect(url_for("home", updated=1))

@app.route("/points", methods=["GET", "POST"])
def points():
    ensure_points_table()

    # Load categories fresh from DB (only active ones)
    raw_categories = load_categories(include_deleted=False)  # must return code, display_name, type

    def _numeric_part(code: str) -> int:
        digits = ''.join(ch for ch in code if ch.isdigit())
        return int(digits) if digits else 0

    def sort_by_numeric(items):
        # items are (code, display_name) tuples
        return sorted(items, key=lambda x: _numeric_part(x[0]))

    # keep tuples (code, display_name)
    domestic = [(r["code"], r["display_name"]) for r in raw_categories if r["type"] == "domestic"]
    international = [(r["code"], r["display_name"]) for r in raw_categories if r["type"] == "international"]

    # Sort each group
    domestic_sorted = sort_by_numeric(domestic)          # T100 → T200 → T500 → T1000
    international_sorted = sort_by_numeric(international) # TE1 → TP500 → TP1000

    # Merge in desired order for the points table
    point_columns = domestic_sorted + international_sorted

    points_map = load_points_map(point_columns)

    if request.method == "POST":
        action = request.form.get("action")

        conn = get_db_connection()
        cur = conn.cursor()

        if action == "add_category":
            code = request.form["new_code"].strip()
            display = request.form["new_display"].strip()
            type_ = request.form["new_type"]

            # check if category already exists (even if deleted)
            cur.execute("SELECT deleted FROM categories WHERE code = ?", (code,))
            row = cur.fetchone()
            if row:
                # if it exists but is marked deleted, undelete it
                cur.execute(
                    "UPDATE categories SET display_name = ?, type = ?, deleted = 0 WHERE code = ?",
                    (display, type_, code)
                )
            else:
                # insert new category
                cur.execute(
                    "INSERT INTO categories (code, display_name, type, deleted) VALUES (?, ?, ?, 0)",
                    (code, display, type_)
                )
                # add column to points table
                cur.execute(f'ALTER TABLE points ADD COLUMN "{code}" INTEGER')

        elif action == "delete_category":
            code = request.form["delete_code"]
            # soft delete
            cur.execute("UPDATE categories SET deleted = 1 WHERE code = ?", (code,))

        else:
            # update points values
            for place in range(0, 33):
                for col_name, _ in point_columns:
                    field_name = f"{col_name}_{place}"
                    val = request.form.get(field_name)
                    quoted_col = f'"{col_name}"'
                    if val is None or val == "":
                        cur.execute(f"UPDATE points SET {quoted_col} = NULL WHERE Place = ?", (place,))
                    else:
                        cur.execute(f"UPDATE points SET {quoted_col} = ? WHERE Place = ?", (int(val), place))

        conn.commit()
        conn.close()
        return redirect(url_for("points", saved=1))

    # GET: load current values
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM points ORDER BY Place;")
    rows = cur.fetchall()
    conn.close()

    saved = request.args.get("saved") == "1"

    return render_template(
        "points.html",
        rows=rows,
        point_columns=point_columns,  # ✅ sorted
        saved=saved,
        age_groups=AGE_GROUPS,
    )


@app.route("/tournaments", methods=["GET", "POST"])
def tournaments():
    ensure_tournaments_table()

    # Define options for Status dropdown
    STATUS_OPTIONS = [
        "Interest",
        "Entered",
        "Played",
        "W/D",
        "Rank ↓",
        "Rank ↑",
    ]

    # Player name & age_group selection
    player_name = request.args.get("player", "").strip()
    if request.method == "POST":
        player_name = request.form.get("player_name", "").strip()

    if not player_name:
        player_name = "Kevin Nita"  # default

    age_group = request.args.get("age_group", "BS14")
    if request.method == "POST":
        age_group = request.form.get("age_group", age_group)

    if age_group not in AGE_GROUPS:
        age_group = "BS14"

    
    action = request.form.get("action")

    # Save edits
    if request.method == "POST" and action == "save":
        conn = get_db_connection()
        cur = conn.cursor()
        for key, value in request.form.items():
            if key.startswith("place_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET place=? WHERE id=?", (value or None, row_id))
            elif key.startswith("won_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET won=? WHERE id=?", (value or None, row_id))
            elif key.startswith("lost_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET lost=? WHERE id=?", (value or None, row_id))
            elif key.startswith("cat_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET category_code=? WHERE id=?", (value or None, row_id))
            elif key.startswith("name_"):   # Tournament name
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET tournament_name=? WHERE id=?", (value or None, row_id))
            elif key.startswith("start_"):  # Start date
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET start_date=? WHERE id=?", (value or None, row_id))
            elif key.startswith("finish_"): # Finish date
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET end_date=? WHERE id=?", (value or None, row_id))
            elif key.startswith("opens_"):  # Opens date
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET opens_date=? WHERE id=?", (value or None, row_id))
            elif key.startswith("close_"):  # Close date
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET close_date=? WHERE id=?", (value or None, row_id))
            elif key.startswith("status_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET status=? WHERE id=?", (value or None, row_id))
            elif key.startswith("event_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET event_id = ? WHERE id = ?", (value if value != "" else None, row_id))
            elif key.startswith("draw_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET draw_id = ? WHERE id = ?", (value if value != "" else None, row_id))
            elif key.startswith("player_"):
                row_id = key.split("_")[1]
                cur.execute("UPDATE tournaments SET player_id = ? WHERE id = ?", (value if value != "" else None, row_id))    

        conn.commit()
        conn.close()
        flash("Tournament Saved")
        return redirect(url_for("tournaments", player=player_name, age_group=age_group))

    # Add domestic tournament from TI ID
    if request.method == "POST" and action == "add_domestic":
        tournament_id = request.form.get("tournament_id", "").strip()
        if tournament_id:
            details = fetch_tournament_details(tournament_id)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO tournaments (
                    player_name, age_group, tournament_id, is_international,
                    start_date, end_date, tournament_name,
                    opens_date, close_date, category_code,
                    created_at
                )
                VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    player_name,
                    age_group,       # 👈 store BS12, BS14, etc.
                    tournament_id,
                    details.get("start_date"),
                    details.get("end_date"),
                    details.get("name"),
                    details.get("opens_date"),
                    details.get("close_date"),
                    now,
                ),
            )
            conn.commit()
            conn.close()
        return redirect(url_for("tournaments", player=player_name, age_group=age_group))

    # Add international/manual tournament (blank row)
    if request.method == "POST" and action == "add_international":
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tournaments (
                player_name, age_group, tournament_id, is_international,
                start_date, end_date, tournament_name,
                opens_date, close_date, category_code,
                created_at
            )
            VALUES (?, ?, NULL, 1, NULL, NULL, '', NULL, NULL, NULL, ?)
            """,
            (
                player_name,
                age_group,   # 👈 store BS12, BS14, etc.
                now,
            ),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("tournaments", player=player_name, age_group=age_group))

    # Default: load view data
    view = build_tournament_view(player_name, age_group)
    rows = view["rows"]
    totals = view["totals"]
    cat_counts = view["cat_counts"]
    cat_points = view["cat_points"]

    return render_template(
        "tournaments.html",
        player_name=player_name,
        age_group=age_group,
        age_groups=AGE_GROUPS,
        rows=rows,
        totals=totals,
        cat_counts=cat_counts,
        cat_points=cat_points,
        point_columns=view["point_columns"],
        latest_week=view["latest_week"],
        latest_rank=view["latest_rank"], 
        best_rank=view["best_rank"],
        best_rank_week=view["best_rank_week"],
        latest_wtn=view["latest_wtn"],
        best_wtn=view["best_wtn"],
        best_wtn_week=view["best_wtn_week"],
        stat_summary=view["stat_summary"],
        domestic_columns=view["domestic_columns"], 
        international_columns=view["international_columns"]
    )
   
@app.route("/tournaments/delete/<int:row_id>", methods=["POST"])
def delete_tournament(row_id):
    player_name = request.form.get("player_name", "").strip() or "Kevin Nita"
    age_group = request.form.get("age_group", "BS14")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM tournaments WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("tournaments", player=player_name, age_group=age_group))


@app.route("/entries")
def entries_page():
    tournament_id = request.args.get("tournament_id")
    event_id = request.args.get("event_id", type=int)

    if not tournament_id or not event_id:
        return "Missing tournament_id or event_id", 400

    # 🔄 Use cached helper instead of scraping directly
    entries = fetch_entries(tournament_id)
    entries_df = pd.DataFrame(entries)

    # Load rankings
    rankings_df = load_rankings().copy()
    rankings_df["name_norm"] = rankings_df["name"].apply(normalize_name)

    # ✅ Fill NaN in the seed column
    if "seed" in entries_df.columns:
        entries_df["seed"] = entries_df["seed"].fillna("").astype(str).replace("nan", "")

    # --- Return to browser ---
    return render_template(
        "entries.html",
        tournament_id=tournament_id,
        event_id=event_id,
        entries=entries_df.to_dict(orient="records"),
        rankings=rankings_df.to_dict(orient="records")
    )


@app.route("/import_entries/<tournament_id>", methods=["GET", "POST"])
def import_entries(tournament_id):
    # print("HIT /import_entries (start)")
    if request.method == "POST":
        draw_id = request.form.get("draw_id")
        
        url = f"https://ti.tournamentsoftware.com/sport/event.aspx?id={tournament_id}&event={draw_id}"
        headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
        resp = requests.get(url, headers=headers, timeout=10)

        # Parse tables (raw)
        tables = pd.read_html(resp.text)
        # print(f"Found {len(tables)} tables")
        # print("Raw entries table sample (before formatting):\n", tables[1].head())

        # Use the entries table
        entries_df = tables[1].copy()

        # Assign headers
        entries_df.columns = ["draw", "player", "seed"] + [f"col{i}" for i in range(3, len(entries_df.columns))]

        # Display-format names and clean seed
        entries_df["player"] = entries_df["player"].astype(str).apply(format_display_name)
        entries_df["seed"] = entries_df["seed"].apply(clean_seed)

        # Clean player names
        entries_df["player"] = entries_df["player"].astype(str).apply(clean_player_name)

        # Normalize player names for matching
        entries_df["player_norm"] = entries_df["player"].str.lower().str.strip()

        # Normalize column names
        entries_df.columns = [c.strip().lower().replace(" ", "_") for c in entries_df.columns]
        
        # Rankings
        rankings_df = load_rankings().copy()
        rankings_df.columns = [c.strip().lower().replace(" ", "_") for c in rankings_df.columns]

        if "player_name" in rankings_df.columns:
            rankings_df.rename(columns={"player_name": "player"}, inplace=True)
        if "agegroup" not in rankings_df.columns and "age_group" in rankings_df.columns:
            rankings_df.rename(columns={"age_group": "agegroup"}, inplace=True)


        # Filter latest week
        latest_week = rankings_df["week"].max()
        latest_rankings = rankings_df[rankings_df["week"] == latest_week].copy()

        # Best rank per normalized player
        latest_rankings["rank_num"] = pd.to_numeric(latest_rankings["rank"], errors="coerce")
        best_rankings = (
            latest_rankings.sort_values("rank_num", na_position="last")
            .groupby("player_norm", as_index=False)
            .first()
        )

        # Merge on normalized key
        merged_df = entries_df.merge(best_rankings, on="player_norm", how="left")


        # Fill defaults
        merged_df["rank"] = pd.to_numeric(merged_df["rank"], errors="coerce").fillna(999).astype(int)
        if "agegroup" in merged_df.columns:
            merged_df["agegroup"] = merged_df["agegroup"].fillna("N/A")
        for col in ["wtn", "ranking_points", "year_of_birth", "tournaments", "province"]:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna("N/A")

        # Merge entries with rankings on normalized name
    merged_df = entries_df.merge(rankings_df, on="player_norm", how="left")

    # Fuzzy match fallback for players with missing rank
    for idx, row in merged_df.iterrows():
        if pd.isna(row.get("rank")) or str(row.get("rank")).strip() == "" or row.get("rank") == 999:
            # Try fuzzy match against rankings
            match = process.extractOne(row["player_norm"], rankings_df["player_norm"].tolist())
            if match and match[1] >= 85:  # threshold score
                candidate = match[0]
                rank_row = rankings_df[rankings_df["player_norm"] == candidate].iloc[0]
                # Fill in missing fields from the matched ranking row
                for col in ["rank", "wtn", "ranking_points", "year_of_birth", "tournaments", "province", "agegroup"]:
                    if col in rank_row:
                        merged_df.at[idx, col] = rank_row[col]


        merged_df = sort_entries(merged_df)
        entries = merged_df.to_dict(orient="records")

        matches = (merged_df["rank"] != 999).sum()
        total = len(merged_df)
        # print(f"Matched {matches}/{total} entries")

        return render_template("entries.html", tournament_id=tournament_id, entries=entries)

    return render_template("import_entries_form.html", tournament_id=tournament_id)


@app.route("/entries/<tournament_id>")
def entries(tournament_id):
    event_id = request.args.get("event_id", type=int)
    if not event_id:
        return "Missing event_id", 400

    # 🔄 Use cached helper instead of scraping directly
    entries = fetch_entries(tournament_id)

    # Convert list of dicts into DataFrame for downstream merging/sorting
    entries_df = pd.DataFrame(entries)
    entries_df.rename(columns={"draw": "Draw", "player": "Player", "seed": "Seed"}, inplace=True)
    entries_df.rename(columns={"Unnamed: 0": "Draw", "Player": "player"}, inplace=True)

    # Clean player names
    entries_df["player"] = entries_df["player"].astype(str).apply(clean_player_name)
    entries_df["player_norm"] = entries_df["player"].str.lower().str.strip()

    # Load rankings
    rankings_df = load_rankings().copy()
    rankings_df.columns = [c.strip().lower().replace(" ", "_") for c in rankings_df.columns]
    rankings_df["player_norm"] = rankings_df["player"].astype(str).apply(normalize_name)
    rankings_df["rank_num"] = pd.to_numeric(rankings_df["rank"], errors="coerce")

    # ✅ Tournament info
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT close_date, tournament_name, draw_id, player_id, age_group FROM tournaments WHERE tournament_id = ?",
        (tournament_id,)
    )
    row = cur.fetchone()
    conn.close()

    close_date = row["close_date"] if row else None
    tournament_name = row["tournament_name"] if row else ""
    draw_id = row["draw_id"] if row else None
    player_id = row["player_id"] if row else None
    tournament_age_group = row["age_group"] if row else None

    def parse_week_label(s: str):
        try:
            w, y = s.split("-")
            return int(w), int(y)
        except Exception:
            return None, None

    # Normalize rankings week labels to numeric components
    rankings_df["week_label"] = rankings_df["week"].astype(str)
    rankings_df[["week_num", "week_year"]] = rankings_df["week_label"].apply(parse_week_label).tolist()

    # Guard against bad/missing labels
    valid_weeks = rankings_df.dropna(subset=["week_num", "week_year"]).copy()
    valid_weeks["week_num"] = valid_weeks["week_num"].astype(int)
    valid_weeks["week_year"] = valid_weeks["week_year"].astype(int)

    # Compute target_week correctly
    today = datetime.now(timezone.utc).date()
    if close_date:
        try:
            close_date_obj = datetime.fromisoformat(close_date).date()
        except ValueError:
            d, m, y = close_date.split("/")
            close_date_obj = datetime(int(y), int(m), int(d), tzinfo=timezone.utc).date()

        if today < close_date_obj:
            latest = valid_weeks.sort_values(["week_year", "week_num"]).iloc[-1]
            target_week = f"{latest['week_num']}-{latest['week_year']}"
        else:
            week_num = close_date_obj.isocalendar()[1]
            year = close_date_obj.isocalendar()[0]
            target_week = f"{week_num}-{year}"
    else:
        latest = valid_weeks.sort_values(["week_year", "week_num"]).iloc[-1]
        target_week = f"{latest['week_num']}-{latest['week_year']}"

    selected_rankings = rankings_df[rankings_df["week_label"] == target_week].copy()
    print(f"[INFO] Rankings imported from week {target_week}", flush=True)

    # Step 1: rankings for this tournament's age_group
    if tournament_age_group:
        age_group_rankings = selected_rankings[selected_rankings["agegroup"] == tournament_age_group]
    else:
        age_group_rankings = pd.DataFrame()

    # Step 2: best rank across all age groups
    best_rankings_all = (
        selected_rankings.loc[selected_rankings.groupby("player_norm")["rank_num"].idxmin()]
        .reset_index(drop=True)
    )

    # Merge entries with rankings
    merged_df = entries_df.merge(age_group_rankings, on="player_norm", how="left", suffixes=("", "_age"))
    merged_df = merged_df.merge(best_rankings_all, on="player_norm", how="left", suffixes=("", "_best"))

    def choose_value(row, col):
        if pd.notna(row.get(col)):
            return row[col]
        elif pd.notna(row.get(f"{col}_best")):
            return row[f"{col}_best"]
        else:
            return 999 if col == "rank" else "N/A"

    for col in ["rank", "agegroup", "wtn", "ranking_points", "total_points",
                "tournaments", "avg_points", "province", "year_of_birth"]:
        merged_df[col] = merged_df.apply(lambda r: choose_value(r, col), axis=1)

    merged_df["rank"] = pd.to_numeric(merged_df["rank"], errors="coerce").fillna(999).astype(int)

    if "Seed" in merged_df.columns:
        merged_df["Seed"] = merged_df["Seed"].fillna("").astype(str).replace("nan", "")

    ranking_cols = [
        "member_id", "year_of_birth", "wtn", "ranking_points", "total_points",
        "tournaments", "avg_points", "province", "agegroup", "week", "updatedat"
    ]
    for col in ranking_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("N/A")

    # ✅ Custom sort
    merged_df["draw_lower"] = merged_df["Draw"].astype(str).str.lower()
    merged_df["draw_priority"] = np.select(
        [
            merged_df["draw_lower"].str.contains("maindraw"),
            merged_df["draw_lower"].str.contains("qualification"),
            merged_df["draw_lower"].str.contains("reserve"),
            merged_df["draw_lower"].str.contains("exclude"),
        ],
        [1, 2, 3, 4],
        default=5
    )
    merged_df["orig_idx"] = range(len(merged_df))

    def sort_entries(df, tournament_age_group):
        maindraw = df[df["draw_priority"] == 1].copy()
        maindraw["is_tournament_age"] = maindraw["agegroup"].eq(tournament_age_group)
        maindraw = maindraw.sort_values(
            by=["is_tournament_age", "rank", "orig_idx"],
            ascending=[False, True, True],
            kind="mergesort"
        )
        qualification = df[df["draw_priority"] == 2].copy().sort_values(
            by=["rank", "orig_idx"], ascending=[True, True], kind="mergesort"
        )
        reserve = df[df["draw_priority"] == 3].copy()
        def extract_reserve_number(draw_str):
            m = re.search(r"reserve\s*list\s*(\d+)$", str(draw_str).lower())
            return int(m.group(1)) if m else None
        reserve["reserve_num"] = reserve["Draw"].apply(extract_reserve_number)
        reserve_numbered = reserve[reserve["reserve_num"].notna()].copy().sort_values(
            by=["reserve_num", "orig_idx"], ascending=[True, True], kind="mergesort"
        )
        reserve_plain = reserve[reserve["reserve_num"].isna()].copy().sort_values(
            by=["Draw", "orig_idx"], ascending=[True, True], kind="mergesort"
        )
        reserve = pd.concat([reserve_numbered, reserve_plain])
        exclude = df[df["draw_priority"] == 4].copy().sort_values(by=["orig_idx"], kind="mergesort")
        others = df[df["draw_priority"] == 5].copy().sort_values(by=["orig_idx"], kind="mergesort")
        return pd.concat([maindraw, qualification, reserve, exclude, others]).reset_index(drop=True)

    merged_df = sort_entries(merged_df, tournament_age_group)
    entries = merged_df.to_dict(orient="records")

    return render_template(
        "entries.html",
        tournament_id=tournament_id,
        event_id=event_id,
        entries=entries,
        tournament_name=tournament_name,
        draw_id=draw_id,
        player_id=player_id,
        tournament_age_group=tournament_age_group,
    )



@app.route("/matches/<tournament_id>/<player_id>")
def matches(tournament_id, player_id):
    refresh = request.args.get("refresh") == "1"
    # --- Ensure matches table exists ---
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            tournament_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            data TEXT NOT NULL,
            updated_at TEXT,
            PRIMARY KEY (tournament_id, player_id)
        )
    """)
    conn.commit()

    # --- Add updated_at column if missing ---
    try:
        cur.execute("ALTER TABLE matches ADD COLUMN updated_at TEXT")
        conn.commit()
        print("[INFO] Added updated_at column to matches table")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # --- Get tournament info ---
    cur.execute("SELECT end_date, tournament_name, draw_id, event_id FROM tournaments WHERE tournament_id=?", (tournament_id,))
    row = cur.fetchone()
    conn.close()

    end_date = None
    if row and row["end_date"]:
        try:
            end_date = datetime.fromisoformat(row["end_date"]).date()
        except Exception:
            pass

    today = datetime.now(timezone.utc).date()
    tournament_name = row["tournament_name"] if row else f"Tournament {tournament_id}"
    draw_id = row["draw_id"] if row else None
    event_id = row["event_id"] if row else None

    # --- Check cache ---
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT data, updated_at FROM matches WHERE tournament_id=? AND player_id=?", (tournament_id, player_id))
    cached = cur.fetchone()
    conn.close()

    use_cache = False

    if not refresh and cached and cached["updated_at"]:
        try:
            cache_date = datetime.fromisoformat(cached["updated_at"]).date()
            if end_date and cache_date >= end_date:
                use_cache = True
        except Exception:
            pass

    if use_cache:
        print(f"[INFO] Using cached matches for {tournament_id}/{player_id} (updated_at={cached['updated_at']})")
        matches = json.loads(cached["data"])
        soup = None
    else:
        print(f"[INFO] Cache missing/stale, scraping fresh matches for {tournament_id}/{player_id}")
        url = f"https://ti.tournamentsoftware.com/tournament/{tournament_id}/player/{player_id}"
        headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        matches = []
        for match_div in soup.select(".match-group__item .match"):
            draw = match_div.select_one(".match__header-title .nav-link__value")
            draw_name = draw.get_text(strip=True) if draw else ""
            players = [a.get_text(strip=True) for a in match_div.select(".match__row-title-value-content a")]
            statuses = [s.get_text(strip=True) for s in match_div.select(".match__status")]
            scores = [[li.get_text(strip=True) for li in ul.select("li")] for ul in match_div.select(".match__result ul.points")]
            date_time = match_div.select_one(".icon-clock + .nav-link__value")
            court = match_div.select_one(".icon-marker + .nav-link__value")
            h2h_link = match_div.select_one(".match__btn-h2h")
            h2h_url = h2h_link["href"] if h2h_link else ""

            matches.append({
                "draw": draw_name,
                "players": players,
                "statuses": statuses,
                "scores": scores,
                "date_time": date_time.get_text(strip=True) if date_time else "",
                "court": court.get_text(strip=True) if court else "",
                "h2h_url": h2h_url
            })

        # --- Cache scraped matches with updated_at ---
        print(f"[DEBUG] Caching {len(matches)} matches for tournament {tournament_id}, player {player_id}")
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO matches (tournament_id, player_id, data, updated_at) VALUES (?, ?, ?, ?)",
            (tournament_id, player_id, json.dumps(matches), datetime.now(timezone.utc).isoformat(timespec="seconds"))
        )
        conn.commit()
        conn.close()

    # --- Parse statistics table (all columns) ---
    stats = []
    stats_table = soup.select_one(".module.module--card .table--new") if soup else None
    if stats_table:
        for row in stats_table.select("tbody tr"):
            cells = []
            for idx, td in enumerate(row.select("td")):
                cell_text = td.get_text(strip=True)
                percent = None
                if idx == 2:  # W column
                    bar = td.select_one(".progress-bar__line")
                    if bar and bar.has_attr("style"):
                        width = bar["style"].replace("width:", "").replace("%;", "").replace("%", "").strip()
                        try:
                            percent = int(width)
                        except ValueError:
                            percent = None
                cells.append({"text": cell_text, "percent": percent})
            if cells:
                stats.append(cells)

    # --- Fetch Singles stats separately ---
    singles_won, singles_lost = fetch_singles_stats(tournament_id, player_id)

    # --- Update tournaments table with Won/Lost from Singles row ---
    if singles_won is not None and singles_lost is not None:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE tournaments SET won=?, lost=? WHERE tournament_id=?",
            (singles_won, singles_lost, tournament_id)
        )
        conn.commit()
        conn.close()

    # --- Build stat_summary (unchanged) ---
    def extract_wl(cells):
        try:
            wl_text = cells[2]["text"]
            wl_part = wl_text.split()[0] if wl_text else ""
            if "-" in wl_part:
                won, lost = map(int, wl_part.split("-"))
                return won, lost
        except Exception:
            pass
        return 0, 0

    singles_won_summary, singles_lost_summary = 0, 0
    doubles_won_summary, doubles_lost_summary = 0, 0

    for row in stats:
        label = row[0]["text"].lower()
        if label == "singles":
            singles_won_summary, singles_lost_summary = extract_wl(row)
        elif label == "doubles":
            doubles_won_summary, doubles_lost_summary = extract_wl(row)

    total_won = singles_won_summary + doubles_won_summary
    total_lost = singles_won_summary + doubles_lost_summary

    def pct(part, total):
        return round(part / total * 100, 1) if total > 0 else 0.0

    stat_summary = {
        "singles": {"won": singles_won_summary, "lost": singles_lost_summary,
                    "won_pct": pct(singles_won_summary, singles_won_summary + singles_lost_summary),
                    "lost_pct": pct(singles_lost_summary, singles_won_summary + singles_lost_summary),
                    "total": singles_won_summary + singles_lost_summary},
        "doubles": {"won": doubles_won_summary, "lost": doubles_lost_summary,
                    "won_pct": pct(doubles_won_summary, doubles_won_summary + doubles_lost_summary),
                    "lost_pct": pct(doubles_lost_summary, doubles_won_summary + doubles_lost_summary),
                    "total": doubles_won_summary + doubles_lost_summary},
        "total": {"won": total_won, "lost": total_lost,
                  "won_pct": pct(total_won, total_won + total_lost),
                  "lost_pct": pct(total_lost, total_won + total_lost),
                  "total": total_won + total_lost}
    }

    return render_template(
        "matches.html",
        matches=matches,
        stats=stats,
        tournament_id=tournament_id,
        tournament_name=tournament_name,
        player_id=player_id,
        singles_won=singles_won,
        singles_lost=singles_lost,
        stat_summary=stat_summary,
        draw_id=draw_id,
        event_id=event_id
    )




@app.route("/categories", methods=["GET","POST"])
def categories():
    conn = get_db_connection()
    cur = conn.cursor()

    if request.method == "POST":
        action = request.form.get("action")
        if action == "add":
            name = request.form.get("name").strip()
            type_ = request.form.get("type")
            cur.execute("INSERT INTO categories (name,type) VALUES (?,?)",(name,type_))
        elif action == "delete":
            cat_id = request.form.get("id")
            cur.execute("DELETE FROM categories WHERE id=?",(cat_id,))
        elif action == "edit":
            cat_id = request.form.get("id")
            name = request.form.get("name").strip()
            type_ = request.form.get("type")
            cur.execute("UPDATE categories SET name=?, type=? WHERE id=?",(name,type_,cat_id))
        conn.commit()

    cur.execute("SELECT * FROM categories ORDER BY type, code")
    cats = cur.fetchall()
    conn.close()
    return render_template("categories.html", categories=cats)

    
@app.route("/player")
def player():
    name = request.args.get("name", "Kevin Nita").strip()
    age_group = request.args.get("age_group", "BS14")  # selected age group from query string

    def parse_week(w):
        try:
            week_num, year = w.split("-")
            return (int(year), int(week_num))
        except Exception:
            return (9999, 9999)

    if not name:
        # simple form on the same page
        return render_template("player.html", name=None, rows=[], age_groups=AGE_GROUPS)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT Week, AgeGroup, Rank, Player, wtn, [Ranking points], [Total points]
        FROM rankings
        WHERE Player = ?
        ORDER BY Week, AgeGroup
        """,
        (name,),
    )
    rows = cur.fetchall()
    conn.close()

    # --- Build WTN timeline
    def week_label_to_date(label: str) -> date: 
        # label like "51-2025" 
        w, y = label.split("-") 
        return date.fromisocalendar(int(y), int(w), 1) # Monday of that ISO week
    wtn_weeks = []
    wtn_values = []
    for r in rows:
        if r["WTN"] is not None:
            wtn_weeks.append(r["Week"])
            wtn_values.append(float(r["WTN"]))

    if wtn_weeks and wtn_values:
        # Sort by week ascending
        sorted_pairs = sorted(zip(wtn_weeks, wtn_values), key=lambda x: week_label_to_date(x[0]))
        wtn_weeks, wtn_values = zip(*sorted_pairs)

        # Always include the latest, then step backwards every 4 weeks
        latest_date = week_label_to_date(wtn_weeks[-1])
        earliest_date = week_label_to_date(wtn_weeks[0])

        sampled_weeks = []
        sampled_values = []
        target = latest_date

        while target >= earliest_date:
            # Find the last record <= target
            for wk, val in reversed(sorted_pairs):
                if week_label_to_date(wk) <= target:
                    sampled_weeks.append(wk)
                    sampled_values.append(val)
                    break
            target -= timedelta(weeks=4)

        wtn_weeks, wtn_values = list(reversed(sampled_weeks)), list(reversed(sampled_values))

    wtn_plot_json = None
    if wtn_weeks and wtn_values:
        wtn_fig = go.Figure(
            data=[
                go.Scatter(
                    x=wtn_weeks,
                    y=wtn_values,
                    mode="lines+markers",
                    line=dict(color="blue", width=3),
                    marker=dict(size=8),
                    textposition="top center"
                )
            ],
            layout=go.Layout(
                title=f"WTN Evolution for {name}",
                xaxis=dict(title="Week", tickangle=-45),
                yaxis=dict(title="WTN", autorange="reversed"),
                margin=dict(l=40, r=20, t=50, b=80),
                height=400
            )
        )
        wtn_plot_json = json.dumps(wtn_fig, cls=plotly.utils.PlotlyJSONEncoder)


    # --- Build weekly series for the selected age group ---
    weeks = []
    ranks = []
    if age_group:
        for r in rows:
            if r["AgeGroup"] == age_group and r["Rank"] is not None:
                weeks.append(r["Week"])
                ranks.append(int(r["Rank"]))

    if weeks and ranks:
        sorted_pairs = sorted(zip(weeks, ranks), key=lambda x: parse_week(x[0]))
        weeks, ranks = zip(*sorted_pairs)

    plot_json = None
    if weeks and ranks:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=weeks,
                    y=ranks,
                    mode="lines+markers+text",
                    line=dict(color="green", width=3),
                    marker=dict(size=8),
                    textposition="top center"
                )
            ],
            layout=go.Layout(
                title=f"Ranking progress for {name} ({age_group})",
                xaxis=dict(title="Week", tickangle=-45),
                yaxis=dict(title="Rank", autorange="reversed"),
                margin=dict(l=40, r=20, t=50, b=80),
                height=400
            )
        )
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Call your tournament helper
    tournament_data = build_tournament_view(name, age_group)
    # print(f"\n=== Top 6 tournaments for {name} ({age_group}) ===")
    for r in tournament_data["top6_tournaments"]:
        print(f"{r['tournament_name']} → {r['points']}")


    # Extract top 6 tournaments for bar chart
    bar_plot_json = None
    if tournament_data["top6_tournaments"]:
        # Sort tournaments by points ascending so largest is last
        sorted_tournaments = sorted(
            tournament_data["top6_tournaments"], key=lambda r: r["points"]
        )
        names = [r["tournament_name"] for r in sorted_tournaments]
        points = [r["points"] for r in sorted_tournaments]

        y_vals = list(range(1, len(points) + 1))

        bar_fig = go.Figure(
            data=[go.Bar(
                x=points,
                y=y_vals,
                orientation="h",
                marker=dict(color="green"),
                text=names,
                textposition="inside"
            )],
            layout=go.Layout(
                title=f"Top 6 Tournaments by Points ({age_group})",
                xaxis=dict(title="Points"),
                yaxis=dict(
                    title="Rank",
                    tickmode="array",
                    tickvals=y_vals,
                    # ✅ reverse the tick labels manually
                    ticktext=[str(i) for i in reversed(y_vals)],
                    autorange="reversed"   # keep bars sorted with best at bottom
                ),
                margin=dict(l=60, r=20, t=50, b=80),
                height=400
            )
        )
        bar_plot_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)


    # Category totals chart
    cat_bar_plot_json = None
    if tournament_data["cat_points"]:
        filtered = {c: p for c, p in tournament_data["cat_points"].items() if p > 0}
        sorted_cats = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        categories = [c for c, _ in sorted_cats]
        totals = [p for _, p in sorted_cats]
        cat_fig = go.Figure(
            data=[go.Bar(
                y=categories,
                x=totals,
                orientation="h",
                marker=dict(color="green"),
                text=totals,
                textposition="inside"
            )],
            layout=go.Layout(
                title=f"Total Points by Category ({age_group})",
                xaxis=dict(title="Total Points"),
                yaxis=dict(title="Category", autorange="reversed"),
                margin=dict(l=40, r=20, t=50, b=80),
                height=400
            )
        )
        cat_bar_plot_json = json.dumps(cat_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ✅ Pass the full tournament_data dict into the template
    return render_template(
        "player.html",
        name=name,
        age_group=age_group,
        age_groups=AGE_GROUPS,
        rows=rows,
        plot_json=plot_json,
        bar_plot_json=bar_plot_json,
        cat_bar_plot_json=cat_bar_plot_json,
        wtn_plot_json=wtn_plot_json,
        category_summary=tournament_data["category_summary"],  
        grand_total=tournament_data["grand_total"],
        cat_ranking_points=tournament_data["cat_ranking_points"],
        latest_rank=tournament_data["latest_rank"],
        latest_week=tournament_data["latest_week"],
        best_rank=tournament_data["best_rank"],
        best_rank_week=tournament_data["best_rank_week"],
        latest_wtn=tournament_data["latest_wtn"],
        best_wtn=tournament_data["best_wtn"],
        best_wtn_week=tournament_data["best_wtn_week"],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
