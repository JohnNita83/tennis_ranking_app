import sqlite3
from update_current_week import update_current_week as run_update
from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, UserMixin, LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from pathlib import Path
from datetime import datetime, date, timedelta
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


COOKIE_FILE = Path(__file__).with_name("cookie.txt")

def load_cookie() -> str:
    try:
        return COOKIE_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""

DB_PATH = "rankings.db"

# Age groups we support
AGE_GROUPS = ["BS12", "BS14", "BS16", "BS18", "GS12", "GS14", "GS16", "GS18"]

# Internal column names + display labels for points table
POINT_COLUMNS = [
    ("T100", "T100"),
    ("T200", "T200"),
    ("T500", "T500"),
    ("TP500", "TP-500"),
    ("T1000", "T1000"),
    ("TP1000", "TP-1000"),
    ("T1250", "T1250"),
    ("T1500", "T1500"),
    ("T2000", "T2000"),
    ("TE3", "TE-3"),
    ("TE2", "TE-2"),
    ("TE1", "TE-1"),
]

app = Flask(__name__)
app.secret_key = "IWA@1StJohns"
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['ENV'] = 'development'
app.config['DEBUG'] = True

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
        try:
            # Call the function you imported and aliased as run_update
            run_update()
            print("Database bootstrapped with current week rankings.")
        except Exception as e:
            print(f"Bootstrap failed: {e}")
        _bootstrapped = True


from datetime import datetime, date, timedelta

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
    combined["UpdatedAt"] = datetime.utcnow().isoformat(timespec="seconds")

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

    print(f"Saved {len(combined)} rows for Week={week_value} from WeekID={week_id}.")

def ensure_tournaments_table():
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

    conn.commit()
    conn.close()

def load_points_map():
    """
    Load points table into a nested dict:
    {place: {col_code: points_int_or_None}}
    Using the internal codes in POINT_COLUMNS (e.g. T100, TP500, TE1)
    """
    ensure_points_table()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM points;")
    rows = cur.fetchall()
    conn.close()

    result = {}
    for r in rows:
        place = r["Place"]
        inner = {}
        for col_code, _disp in POINT_COLUMNS:
            inner[col_code] = r[col_code]
        result[place] = inner
    return result


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
    conn = sqlite3.connect("rankings.db")
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


from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3

def build_tournament_view(player_name: str, age_group: str) -> dict:
    ensure_tournaments_table()
    points_map = load_points_map()
    rankings_map = load_player_rankings(player_name, age_group)

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
    cat_counts = {code: 0 for code, _disp in POINT_COLUMNS}
    cat_points = {code: 0 for code, _disp in POINT_COLUMNS}

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
        total_won += won
        total_lost += lost

        if cat_code in cat_counts:
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

        row = {
            "id": row_id,
            "no": idx,
            "start_date": start_date,
            "end_date": end_date,
            "wk_label": wk_label,
            "tournament_name": r["tournament_name"],
            "cat_code": cat_code,
            "opens_date": r["opens_date"],
            "close_date": r["close_date"],
            "place": place,
            "points": pts,
            "ranking_points": None,  # set below
            "rank": rank_value,
            "diff": diff,
            "arrow": arrow,
            "won": won,
            "lost": lost,
            "matches": matches,
            "tournament_id": r["tournament_id"],
            "is_international": r["is_international"],
            "event_id": r["event_id"],
            "is_top6": False,
        }
        rows.append(row)

        # per-row ranking_points: top-6 within 365 days from this row's end_date
        current_end = parse_date_safe(end_date)
        if current_end:
            valid_points = []
            for prev in rows:  # includes current + all previous
                prev_end = parse_date_safe(prev["end_date"])
                if prev_end and 0 <= (current_end - prev_end).days <= 365:
                    p = prev["points"]
                    if p is not None and p > 0:
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
                    p = rr["points"]
                    if p is not None and p > 0:
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
    })

    for rr in rows:
        cat = rr["cat_code"]
        if not cat:
            continue
        summary = category_summary[cat]
        summary["tournaments"] += 1
        summary["total_points"] += rr["points"] if rr["points"] else 0
        place_val = rr["place"]
        if place_val is not None and place_val > 0:  # skip zeros
            if summary["best_place"] is None or place_val < summary["best_place"]:
                summary["best_place"] = place_val
        summary["won"] += rr["won"]
        summary["lost"] += rr["lost"]
        summary["total"] += rr["matches"]

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
    }

    # Ranking points per category from rows marked is_top6 (latest 12-month window)
    cat_ranking_points = {code: 0 for code, _disp in POINT_COLUMNS}
    for rr in rows:
        if rr.get("is_top6"):
            cat = rr.get("cat_code")
            pts = rr.get("points") or 0
            if cat and pts:
                cat_ranking_points[cat] += pts

    total_matches = total_won + total_lost
    won_pct = (total_won / total_matches * 100) if total_matches > 0 else 0
    lost_pct = (total_lost / total_matches * 100) if total_matches > 0 else 0

    latest_week = None
    latest_rank = None
    if rankings_map:
        def parse_week_label(label: str):
            wk, yr = label.split("-")
            return int(yr), int(wk)
        latest_week = max(rankings_map.keys(), key=parse_week_label)
        latest_rank = rankings_map[latest_week]

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
        "max_ranking_points": max_ranking_points,
        "cat_ranking_points": cat_ranking_points,
        "top6_tournaments": top6_tournaments,
        "latest_rank": latest_rank,
        "latest_week": latest_week,
        "category_summary": filtered_summary,
        "grand_total": grand_total,
    }



def get_db_connection():
    print("Using DB file:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
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

def extract_entries(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = pd.read_html(str(soup))
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
                updated_at = row2["UpdatedAt"]
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

            elif action == "add_week":   # 👈 place the block here
                week_id = request.form.get("week_id", "").strip()
                week_label = request.form.get("week_label", "").strip()

                if not week_id:
                    error = "Week ID is required."
                else:
                    add_ranking_week(week_id, week_label or None)
                    if week_label:
                        status = f"Ranking week '{week_label}' added from WeekID {week_id}."
                    else:
                        status = f"Ranking week added from WeekID {week_id}."
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
    saved = request.args.get("saved") == "1"

    if request.method == "POST":
        conn = get_db_connection()
        cur = conn.cursor()

        # loop through places 0..32
        for place in range(0, 33):
            for col_name, _ in POINT_COLUMNS:
                field_name = f"{col_name}_{place}"
                val = request.form.get(field_name)

                if val is None or val == "":
                    cur.execute(
                        f"UPDATE points SET {col_name} = NULL WHERE Place = ?",
                        (place,),
                    )
                else:
                    cur.execute(
                        f"UPDATE points SET {col_name} = ? WHERE Place = ?",
                        (int(val), place),
                    )

        conn.commit()
        conn.close()
        return redirect(url_for("points", saved=1))

    # GET: load current values
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM points ORDER BY Place;")
    rows = cur.fetchall()
    conn.close()

    return render_template(
        "points.html",
        rows=rows,
        point_columns=POINT_COLUMNS,
        saved=saved,
        age_groups=AGE_GROUPS,
    )

@app.route("/tournaments", methods=["GET", "POST"])
def tournaments():
    ensure_tournaments_table()

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

    # Debug: show what form posted
    if request.method == "POST":
        print("POST form data:", request.form)

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
            elif key.startswith("event_"):
                row_id = key.split("_")[1]
                print("Updating event_id for row", row_id, "to", value)
                cur.execute("UPDATE tournaments SET event_id = ? WHERE id = ?", (value if value != "" else None, row_id))

        conn.commit()
        conn.close()
        flash("Tournament Saved")
        return redirect(url_for("tournaments", player=player_name, age_group=age_group))

    # Add domestic tournament from TI ID
    if request.method == "POST" and action == "add_domestic":
        tournament_id = request.form.get("tournament_id", "").strip()
        if tournament_id:
            details = fetch_tournament_details(tournament_id)
            now = datetime.utcnow().isoformat(timespec="seconds")
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
        now = datetime.utcnow().isoformat(timespec="seconds")
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
        point_columns=POINT_COLUMNS,
        latest_week=view["latest_week"],
        latest_rank=view["latest_rank"], 
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

from services.entries import load_entries

@app.route("/entries_debug")
def entries_debug():
    try:
        tournament_id = request.args.get("tournament_id")
        event_id = request.args.get("event_id", type=int)

        # Fetch the event page
        url = f"https://ti.tournamentsoftware.com/sport/event.aspx?id={tournament_id}&event={event_id}"
        headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        # Parse entries table
        tables = pd.read_html(resp.text)
        entries_df = tables[1].copy()
        entries_df = entries_df.rename(columns={"Player": "player"})
        entries_df["player_norm"] = entries_df["player"].apply(normalize_name)

        # Load rankings
        rankings_df = load_rankings().copy()
        rankings_df.columns = [c.strip().lower().replace(" ", "_") for c in rankings_df.columns]
        rankings_df["player_norm"] = rankings_df["player"].astype(str).apply(normalize_name)

        # Merge entries with rankings on normalized name
        merged_df = entries_df.merge(rankings_df, on="player_norm", how="left")

        # 1. Unmatched entries
        unmatched_entries = []
        for key in set(entries_df["player_norm"]) - set(rankings_df["player_norm"]):
            raw = entries_df.loc[entries_df["player_norm"] == key, "player"].tolist()
            unmatched_entries.append({"raw": raw, "norm": key})

        # 2. Unmatched rankings
        unmatched_rankings = []
        for key in set(rankings_df["player_norm"]) - set(entries_df["player_norm"]):
            raw = rankings_df.loc[rankings_df["player_norm"] == key, "player"].tolist()
            unmatched_rankings.append({"raw": raw, "norm": key})

        # 3. Invalid rank entries (matched but rank is NaN/blank)
        invalid_rank_entries = []
        for _, row in merged_df.iterrows():
            if pd.isna(row.get("rank")) or str(row.get("rank")).strip() == "":
                invalid_rank_entries.append({
                    "raw": row.get("player_x", ""),   # entry-side name
                    "rank_player": row.get("player_y", ""),  # ranking-side name
                    "norm": row["player_norm"],
                    "reason": "no valid rank"
                })

        return render_template(
            "entries_debug.html",
            unmatched_entries=unmatched_entries,
            unmatched_rankings=unmatched_rankings,
            invalid_rank_entries=invalid_rank_entries
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route("/entries")
def entries_page():
    tournament_id = request.args.get("tournament_id")
    event_id = request.args.get("event_id", type=int)

    url = f"https://ti.tournamentsoftware.com/sport/event.aspx?id={tournament_id}&event={event_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    # --- Load entries via pandas ---
    tables = pd.read_html(resp.text)
    entries_df = tables[1]  # adjust index if needed
    entries_df = entries_df.rename(columns={"Player": "player"})
    entries_df["player_norm"] = entries_df["player"].apply(normalize_name)

    # Load rankings
    rankings_df = load_rankings()
    rankings_df["name_norm"] = rankings_df["name"].apply(normalize_name)


    # --- Return something to the browser ---
    # Option A: render a template
    return render_template(
        "entries.html",
        entries=entries_df.to_dict(orient="records"),
        rankings=rankings_df.to_dict(orient="records")
    )

    # Option B: return JSON for quick testing
    # return jsonify({
    #     "entries": entries_df.to_dict(orient="records"),
    #     "rankings": rankings_df.to_dict(orient="records")
    # })

@app.route("/import_entries/<tournament_id>", methods=["GET", "POST"])
def import_entries(tournament_id):
    print("HIT /import_entries (start)")
    if request.method == "POST":
        draw_id = request.form.get("draw_id")

        url = f"https://ti.tournamentsoftware.com/sport/event.aspx?id={tournament_id}&event={draw_id}"
        headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
        resp = requests.get(url, headers=headers)

        # Parse tables (raw)
        tables = pd.read_html(resp.text)
        print(f"Found {len(tables)} tables")
        print("Raw entries table sample (before formatting):\n", tables[1].head())

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

        # Debug merged view
        print("Merged sample:\n", merged_df[["player", "player_norm", "rank"]].head(10))

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
        print(f"Matched {matches}/{total} entries")

        return render_template("entries.html", tournament_id=tournament_id, entries=entries)

    return render_template("import_entries_form.html", tournament_id=tournament_id)


@app.route("/entries/<tournament_id>")
def entries(tournament_id):
    event_id = request.args.get("event_id", type=int)
    if not event_id:
        return "Missing event_id", 400

    url = f"https://ti.tournamentsoftware.com/sport/event.aspx?id={tournament_id}&event={event_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
    resp = requests.get(url, headers=headers)

    tables = pd.read_html(resp.text)
    entries_df = tables[1].copy()

    # Fix column names: Unnamed:0 is actually the Draw column
    entries_df.rename(columns={"Unnamed: 0": "Draw", "Player": "player"}, inplace=True)

    # Clean player names
    entries_df["player"] = entries_df["player"].astype(str).apply(clean_player_name)

    # Normalize player names
    entries_df["player_norm"] = entries_df["player"].str.lower().str.strip()

    # Load rankings
    rankings_df = load_rankings().copy()
    rankings_df.columns = [c.strip().lower().replace(" ", "_") for c in rankings_df.columns]
    rankings_df["player_norm"] = rankings_df["player"].astype(str).apply(normalize_name)

    # Convert rank to numeric
    rankings_df["rank_num"] = pd.to_numeric(rankings_df["rank"], errors="coerce")

    # ✅ Decide which week to use based on close_date and also fetch tournament_name
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT close_date, tournament_name FROM tournaments WHERE tournament_id = ?",
        (tournament_id,)
    )
    row = cur.fetchone()
    conn.close()

    close_date = row["close_date"] if row else None
    tournament_name = row["tournament_name"] if row else ""

    today = datetime.utcnow().date()
    if close_date:
        close_date_obj = datetime.fromisoformat(close_date).date()
        if today < close_date_obj:
            # Use latest week
            target_week = rankings_df["week"].max()
        else:
            # Use the week of the close date
            week_num = close_date_obj.isocalendar()[1]
            year = close_date_obj.isocalendar()[0]
            target_week = f"{week_num}-{year}"
    else:
        target_week = rankings_df["week"].max()

    selected_rankings = rankings_df[rankings_df["week"] == target_week].copy()

    # For each player_norm, keep the lowest rank across all age groups
    best_rankings = (
        selected_rankings.loc[selected_rankings.groupby("player_norm")["rank_num"].idxmin()]
        .reset_index(drop=True)
    )

    # Drop duplicate player column before merge
    if "player" in best_rankings.columns:
        best_rankings = best_rankings.drop(columns=["player"])

    # Merge on player_norm only
    merged_df = entries_df.merge(best_rankings, on="player_norm", how="left")

    # Fill missing ranks with 999
    merged_df["rank"] = pd.to_numeric(merged_df["rank"], errors="coerce").fillna(999).astype(int)

    # Debug check
    print("Merged columns:", merged_df.columns.tolist())
    print(merged_df.head())

    # Clean Seed column: replace NaN with empty string
    if "Seed" in merged_df.columns:
        merged_df["Seed"] = merged_df["Seed"].fillna("")

    # Replace NaN in all other ranking columns with "N/A"
    ranking_cols = [
        "member_id", "year_of_birth", "wtn", "ranking_points", "total_points",
        "tournaments", "avg_points", "province", "agegroup", "week", "updatedat"
    ]
    for col in ranking_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("N/A")

    # ✅ Apply your custom sort
    merged_df = sort_entries(merged_df)

    entries = merged_df.to_dict(orient="records")

    return render_template(
        "entries.html",
        tournament_id=tournament_id,
        event_id=event_id,
        entries=entries,
        tournament_name=tournament_name
    )

    
@app.route("/player")
def player():
    name = request.args.get("name", "Kevin Nita").strip()
    age_group = request.args.get("age_group", "BS14")  # selected age group from query string

    if not name:
        # simple form on the same page
        return render_template("player.html", name=None, rows=[], age_groups=AGE_GROUPS)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT Week, AgeGroup, Rank, Player, [Ranking points], [Total points]
        FROM rankings
        WHERE Player = ?
        ORDER BY Week, AgeGroup
        """,
        (name,),
    )
    rows = cur.fetchall()
    conn.close()

    # --- Build weekly series for the selected age group ---
    weeks = []
    ranks = []
    if age_group:
        for r in rows:
            if r["AgeGroup"] == age_group and r["Rank"] is not None:
                weeks.append(r["Week"])
                ranks.append(int(r["Rank"]))

    def parse_week(w):
        try:
            week_num, year = w.split("-")
            return (int(year), int(week_num))
        except Exception:
            return (9999, 9999)

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
    print(f"\n=== Top 6 tournaments for {name} ({age_group}) ===")
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

        bar_fig = go.Figure(
            data=[go.Bar(
                x=points,
                y=list(range(1, len(points)+1)),
                orientation="h",
                marker=dict(color="green"),
                text=names,
                textposition="inside"
            )],
            layout=go.Layout(
                title=f"Top 6 Tournaments by Points ({age_group})",
                xaxis=dict(title="Points"),
                yaxis=dict(title="Tournaments", autorange="reversed"),  # ✅ flip so highest at bottom
                margin=dict(l=120, r=20, t=50, b=80),
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
        category_summary=tournament_data["category_summary"],  
        grand_total=tournament_data["grand_total"],
        cat_ranking_points=tournament_data["cat_ranking_points"],             
    )


# Add this near the bottom of app.py, before app.run
print("Registered routes:")
for rule in app.url_map.iter_rules():
    print(rule)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
