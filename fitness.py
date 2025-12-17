import sqlite3
from flask import Blueprint, render_template, request, redirect, url_for
from datetime import datetime
from config import DATABASE

fitness_bp = Blueprint("fitness", __name__, url_prefix="/fitness")

# --- Shared DB connection ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# --- Ensure tables exist ---
def ensure_fitness_tables():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS fitness_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT NOT NULL,
            entry_type TEXT NOT NULL,
            value TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

# --- Categories and subcategories definition ---
CATEGORIES = {
    "REACTION": [("React ball", "repetitions")],
    "COORDINATION": [("FH", "repetitions"), ("BH", "repetitions")],
    "AGILITY": [("Spider Drill", "time")],
    "EXPLOSIVENESS": [("Height", "distance"), ("Length", "distance")],
    "BALANCE": [("L BOSU", "time"), ("R BOSU", "time")],
    "STRENGTH": [("Arm Hold", "time"), ("Legs Hold", "time"),
                 ("Plank", "time"), ("2kg Throw", "distance")],
    "SPEED": [("10m Sprint", "time"), ("20m Sprint", "time")],
    "ENDUR": [("1200m Run", "time")]
}

# --- Helper: parse time/distance/reps values ---
def parse_val(v: str) -> float:
    """
    Parse a string into seconds (float) or numeric value.
    Supports:
      - "M:SS" or "MM:SS" (minutes:seconds)
      - "M:SS.sss" or "MM:SS.sss" (minutes:seconds.milliseconds)
      - "S.sss" or "SS.sss" (seconds.milliseconds)
      - plain integer seconds "45"
    """
    v = str(v).strip().replace('"', '')

    if ":" in v:
        parts = v.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            raise ValueError(f"Unsupported time format: {v}")
    else:
        return float(v)

# --- Routes ---
@fitness_bp.route("/")
def index():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM fitness_entries ORDER BY date DESC").fetchall()
    conn.close()

    # Pivot rows into {date: {cat__sub: value}}
    grouped = {}
    for r in rows:
        d = r["date"].split(" ")[0]  # keep only YYYY-MM-DD
        key = f"{r['category']}__{r['subcategory']}"
        if d not in grouped:
            grouped[d] = {"date": d, "id": r["id"]}
        grouped[d][key] = r["value"]

    # Convert to list sorted by date descending
    pivoted_rows = sorted(grouped.values(), key=lambda x: x["date"], reverse=True)

    # Special list of "longer is better" time tests
    longer_is_better = {
        "BALANCE__L BOSU",
        "BALANCE__R BOSU",
        "STRENGTH__Arm Hold",
        "STRENGTH__Legs Hold",
        "STRENGTH__Plank",
    }

    # Compute best values per subcategory
    best_values = {}
    for cat, subs in CATEGORIES.items():
        for sub, entry_type in subs:
            key = f"{cat}__{sub}"
            values = [r.get(key) for r in pivoted_rows if r.get(key)]
            if values:
                if entry_type == "time":
                    if key in longer_is_better:
                        best_values[key] = max(values, key=parse_val)
                    else:
                        best_values[key] = min(values, key=parse_val)
                else:
                    best_values[key] = max(values, key=parse_val)

    return render_template("fitness.html",
                           categories=CATEGORIES,
                           rows=pivoted_rows,
                           now=datetime.now(),
                           best_values=best_values)


@fitness_bp.route("/add", methods=["POST"])
def add_entry():
    date = request.form.get("date") or datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    cur = conn.cursor()

    for cat, subs in CATEGORIES.items():
        for sub, entry_type in subs:
            key = f"{cat}__{sub}"
            value = request.form.get(key)
            if value:  # only insert if user entered something
                cur.execute(
                    "INSERT INTO fitness_entries (date, category, subcategory, entry_type, value) VALUES (?, ?, ?, ?, ?)",
                    (date, cat, sub, entry_type, value)
                )

    conn.commit()
    conn.close()
    return redirect(url_for("fitness.index"))


@fitness_bp.route("/delete/<date>")
def delete_entry(date):
    conn = get_db_connection()
    conn.execute("DELETE FROM fitness_entries WHERE date LIKE ?", (f"{date}%",))
    conn.commit()
    conn.close()
    return redirect(url_for("fitness.index"))
