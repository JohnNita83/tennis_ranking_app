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
    "REACTION": [("React lights", "Reps (3)")],
    "COORDINATION": [("FH", "Reps (3)"), ("BH", "Reps (3)")],
    "AGILITY": [("Spider Drill", "Time (2)")],
    "EXPLOSIVENESS": [("Height", "Dist (3)"), ("Length", "Dist (3)")],
    "BALANCE": [("L BOSU", "Time (1)"), ("R BOSU", "Time (1)")],
    "STRENGTH": [("Arm Hold", "Time (1)"), ("Leg Hops", "Reps (1)"),
                 ("Plank", "Time (1)"), ("2kg Throw", "Dist (3)")],
    "SPEED": [("10m Sprint", "Time (2)"), ("20m Sprint", "Time (2)")],
    "ENDUR": [("800m Run", "Time (1)")]
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
        raw_date = r["date"].split(" ")[0]  # keep only YYYY-MM-DD

        # Convert to pretty format
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        pretty_date = dt.strftime("%d/%m/%Y")

        key = f"{r['category']}__{r['subcategory']}"

        if raw_date not in grouped:
            grouped[raw_date] = {
                "date": pretty_date,   # formatted for display
                "raw_date": raw_date,  # keep original for editing/deleting
                "id": r["id"]
            }

        grouped[raw_date][key] = r["value"]

    # Convert to list sorted by date descending
    pivoted_rows = sorted(grouped.values(), key=lambda x: x["date"], reverse=True)

    # Special list of "longer is better" time tests
    longer_is_better = {
        "BALANCE__L BOSU",
        "BALANCE__R BOSU",
        "STRENGTH__Arm Hold",
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


@fitness_bp.route("/update", methods=["POST"])
def update_entry():
    date = request.form["date"]
    category = request.form["category"]
    subcategory = request.form["subcategory"]
    value = request.form["value"]

    conn = get_db_connection()
    cur = conn.cursor()

    # Delete existing entry for that category/subcategory/date
    cur.execute("""
        DELETE FROM fitness_entries
        WHERE date LIKE ? AND category = ? AND subcategory = ?
    """, (f"{date}%", category, subcategory))

    # Insert new value
    if value.strip():
        cur.execute("""
            INSERT INTO fitness_entries (date, category, subcategory, entry_type, value)
            VALUES (?, ?, ?, ?, ?)
        """, (date, category, subcategory, "time", value))  # entry_type preserved if needed

    conn.commit()
    conn.close()

    return "OK"



@fitness_bp.route("/delete/<date>")
def delete_entry(date):
    conn = get_db_connection()
    conn.execute("DELETE FROM fitness_entries WHERE date LIKE ?", (f"{date}%",))
    conn.commit()
    conn.close()
    return redirect(url_for("fitness.index"))
