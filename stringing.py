from flask import Blueprint, render_template, request, redirect, flash, url_for
import sqlite3

DB_PATH = "rankings.db"

# --- Shared DB connection ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Ensure tables exist ---
def ensure_stringing_tables():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS rackets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            make_model TEXT NOT NULL,
            serial TEXT,
            FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS stringing_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            racket_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            mains TEXT,
            m_gauge TEXT,
            m_tension REAL,
            crosses TEXT,
            c_gauge TEXT,
            c_tension REAL,
            FOREIGN KEY (racket_id) REFERENCES rackets(id) ON DELETE CASCADE
        );
    """)
    try:
        cur.execute("ALTER TABLE stringing_records ADD COLUMN favorite INTEGER DEFAULT 0;")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    conn.commit()
    conn.close()

# --- Blueprint setup ---
stringing_bp = Blueprint("stringing", __name__)

# --- Routes ---
@stringing_bp.route("/stringing")
def stringing():
    ensure_stringing_tables()
    player_id = request.args.get("player_id", type=int)
    racket_id = request.args.get("racket_id", type=int)

    conn = get_db_connection()
    players = conn.execute("SELECT * FROM players ORDER BY name ASC").fetchall()

    selected_player = None
    rackets = []
    selected_racket = None

    # ✅ Default to Kevin Nita if no player_id is provided
    if player_id is None:
        kevin = conn.execute("SELECT id FROM players WHERE name = ?", ("Kevin Nita",)).fetchone()
        if kevin:
            player_id = kevin["id"]

    if player_id:
        selected_player = conn.execute(
            "SELECT * FROM players WHERE id=?", (player_id,)
        ).fetchone()

        if selected_player:
            rackets = conn.execute(
                "SELECT * FROM rackets WHERE player_id=? ORDER BY id DESC", (player_id,)
            ).fetchall()

            # ✅ Default to latest racket if none provided
            if racket_id is None and rackets:
                racket_id = rackets[0]["id"]

            # Fetch racket + its records
            if racket_id:
                racket = conn.execute(
                    "SELECT * FROM rackets WHERE id=?", (racket_id,)
                ).fetchone()
                if racket:
                    records = conn.execute(
                        "SELECT * FROM stringing_records WHERE racket_id=? ORDER BY date DESC",
                        (racket_id,)
                    ).fetchall()
                    racket_dict = dict(racket)
                    racket_dict["stringing_records"] = records
                    selected_racket = racket_dict

    conn.close()

    return render_template(
        "stringing.html",
        players=players,
        selected_player=selected_player,
        selected_player_id=player_id,
        rackets=rackets,
        selected_racket=selected_racket,
        selected_racket_id=racket_id,
    )



@stringing_bp.route("/add_player", methods=["POST"])
def add_player():
    name = request.form["name"]
    conn = get_db_connection()
    conn.execute("INSERT INTO players (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()
    return redirect(url_for("stringing.stringing"))

@stringing_bp.route("/delete_player/<int:player_id>", methods=["POST"])
def delete_player(player_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM players WHERE id=?", (player_id,))
    conn.execute("DELETE FROM rackets WHERE player_id=?", (player_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("stringing.stringing"))

@stringing_bp.route("/add_racket/<int:player_id>", methods=["POST"])
def add_racket(player_id):
    make_model = request.form["make_model"]
    serial = request.form.get("serial")
    conn = get_db_connection()
    conn.execute("INSERT INTO rackets (player_id, make_model, serial) VALUES (?, ?, ?)", (player_id, make_model, serial))
    conn.commit()
    conn.close()
    return redirect(url_for("stringing.stringing", player_id=player_id))

@stringing_bp.route("/delete_racket/<int:racket_id>", methods=["POST"])
def delete_racket(racket_id):
    conn = get_db_connection()
    # find the player before deleting, so we can redirect back
    player = conn.execute("SELECT player_id FROM rackets WHERE id=?", (racket_id,)).fetchone()
    conn.execute("DELETE FROM rackets WHERE id=?", (racket_id,))
    conn.commit()
    conn.close()
    player_id = player["player_id"] if player else None
    return redirect(url_for("stringing.stringing", player_id=player_id))

@stringing_bp.route("/edit_racket/<int:racket_id>", methods=["POST"])
def edit_racket(racket_id):
    make_model = request.form.get("make_model")
    serial = request.form.get("serial")

    conn = get_db_connection()
    cur = conn.cursor()

    # Update racket
    cur.execute(
        "UPDATE rackets SET make_model=?, serial=? WHERE id=?",
        (make_model, serial, racket_id)
    )
    conn.commit()

    # ✅ Fetch the player_id for this racket
    cur.execute("SELECT player_id FROM rackets WHERE id=?", (racket_id,))
    row = cur.fetchone()
    conn.close()

    player_id = row["player_id"] if row else None

    flash("Racket updated successfully!", "success")

    # Redirect back with both player_id and racket_id
    return redirect(url_for("stringing.stringing",
                            player_id=player_id,
                            racket_id=racket_id))



@stringing_bp.route("/add_stringing/<int:racket_id>", methods=["POST"])
def add_stringing(racket_id):
    date = request.form["date"]
    mains = request.form["mains"]
    m_gauge = request.form["m_gauge"]
    m_tension = request.form["m_tension"]
    crosses = request.form["crosses"]
    c_gauge = request.form["c_gauge"]
    c_tension = request.form["c_tension"]

    conn = get_db_connection()
    conn.execute(
        """INSERT INTO stringing_records
        (racket_id, date, mains, m_gauge, m_tension, crosses, c_gauge, c_tension)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (racket_id, date, mains, m_gauge, m_tension, crosses, c_gauge, c_tension)
    )
    conn.commit()

    racket_player = conn.execute(
        "SELECT player_id FROM rackets WHERE id=?", (racket_id,)
    ).fetchone()
    conn.close()

    # Redirect back to the stringing page, preserving racket selection
    return redirect(
        url_for("stringing.stringing", player_id=racket_player["player_id"], racket_id=racket_id)
    )


@stringing_bp.route("/delete_stringing/<int:record_id>", methods=["POST"])
def delete_stringing(record_id):
    conn = get_db_connection()
    racket = conn.execute(
        "SELECT racket_id FROM stringing_records WHERE id=?", (record_id,)
    ).fetchone()

    conn.execute("DELETE FROM stringing_records WHERE id=?", (record_id,))
    conn.commit()

    player_id = None
    racket_id = None
    if racket:
        racket_id = racket["racket_id"]
        player_id = conn.execute(
            "SELECT player_id FROM rackets WHERE id=?", (racket_id,)
        ).fetchone()["player_id"]

    conn.close()

    # Redirect back to the stringing page, preserving racket selection
    return redirect(
        url_for("stringing.stringing", player_id=player_id, racket_id=racket_id)
    )

@stringing_bp.route("/toggle_favorite/<int:record_id>", methods=["POST"])
def toggle_favorite(record_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get current favorite state and the racket_id
    cur.execute("SELECT favorite, racket_id FROM stringing_records WHERE id=?", (record_id,))
    row = cur.fetchone()

    if row:
        new_value = 0 if row["favorite"] else 1
        cur.execute("UPDATE stringing_records SET favorite=? WHERE id=?", (new_value, record_id))
        conn.commit()

        racket_id = row["racket_id"]

        # ✅ Now fetch the player_id from the rackets table
        cur.execute("SELECT player_id FROM rackets WHERE id=?", (racket_id,))
        racket_row = cur.fetchone()
        player_id = racket_row["player_id"] if racket_row else None
    else:
        racket_id = None
        player_id = None

    conn.close()

    flash("Favourite updated!", "info")
    return redirect(url_for("stringing.stringing", player_id=player_id, racket_id=racket_id))


