import sqlite3
from config import DATABASE

# DB_PATH = "rankings.db"

conn = sqlite3.connect(DATABASE)
cur = conn.cursor()

# Ensure points table exists
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

# Insert Place = 0 if missing
cur.execute("SELECT COUNT(*) FROM points WHERE Place = 0;")
exists = cur.fetchone()[0]

if exists == 0:
    cur.execute("INSERT INTO points (Place) VALUES (0);")
    print("Inserted Place = 0 into points table.")
else:
    print("Place = 0 already exists in points table.")

conn.commit()
conn.close()
