import sqlite3

DB_PATH = "rankings.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Drop the rankings table if it exists
cur.execute("DROP TABLE IF EXISTS rankings;")
conn.commit()
conn.close()

print("Dropped table 'rankings'.")
