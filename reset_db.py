import sqlite3
from config import DATABASE

# DB_PATH = "rankings.db"

conn = sqlite3.connect(DATABASE)
cur = conn.cursor()

# Drop the rankings table if it exists
cur.execute("DROP TABLE IF EXISTS rankings;")
conn.commit()
conn.close()

print("Dropped table 'rankings'.")
