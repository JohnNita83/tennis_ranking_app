import sqlite3

# Replace with the path to your database file
db_path = "mydatabase.db"

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# List all tables
print("Tables in database:")
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cur.fetchall()]
print(tables)

# For each table, list its columns
for table in tables:
    print(f"\nColumns in table '{table}':")
    cur.execute(f"PRAGMA table_info({table});")
    for row in cur.fetchall():
        print(f" - {row[1]} ({row[2]})")

conn.close()
