import sqlite3

conn = sqlite3.connect("rankings.db")  # replace with your DB filename
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== BS14 tournaments ===")
cur.execute("""
    SELECT id, player_name, age_group, tournament_name, category_code, place, won, lost, start_date, end_date
    FROM tournaments
    WHERE age_group = 'BS14'
""")
for row in cur.fetchall():
    print(dict(row))

print("\n=== BS12 tournaments ===")
cur.execute("""
    SELECT id, player_name, age_group, tournament_name, category_code, place, won, lost, start_date, end_date
    FROM tournaments
    WHERE age_group = 'BS12'
""")
for row in cur.fetchall():
    print(dict(row))

conn.close()
