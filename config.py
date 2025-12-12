import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If DATABASE_URL is set (prod), use it. Otherwise default to local app.db.
DATABASE = os.getenv("DATABASE_URL", os.path.join(BASE_DIR, "rankings.db"))
