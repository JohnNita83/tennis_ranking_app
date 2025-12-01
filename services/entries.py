# services/entries.py
import requests
from bs4 import BeautifulSoup
from utils.names import normalize_name, log_normalization
from ranking_fetcher import load_cookie

def load_entries(tournament_id, event_id):
    url = f"https://ti.tournamentsoftware.com/sport/event.aspx?id={tournament_id}&event={event_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Cookie": load_cookie()}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    entries = []
    for row in soup.select("table.players tr"):
        cols = row.find_all("td")
        if not cols:
            continue
        name = cols[1].get_text(strip=True)  # adjust if your column index differs
        log_normalization("ENTRY", name)
        entries.append({
            "name": name,
            "match_key": normalize_name(name),
        })
    return entries
