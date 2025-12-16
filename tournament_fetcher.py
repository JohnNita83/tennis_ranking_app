import requests
from bs4 import BeautifulSoup
from datetime import datetime
from ranking_fetcher import load_cookie

BASE_TOURNAMENT_URL = "https://ti.tournamentsoftware.com/tournament"


def parse_iso_or_ddmmyyyy(text: str) -> str | None:
    """Try to parse ISO datetime or dd/mm/yyyy formats, return ISO date string."""
    text = text.strip()
    if not text:
        return None
    # First try ISO
    try:
        d = datetime.fromisoformat(text).date()
        return d.isoformat()
    except Exception:
        pass
    # Then try dd/mm/yyyy or dd-mm-yyyy
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            d = datetime.strptime(text, fmt).date()
            return d.isoformat()
        except ValueError:
            continue
    return None


def fetch_tournament_details(tournament_id: str) -> dict:
    """
    Fetch tournament details from TI tournament page.
    Returns dict with: name, start_date, end_date, opens_date, close_date
    Any field may be None/'' if not found.
    """
    url = f"{BASE_TOURNAMENT_URL}/{tournament_id}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": load_cookie(),
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # ---- Tournament name
    name = ""
    h2 = soup.find("h2", class_="media__title media__title--large")
    if h2:
        name = h2.get("title") or h2.get_text(strip=True)

    # ---- Dates from <li> elements with specific classes
    def extract_date(li_class: str) -> str | None:
        li = soup.find("li", class_=li_class)
        if li:
            time_tag = li.find("time")
            if time_tag:
                dt_val = time_tag.get("datetime") or time_tag.get_text(strip=True)
                return parse_iso_or_ddmmyyyy(dt_val) or dt_val
        return None

    start_date = extract_date("is-started")
    end_date = extract_date("is-finished")
    opens_date = extract_date("is-entry-open")
    close_date = extract_date("is-entry-closed")

    return {
        "name": name,
        "start_date": start_date,
        "end_date": end_date,
        "opens_date": opens_date,
        "close_date": close_date,
    }
