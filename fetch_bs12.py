import requests
import pandas as pd

# TODO: paste your real cookie here (same as you use in Excel Power Query)
COOKIE = "ASP.NET_SessionId=xtar3zy11ooki31ubussmtoq; st=l=6153&exp=46353.8539954398&c=1&cp=23; _ga=GA1.1.1982321751.1764271543; _ga_FNL59NFQ13=GS2.1.s1764271543$o1$g1$t1764271582$j21$l0$h0"
CATEGORY_ID = "2068"  # BS12

BASE_URL = "https://ti.tournamentsoftware.com/ranking/category.aspx"

def fetch_bs12_current():
    params = {
        "rid": "169",          # "current ranking" indicator
        "category": CATEGORY_ID,
        "ps": "100",           # page size
        "p": "1"               # page 1 (we can loop pages later)
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": COOKIE
    }

    resp = requests.get(BASE_URL, params=params, headers=headers)
    resp.raise_for_status()

    # pandas tries to find all tables on the page
    tables = pd.read_html(resp.text)

    # from your Excel work, the main ranking table is usually tables[0]
    df = tables[0]

    return df

if __name__ == "__main__":
    df = fetch_bs12_current()
    print(df.head())
