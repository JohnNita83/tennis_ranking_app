# services/match.py
def debug_matches(entries_df, rankings_df):
    entry_keys = set(entries_df["player_norm"])
    rank_keys = set(rankings_df["name_norm"])

    # Which entries don’t have a match in rankings
    unmatched = entry_keys - rank_keys
    if unmatched:
        print("Unmatched entries (normalized):")
        for key in unmatched:
            raw = entries_df.loc[entries_df["player_norm"] == key, "player"].tolist()
            print(f"  entry_raw={raw} | entry_norm={key}")

    # Which rankings don’t have a match in entries
    extra = rank_keys - entry_keys
    if extra:
        print("Unmatched rankings (normalized):")
        for key in extra:
            raw = rankings_df.loc[rankings_df["name_norm"] == key, "name"].tolist()
            print(f"  rank_raw={raw} | rank_norm={key}")
# services/match.py

def match_entries_to_rankings(entries, rankings):
    """
    Match entries to rankings by their normalized match_key.
    entries: list of dicts with 'name' and 'match_key'
    rankings: list of dicts with 'name', 'points', and 'match_key'
    Returns (matched, unmatched)
    """
    by_key = {r["match_key"]: r for r in rankings}
    matched = []
    unmatched = []

    for e in entries:
        r = by_key.get(e["match_key"])
        if r:
            matched.append({"entry": e, "ranking": r})
        else:
            unmatched.append(e)

    return matched, unmatched
