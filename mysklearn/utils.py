import json

def get_main_genre(genres_str):
    """
    Convert the TMDB 'genres' JSON-like string into one main genre.
    We only classify into Action / Comedy / Drama / Other.
    """
    if not isinstance(genres_str, str):
        return "Other"

    # Fix formatting (' → ")
    try:
        fixed = genres_str.replace("'", "\"")
        genres_list = json.loads(fixed)
    except:
        return "Other"

    # Extract names
    names = [g.get("name", "") for g in genres_list]

    # Priority classification
    if "Action" in names:
        return "Action"
    if "Comedy" in names:
        return "Comedy"
    if "Drama" in names:
        return "Drama"

    return "Other"
