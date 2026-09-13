"""
preprocess.py
=================================================
MovieMind - Data Preprocessing & Model Training Script
=================================================

Loads the TMDB 5000 Movies and TMDB 5000 Credits datasets, cleans and
merges them, engineers a combined "tags" feature for each movie, and
trains a content-based recommendation model using CountVectorizer +
Cosine Similarity.

Run with:
    python preprocess.py

Outputs:
    models/movie_dict.pkl
    models/similarity.pkl
"""

import ast
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Paths (all relative to this file so the script works locally and on
# Railway regardless of the current working directory it's invoked from)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MOVIES_CSV = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

MOVIE_DICT_PATH = os.path.join(MODELS_DIR, "movie_dict.pkl")
SIMILARITY_PATH = os.path.join(MODELS_DIR, "similarity.pkl")

MAX_FEATURES = 5000
TOP_CAST_COUNT = 3


def log(step: str, message: str) -> None:
    """Print a clearly formatted progress message."""
    print(f"[{step}] {message}")


def load_datasets() -> pd.DataFrame:
    """Load and merge the movies and credits datasets."""
    if not os.path.exists(MOVIES_CSV):
        sys.exit(
            f"ERROR: Could not find '{MOVIES_CSV}'.\n"
            "Please download the TMDB 5000 Movies dataset and place "
            "'tmdb_5000_movies.csv' inside the 'data/' folder."
        )
    if not os.path.exists(CREDITS_CSV):
        sys.exit(
            f"ERROR: Could not find '{CREDITS_CSV}'.\n"
            "Please download the TMDB 5000 Credits dataset and place "
            "'tmdb_5000_credits.csv' inside the 'data/' folder."
        )

    log("1/13", "Loading tmdb_5000_movies.csv ...")
    movies = pd.read_csv(MOVIES_CSV)

    log("1/13", "Loading tmdb_5000_credits.csv ...")
    credits = pd.read_csv(CREDITS_CSV)

    log("2/13", f"Merging datasets on 'title' ({len(movies)} movies, {len(credits)} credit records) ...")
    credits = credits.rename(columns={"title": "title_credits"})
    merged = movies.merge(credits, left_on="title", right_on="title_credits", how="left")

    return merged


def safe_literal_eval(value):
    """Safely parse a JSON-like string column, returning [] on failure."""
    if not isinstance(value, str):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def extract_names(obj_list):
    """Extract the 'name' field from a list of dictionaries."""
    return [item.get("name", "") for item in obj_list if isinstance(item, dict)]


def extract_top_cast(cast_json, top_n=TOP_CAST_COUNT):
    """Extract the top N cast member names from the JSON-style cast column."""
    cast_list = safe_literal_eval(cast_json)
    names = extract_names(cast_list)
    return names[:top_n]


def extract_director(crew_json):
    """Extract the director's name from the JSON-style crew column."""
    crew_list = safe_literal_eval(crew_json)
    for member in crew_list:
        if isinstance(member, dict) and member.get("job") == "Director":
            return [member.get("name", "")]
    return []


def collapse_spaces(tokens):
    """
    Remove internal spaces from multi-word tokens (e.g. names) so that
    "Sam Worthington" becomes "SamWorthington". This keeps the vectorizer
    from confusing 'Sam' the first name with unrelated words.
    """
    return [str(token).replace(" ", "") for token in tokens]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the required columns and clean/normalize the data."""

    log("3/13", "Selecting relevant columns ...")
    df = df[["id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()

    log("4/13", "Handling missing values ...")
    df.dropna(subset=["title"], inplace=True)
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("[]")
    df["keywords"] = df["keywords"].fillna("[]")
    df["cast"] = df["cast"].fillna("[]")
    df["crew"] = df["crew"].fillna("[]")
    df.drop_duplicates(subset=["title"], inplace=True)

    log("5/13", "Extracting genres ...")
    df["genres_list"] = df["genres"].apply(lambda x: collapse_spaces(extract_names(safe_literal_eval(x))))

    log("6/13", "Extracting keywords ...")
    df["keywords_list"] = df["keywords"].apply(lambda x: collapse_spaces(extract_names(safe_literal_eval(x))))

    log("7/13", "Extracting top 3 cast members ...")
    df["cast_list"] = df["cast"].apply(lambda x: collapse_spaces(extract_top_cast(x)))

    log("8/13", "Extracting director ...")
    df["director_list"] = df["crew"].apply(lambda x: collapse_spaces(extract_director(x)))

    log("9/13", "Building combined 'tags' column (overview + genres + keywords + cast + director) ...")
    df["overview_tokens"] = df["overview"].apply(lambda x: str(x).split())

    df["tags_list"] = (
        df["overview_tokens"]
        + df["genres_list"]
        + df["keywords_list"]
        + df["cast_list"]
        + df["director_list"]
    )

    df["tags"] = df["tags_list"].apply(lambda tokens: " ".join(tokens).lower().strip())
    df["tags"] = df["tags"].apply(lambda text: " ".join(text.split()))  # remove extra whitespace

    df.rename(columns={"id": "movie_id"}, inplace=True)
    final_df = df[["movie_id", "title", "tags"]].reset_index(drop=True)

    # Guard against any rows that ended up with empty tags after cleaning
    final_df = final_df[final_df["tags"].str.strip() != ""].reset_index(drop=True)

    return final_df


def build_model(final_df: pd.DataFrame):
    """Vectorize the tags and compute the cosine similarity matrix."""

    log("10/13", f"Vectorizing tags with CountVectorizer(max_features={MAX_FEATURES}, stop_words='english') ...")
    cv = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
    vectors = cv.fit_transform(final_df["tags"]).toarray()

    log("11/13", "Calculating cosine similarity matrix ...")
    similarity = cosine_similarity(vectors)

    return similarity


def save_artifacts(final_df: pd.DataFrame, similarity: np.ndarray) -> None:
    """Persist the processed movie dataframe (as a dict) and similarity matrix."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    log("12/13", f"Saving movie dictionary to {MOVIE_DICT_PATH} ...")
    with open(MOVIE_DICT_PATH, "wb") as f:
        pickle.dump(final_df.to_dict(), f)

    log("13/13", f"Saving similarity matrix to {SIMILARITY_PATH} ...")
    with open(SIMILARITY_PATH, "wb") as f:
        pickle.dump(similarity, f)


def main():
    print("=" * 60)
    print("MovieMind Preprocessing Pipeline")
    print("=" * 60)

    merged = load_datasets()
    final_df = clean_dataframe(merged)
    log("INFO", f"Final processed dataset contains {len(final_df)} movies.")

    similarity = build_model(final_df)
    save_artifacts(final_df, similarity)

    print("=" * 60)
    print("Preprocessing complete! Model files are ready in 'models/'.")
    print("You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
