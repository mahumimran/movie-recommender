"""
recommender.py
=================================================
MovieMind - Recommendation & TMDB Poster Logic
=================================================

Keeps all machine-learning / data-access logic separate from the
Streamlit UI layer (app.py). Designed to run safely in a cloud
environment (Railway): no hard-coded secrets, no local-only paths,
graceful handling of missing files / network failures.
"""

import os
import pickle

import pandas as pd
import requests
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT_SECONDS = 6

# A lightweight, self-contained placeholder poster encoded directly as an
# inline SVG data URI. This never depends on an external image host being
# reachable, so it always renders even with no internet / no API key.
PLACEHOLDER_POSTER_URL = (
    "data:image/svg+xml;utf8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='500' height='750'%3E"
    "%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E"
    "%3Cstop offset='0%25' stop-color='%231b1230'/%3E"
    "%3Cstop offset='100%25' stop-color='%23050508'/%3E"
    "%3C/linearGradient%3E%3C/defs%3E"
    "%3Crect width='500' height='750' fill='url(%23g)'/%3E"
    "%3Ctext x='50%25' y='46%25' font-family='sans-serif' font-size='64' "
    "text-anchor='middle' fill='%23e94560'%3E%F0%9F%8E%AC%3C/text%3E"
    "%3Ctext x='50%25' y='56%25' font-family='sans-serif' font-size='24' "
    "text-anchor='middle' fill='%23b8b8c8'%3EPoster Unavailable%3C/text%3E"
    "%3C/svg%3E"
)


class ModelLoadError(Exception):
    """Raised when the trained model files cannot be loaded."""


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the processed movie dataframe and similarity matrix from disk.
    Cached so the (potentially large) similarity matrix is loaded into
    memory only once per running instance.

    Returns:
        tuple(pd.DataFrame, np.ndarray)

    Raises:
        ModelLoadError: if the model files are missing or unreadable.
    """
    movie_dict_path = os.environ.get("MOVIE_DICT_PATH", os.path.join(MODELS_DIR, "movie_dict.pkl"))
    similarity_path = os.environ.get("SIMILARITY_PATH", os.path.join(MODELS_DIR, "similarity.pkl"))

    if not os.path.exists(movie_dict_path) or not os.path.exists(similarity_path):
        raise ModelLoadError(
            "Model files not found. Please run 'python preprocess.py' first "
            "to generate 'models/movie_dict.pkl' and 'models/similarity.pkl'."
        )

    try:
        with open(movie_dict_path, "rb") as f:
            movie_dict = pickle.load(f)
        with open(similarity_path, "rb") as f:
            similarity = pickle.load(f)
    except Exception as exc:  # noqa: BLE001 - surface a clean, safe message
        raise ModelLoadError(f"Failed to load model files: {exc}") from exc

    movies_df = pd.DataFrame(movie_dict)
    return movies_df, similarity

def get_movie_titles(movies_df: pd.DataFrame):
    """Return a sorted list of all available movie titles."""
    return sorted(movies_df["title"].dropna().unique().tolist())


def recommend_movies(movie_title: str, movies_df: pd.DataFrame, similarity, top_n: int = 5):
    """
    Return the top_n most similar movies to the given title.

    Args:
        movie_title: The title selected by the user.
        movies_df: The processed movie dataframe (movie_id, title, tags).
        similarity: The precomputed cosine similarity matrix.
        top_n: Number of recommendations to return (default 5).

    Returns:
        list[dict]: Each dict has keys 'title' and 'movie_id'.
        Returns an empty list if the title is not found.
    """
    matches = movies_df[movies_df["title"] == movie_title]
    if matches.empty:
        return []

    movie_index = matches.index[0]

    try:
        distances = list(enumerate(similarity[movie_index]))
    except IndexError:
        return []

    # Sort by similarity score, descending, skipping the movie itself (index 0 after sort)
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)

    recommendations = []
    for index, score in sorted_movies:
        if index == movie_index:
            continue
        row = movies_df.iloc[index]
        recommendations.append(
            {
                "title": row["title"],
                "movie_id": row["movie_id"],
                "score": round(float(score) * 100, 1),
            }
        )
        if len(recommendations) == top_n:
            break

    return recommendations


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_movie_poster(movie_id) -> str:
    """
    Fetch the poster image URL for a given TMDB movie_id.
    Cached for one hour per movie_id to reduce redundant API calls.

    Falls back to a placeholder image on any error (missing key,
    network timeout, invalid response, missing poster) so a single
    bad poster never breaks the whole page.
    """
    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    if not api_key:
        return PLACEHOLDER_POSTER_URL

    url = f"{TMDB_API_BASE_URL}/movie/{int(movie_id)}"
    params = {"api_key": api_key, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return PLACEHOLDER_POSTER_URL
        return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    except (requests.RequestException, ValueError):
        # Covers connection errors, timeouts, bad status codes, bad JSON.
        # Never leak the API key or raw exception details to the UI.
        return PLACEHOLDER_POSTER_URL
