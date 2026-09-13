"""
app.py
=================================================
MovieMind - Content-Based Movie Recommendation System
=================================================

Streamlit front-end. All ML / data-access logic lives in recommender.py;
this file is only responsible for layout, state, and presentation.
"""
"""
app.py
=================================================
MovieMind - Content-Based Movie Recommendation System
=================================================

Streamlit front-end. All ML / data-access logic lives in recommender.py;
this file is only responsible for layout, state, and presentation.
"""
import os
import shutil
from huggingface_hub import hf_hub_download

MODEL_DIR = "models"
REPO_ID = "Mahi211/moviemind-artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)

for filename in ["movie_dict.pkl", "similarity.pkl"]:
    local_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(local_path):
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
        shutil.copy(downloaded, local_path)

import streamlit as st
from dotenv import load_dotenv

from recommender import (
    PLACEHOLDER_POSTER_URL,
    ModelLoadError,
    fetch_movie_poster,
    get_movie_titles,
    load_model,
    recommend_movies,
)
# Load variables from a local .env file if present (no-op on Railway,
# where TMDB_API_KEY is provided via Railway's Variables panel instead).
load_dotenv()

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieMind | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS - dark cinematic theme with animation
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Slowly drifting cinematic background gradient */
    @keyframes bgDrift {
        0%   { background-position: 0% 0%; }
        50%  { background-position: 100% 100%; }
        100% { background-position: 0% 0%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #1b1230, #0a0a12, #2a1245, #05050a);
        background-size: 400% 400%;
        animation: bgDrift 22s ease infinite;
        color: #eaeaf0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #120c1e 0%, #0a0a12 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* -------------------- Hero section -------------------- */
    @keyframes heroGlow {
        0%, 100% { box-shadow: 0 0 25px rgba(233, 69, 96, 0.15), 0 0 50px rgba(83, 52, 131, 0.10); }
        50%      { box-shadow: 0 0 45px rgba(233, 69, 96, 0.30), 0 0 80px rgba(83, 52, 131, 0.22); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes shimmerText {
        0%   { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    @keyframes reelSpin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }

    @keyframes floatReel {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50%      { transform: translateY(-8px) rotate(8deg); }
    }

    .hero-container {
        position: relative;
        padding: 3rem 1rem 2.2rem 1rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.12), rgba(83, 52, 131, 0.20));
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        overflow: hidden;
        animation: heroGlow 4.5s ease-in-out infinite, fadeInDown 0.7s ease;
    }

    .hero-reel {
        position: absolute;
        font-size: 2.4rem;
        opacity: 0.18;
        animation: floatReel 5s ease-in-out infinite;
    }
    .hero-reel.left  { top: 12%; left: 6%; animation-delay: 0s; }
    .hero-reel.right { top: 55%; right: 7%; animation-delay: 1.2s; }
    .hero-reel.top-right { top: 10%; right: 14%; font-size: 1.6rem; animation-delay: 2s; }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e94560, #ff9f68, #ffd97d, #e94560);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -1px;
        animation: shimmerText 5s linear infinite;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #b8b8c8;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }

    /* -------------------- Section headers -------------------- */
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-24px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f0f5;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #e94560;
        padding-left: 0.75rem;
        animation: slideInLeft 0.5s ease;
    }

    /* -------------------- Movie cards -------------------- */
    @keyframes cardRise {
        from { opacity: 0; transform: translateY(28px) scale(0.96); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    .movie-card {
        position: relative;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.75rem;
        text-align: center;
        transition: transform 0.35s cubic-bezier(.2,.8,.2,1), box-shadow 0.35s ease, border-color 0.35s ease;
        height: 100%;
        overflow: hidden;
        animation: cardRise 0.6s cubic-bezier(.2,.8,.2,1) both;
    }

    .movie-card:nth-child(1) { animation-delay: 0.05s; }
    .movie-card:nth-child(2) { animation-delay: 0.15s; }
    .movie-card:nth-child(3) { animation-delay: 0.25s; }
    .movie-card:nth-child(4) { animation-delay: 0.35s; }
    .movie-card:nth-child(5) { animation-delay: 0.45s; }

    .movie-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 16px 34px rgba(233, 69, 96, 0.3);
        border-color: rgba(233, 69, 96, 0.5);
    }

    .movie-poster-wrap {
        position: relative;
        overflow: hidden;
        border-radius: 10px;
        margin-bottom: 0.6rem;
    }

    .movie-card img {
        border-radius: 10px;
        width: 100%;
        display: block;
        transition: transform 0.5s ease, filter 0.5s ease;
    }

    .movie-card:hover img {
        transform: scale(1.12);
        filter: brightness(0.65);
    }

    .movie-poster-wrap::after {
        content: "▶ View Match";
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffd97d;
        font-weight: 600;
        font-size: 0.85rem;
        opacity: 0;
        transition: opacity 0.35s ease;
        background: rgba(5, 5, 10, 0.15);
        pointer-events: none;
    }

    .movie-card:hover .movie-poster-wrap::after {
        opacity: 1;
    }

    .movie-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f0f0f5;
        line-height: 1.3rem;
        min-height: 2.6rem;
    }

    .movie-score {
        font-size: 0.8rem;
        color: #ff9f68;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* -------------------- Buttons -------------------- */
    @keyframes buttonSheen {
        0%   { background-position: -150% 0; }
        100% { background-position: 250% 0; }
    }

    div.stButton > button {
        background: linear-gradient(90deg, #e94560, #c2364a, #e94560);
        background-size: 250% auto;
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: all 0.25s ease;
        animation: buttonSheen 6s linear infinite;
    }

    div.stButton > button:hover {
        box-shadow: 0 6px 24px rgba(233, 69, 96, 0.45);
        transform: translateY(-3px) scale(1.02);
    }

    div.stButton > button:active {
        transform: translateY(0) scale(0.99);
    }

    /* -------------------- Selectbox -------------------- */
    div[data-baseweb="select"] {
        border-radius: 10px;
        transition: box-shadow 0.3s ease;
    }

    div[data-baseweb="select"]:focus-within {
        box-shadow: 0 0 0 2px rgba(233, 69, 96, 0.5);
    }

    /* -------------------- Feature cards on Home -------------------- */
    div[data-testid="column"] {
        animation: cardRise 0.6s ease both;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helper render functions
# --------------------------------------------------------------------------
def render_hero():
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-reel left">🎞️</div>
            <div class="hero-reel right">🎞️</div>
            <div class="hero-reel top-right">🍿</div>
            <div class="hero-title">🎬 MovieMind</div>
            <div class="hero-subtitle">
                Discover movies you'll love based on what you already enjoy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home():
    st.markdown('<div class="section-header">Welcome to MovieMind</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🎯 Content-Based")
        st.write("Recommendations are built from each movie's overview, genres, keywords, cast, and director.")
    with col2:
        st.markdown("#### ⚡ Fast & Precomputed")
        st.write("Similarity scores are trained once and cached, so results appear instantly.")
    with col3:
        st.markdown("#### 🎨 Cinematic Experience")
        st.write("A clean, modern interface designed to feel like a real streaming product.")

    st.info("👉 Head over to the **Movie Recommender** section to get started.")


def render_recommender(movies_df, similarity, movie_titles):
    st.markdown('<div class="section-header">Find Your Next Favorite Movie</div>', unsafe_allow_html=True)

    if not movie_titles:
        st.warning("No movies are currently available. Please check back later.")
        return

    left, right = st.columns([3, 1])
    with left:
        selected_movie = st.selectbox(
            "Search or select a movie you like:",
            options=movie_titles,
            index=0,
            placeholder="Start typing a movie title...",
        )
    with right:
        st.write("")
        st.write("")
        recommend_clicked = st.button("🎬 Recommend Movies", use_container_width=True)

    show_scores = st.checkbox("Show similarity scores", value=False)

    if recommend_clicked:
        if not selected_movie:
            st.warning("Please select a movie first.")
            return

        with st.spinner(f"Finding movies similar to \u201c{selected_movie}\u201d..."):
            try:
                recommendations = recommend_movies(selected_movie, movies_df, similarity, top_n=5)
            except Exception:
                st.error(
                    "Something went wrong while generating recommendations. "
                    "Please try a different movie."
                )
                return

        if not recommendations:
            st.error(
                f"Sorry, we couldn't find recommendations for \u201c{selected_movie}\u201d. "
                "Please try selecting a different movie from the list."
            )
            return

        st.markdown('<div class="section-header">Recommended For You</div>', unsafe_allow_html=True)
        columns = st.columns(5)

        for col, movie in zip(columns, recommendations):
            with col:
                try:
                    poster_url = fetch_movie_poster(movie["movie_id"])
                except Exception:
                    poster_url = None

                if not poster_url:
                    poster_url = PLACEHOLDER_POSTER_URL

                score_html = (
                    f'<div class="movie-score">⭐ {movie["score"]}% match</div>' if show_scores else ""
                )

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-poster-wrap">
                            <img src="{poster_url}" alt="{movie['title']} poster" />
                        </div>
                        <div class="movie-title">{movie['title']}</div>
                        {score_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_about():
    st.markdown('<div class="section-header">About MovieMind</div>', unsafe_allow_html=True)
    st.write(
        """
        **MovieMind** is a content-based movie recommendation system built on the
        TMDB 5000 Movies dataset. Rather than relying on other users' ratings
        (collaborative filtering), it analyzes each movie's own content — its
        overview, genres, keywords, cast, and director — to find movies that
        are genuinely similar in substance.
        """
    )
    st.markdown("#### How it works")
    st.write(
        """
        1. **Feature Engineering** — Each movie's overview, genres, keywords,
           top cast, and director are combined into a single text "tag".
        2. **Vectorization** — `CountVectorizer` converts each movie's tags
           into a numerical vector based on word frequency.
        3. **Similarity Scoring** — `Cosine Similarity` measures how close
           two movies are in that vector space.
        4. **Recommendation** — The 5 closest movies (excluding the selected
           movie itself) are returned to the user.
        """
    )
    st.markdown("#### Tech Stack")
    st.write("Python · Pandas · NumPy · Scikit-learn · Streamlit · TMDB API · Railway")


# --------------------------------------------------------------------------
# Main application flow
# --------------------------------------------------------------------------
def main():
    render_hero()

    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio("Go to:", ["Home", "Movie Recommender", "About"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Built with Streamlit · Deployed on Railway")

    try:
        movies_df, similarity = load_model()
        movie_titles = get_movie_titles(movies_df)
        model_error = None
    except ModelLoadError as exc:
        movies_df, similarity, movie_titles = None, None, []
        model_error = str(exc)

    if model_error:
        st.error(f"⚠️ {model_error}")

    if page == "Home":
        render_home()
    elif page == "Movie Recommender":
        if model_error:
            st.warning("The recommendation model isn't available yet. See the message above.")
        else:
            render_recommender(movies_df, similarity, movie_titles)
    else:
        render_about()


if __name__ == "__main__":
    main()
