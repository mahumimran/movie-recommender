#!/bin/bash
# start.sh
# =================================================
# MovieMind - Railway/production startup script
# =================================================
# Model files (models/*.pkl) are intentionally NOT committed to GitHub
# because similarity.pkl can be 100MB+ (over GitHub's 100MB file limit).
# Instead, this script trains the model once at startup if the files
# are missing, then launches Streamlit. On later restarts, if the
# files already exist (e.g. on a persistent volume), training is skipped.

set -e

if [ ! -f "models/movie_dict.pkl" ] || [ ! -f "models/similarity.pkl" ]; then
    echo "Model files not found. Running preprocessing (this may take a minute)..."
    python preprocess.py
else
    echo "Model files already exist. Skipping preprocessing."
fi

echo "Starting Streamlit on port ${PORT:-8501} ..."
streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
