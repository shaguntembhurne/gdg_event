import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Song Twin Generator", layout="centered")

st.title("Find Your Song Twin")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/shaguntembhurne/gdg_event/cleaned_dataset.csv")

    # Keep only required columns
    df = df[[
        "Artist", "Track",
        "Danceability", "Energy", "Loudness",
        "Speechiness", "Acousticness",
        "Instrumentalness", "Liveness",
        "Valence", "Tempo"
    ]]

    # Remove missing values
    df = df.dropna()

    return df

df = load_data()

# -----------------------------
# FEATURE SET
# -----------------------------
features = [
    "Danceability", "Energy", "Loudness",
    "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness",
    "Valence", "Tempo"
]

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Create Your Vibe")

name = st.text_input("Your Name")

dance = st.slider("Dance Energy", 0.0, 1.0, 0.5)
energy = st.slider("Energy Level", 0.0, 1.0, 0.5)
mood = st.slider("Mood (Happy ↔ Sad)", 0.0, 1.0, 0.5)
# -----------------------------
# MORE USER INPUT
# -----------------------------
acoustic = st.slider("Acoustic Feel", 0.0, 1.0, 0.5)
live = st.slider("Live Concert Feel", 0.0, 1.0, 0.3)
tempo = st.slider("Tempo (Speed)", 60.0, 200.0, 120.0)

# Map inputs to feature vector
user_input = np.array([[
    dance,          # Danceability
    energy,         # Energy
    -5.0,           # Loudness (fixed avg)
    0.05,           # Speechiness (fixed)
    acoustic,       # Acousticness
    0.0,            # Instrumentalness (fixed)
    live,           # Liveness
    mood,           # Valence
    tempo           # Tempo
]])

# Scale user input
user_scaled = scaler.transform(user_input)

# -----------------------------
# FIND SIMILAR SONG
# -----------------------------
if st.button("Find My Song Twin"):

    similarity = cosine_similarity(user_scaled, X)[0]

    # Get best match
    idx = np.argmax(similarity)

    song = df.iloc[idx]

    st.markdown("---")

    st.subheader(f"{name}'s Song Twin")

    st.write(f"**Track:** {song['Track']}")
    st.write(f"**Artist:** {song['Artist']}")

    st.progress(float(similarity[idx]))

    st.write("Match Score:", round(similarity[idx], 2))

    # -----------------------------
    # BONUS: TOP 5 SONGS
    # -----------------------------
    st.subheader("Top Matches")

    top_idx = np.argsort(similarity)[-5:][::-1]

    for i in top_idx:
        st.write(f"{df.iloc[i]['Track']} — {df.iloc[i]['Artist']}")