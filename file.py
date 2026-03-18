import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Crowd Twin | Audio Matcher", page_icon="🎧", layout="wide")

# --- DATA LOADING & GLOBAL SCALING ---
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("/Users/shaguntembhurne/gdg_event/cleaned_dataset.csv")
    except FileNotFoundError:
        st.error("⚠️ 'spotify_dataset.csv' not found. Please ensure it's in the same folder.")
        return pd.DataFrame(), [], None

    # Standardize display names
    if 'Artist' in df.columns and 'Track' in df.columns:
        df['Display_Name'] = df['Artist'] + " - " + df['Track']
    else:
        df['Display_Name'] = df['Title']
        
    # The audio features we care about
    features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    df = df.dropna(subset=features)
    
    # INDUSTRY UPGRADE: Global Scaling
    # We must scale the entire dataset so Tempo (0-200) and Danceability (0-1) are on the same playing field
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, features, scaler

df, audio_features, scaler = load_and_prep_data()

# --- SESSION STATE (THE CROWD DATABASE) ---
if "crowd_data" not in st.session_state:
    st.session_state.crowd_data = []

# --- UI HEADER ---
st.title("🎧 Find Your Crowd Song Twin")
st.markdown("Build your sonic profile and our algorithm will pair you with your exact vibe match in the room.")
st.divider()

if df.empty:
    st.stop()

# --- MAIN LAYOUT ---
# Using columns makes the app look like a professional dashboard
col_input, col_results = st.columns([1, 1.5], gap="large")

with col_input:
    st.subheader("1. Profile Setup")
    user_name = st.text_input("Display Name:", placeholder="e.g., DJ Data")
    fav_artist = st.text_input("Who is your all-time favorite artist?", placeholder="e.g., Daft Punk")
    
    st.subheader("2. Build Your Vibe")
    st.markdown("Select up to **3 tracks** that define your current mood.")
    
    # INDUSTRY UPGRADE: Multiselect for a composite vibe
    selected_songs = st.multiselect(
        "Search tracks:", 
        df['Display_Name'].unique(),
        max_selections=3
    )
    
    find_match_btn = st.button("Initialize Match Sequence 🚀", type="primary", use_container_width=True)

# --- MATCHING ENGINE ---
with col_results:
    if find_match_btn:
        if not user_name or not selected_songs:
            st.warning("⚠️ Please enter your name and select at least one track to proceed.")
        else:
            with st.spinner("Analyzing audio features & querying crowd database..."):
                # 1. Create the User's "Composite Vibe Vector"
                # We get the scaled features for all selected songs and average them
                user_song_rows = df[df['Display_Name'].isin(selected_songs)]
                user_vector = user_song_rows[audio_features].mean().values.reshape(1, -1)
                
                # 2. Check Database Size
                if len(st.session_state.crowd_data) == 0:
                    st.success(f"**Profile Saved!** Welcome, {user_name}. You are patient zero. Let the next person step up to find a match!")
                else:
                    # 3. The Matching Algorithm
                    best_match = None
                    highest_score = -1
                    
                    for person in st.session_state.crowd_data:
                        # Don't match with yourself
                        if person['name'].lower() == user_name.lower():
                            continue
                            
                        crowd_vector = person['vector'].reshape(1, -1)
                        
                        # Calculate raw cosine similarity (0 to 1)
                        base_similarity = cosine_similarity(user_vector, crowd_vector)[0][0]
                        
                        # INDUSTRY UPGRADE: Personalization Bonus
                        # If they share a favorite artist, give a 5% similarity bump
                        artist_bonus = 0.05 if (fav_artist and person['artist'].lower() == fav_artist.lower()) else 0.0
                        
                        total_score = min(base_similarity + artist_bonus, 1.0) # Cap at 100%
                        
                        if total_score > highest_score:
                            highest_score = total_score
                            best_match = person
                    
                    # 4. Display Results
                    if best_match:
                        match_percentage = round(highest_score * 100, 1)
                        st.balloons()
                        
                        st.subheader(f"Target Acquired, {user_name}! 🎯")
                        
                        # Use metrics for a clean, professional look
                        m1, m2 = st.columns(2)
                        m1.metric(label="Your Crowd Twin", value=best_match['name'])
                        m2.metric(label="Vibe Alignment Score", value=f"{match_percentage}%")
                        
                        if fav_artist.lower() == best_match['artist'].lower() and fav_artist != "":
                            st.info(f"✨ **Bonus:** You both love {fav_artist.title()}!")
                        
                        # 5. Professional Radar Chart using Plotly
                        st.markdown("### Sonic DNA Comparison")
                        
                        fig = go.Figure()
                        
                        # Add User's Vibe
                        fig.add_trace(go.Scatterpolar(
                            r=user_vector[0],
                            theta=audio_features,
                            fill='toself',
                            name=user_name,
                            line_color='#1DB954' # Spotify Green
                        ))
                        
                        # Add Twin's Vibe
                        fig.add_trace(go.Scatterpolar(
                            r=best_match['vector'],
                            theta=audio_features,
                            fill='toself',
                            name=best_match['name'],
                            line_color='#FFFFFF'
                        ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=20, b=20, l=20, r=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("View their selected tracks"):
                            for song in best_match['songs']:
                                st.write(f"🎵 {song}")
                                
                    else:
                        st.info("You're the only one with this unique vibe so far! Let the next person try.")

            # 6. Save current user to database
            # We do this at the end so they don't match with themselves
            st.session_state.crowd_data.append({
                'name': user_name,
                'artist': fav_artist,
                'songs': selected_songs,
                'vector': user_vector[0]
            })

# --- SIDEBAR ADMIN ---
with st.sidebar:
    st.header("🎛️ Live Database")
    st.metric("Total Profiles", len(st.session_state.crowd_data))
    
    with st.expander("View Roster"):
        for p in st.session_state.crowd_data:
            st.markdown(f"**{p['name']}**")
            st.caption(f"Fav: {p['artist'] if p['artist'] else 'N/A'}")
    
    st.divider()
    if st.button("⚠️ Reset Database", type="secondary", use_container_width=True):
        st.session_state.crowd_data = []
        st.rerun()