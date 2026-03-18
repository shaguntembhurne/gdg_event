import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- 1. PAGE SETUP & CUSTOM CSS ---
st.set_page_config(page_title="Sonic Twin Matcher", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS to make it look less like Streamlit and more like a web app
st.markdown("""
<style>
    /* Hide default Streamlit elements for a cleaner showcase */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Typography and Spacing */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    /* Style the main call-to-action button */
    .stButton>button {
        border-radius: 30px;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("/Users/shaguntembhurne/gdg_event/cleaned_dataset.csv")
    except FileNotFoundError:
        st.error("⚠️ 'spotify_dataset.csv' is missing from the folder.")
        return pd.DataFrame(), [], None

    if 'Artist' in df.columns and 'Track' in df.columns:
        df['Display_Name'] = df['Artist'] + " - " + df['Track']
    else:
        df['Display_Name'] = df['Title']
        
    features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    df = df.dropna(subset=features)
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, features, scaler

df, audio_features, scaler = load_and_prep_data()

# --- 3. DATABASE INITIALIZATION ---
if "crowd_data" not in st.session_state:
    st.session_state.crowd_data = []

# --- 4. THE UI EXPEREINCE ---
st.markdown("<h1 class='hero-title'>Find Your Sonic Twin</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Pick your anthems. We'll analyze your audio signature and pair you with someone in the room.</p>", unsafe_allow_html=True)

if df.empty:
    st.stop()

# Layout: 40% for inputs, 60% for the immersive results
col_input, col_results = st.columns([4, 6], gap="large")

with col_input:
    st.markdown("### Step 1: Set the Vibe")
    # THE HOOK: Let them play with the music first
    selected_songs = st.multiselect(
        "Search for 1 to 3 tracks that define your current mood:", 
        df['Display_Name'].unique(),
        max_selections=3,
        help="Type an artist or song name"
    )
    
    st.markdown("### Step 2: Claim Your Profile")
    user_name = st.text_input("What should we call you?", placeholder="e.g., Alex, DJ Data, etc.")
    
    st.write("") # Spacer
    find_match_btn = st.button("Analyze & Find Match 🚀", type="primary", use_container_width=True)

# --- 5. THE MAGIC (RESULTS & LOGIC) ---
with col_results:
    if find_match_btn:
        if not selected_songs:
            st.warning("🎵 You gotta pick at least one track to set your vibe!")
        elif not user_name:
            st.warning("👋 Don't be shy, enter your name so your twin knows who you are!")
        else:
            with st.spinner("Decoding your audio signature..."):
                # Calculate User Vector
                user_song_rows = df[df['Display_Name'].isin(selected_songs)]
                user_vector = user_song_rows[audio_features].mean().values.reshape(1, -1)
                
                # --- VIBE ANALYSIS FEATURE (Adds personalization) ---
                st.markdown("### 🧬 Your Sonic DNA")
                vibe_traits = []
                # Looking at scaled values (0 to 1) to determine personality traits
                if user_vector[0][audio_features.index('Energy')] > 0.7:
                    vibe_traits.append("High Energy ⚡")
                if user_vector[0][audio_features.index('Acousticness')] > 0.6:
                    vibe_traits.append("Chill & Acoustic 🎸")
                if user_vector[0][audio_features.index('Danceability')] > 0.7:
                    vibe_traits.append("Ready to Dance 🪩")
                if user_vector[0][audio_features.index('Valence')] < 0.4:
                    vibe_traits.append("Moody & Deep 🌧️")
                elif user_vector[0][audio_features.index('Valence')] > 0.7:
                    vibe_traits.append("Uplifting & Happy ☀️")
                
                if not vibe_traits:
                    vibe_traits.append("Perfectly Balanced ☯️")
                    
                st.info(f"**Vibe Read:** {', '.join(vibe_traits)}")
                st.divider()
                
                # --- MATCHING ALGORITHM ---
                if len(st.session_state.crowd_data) == 0:
                    st.success(f"**Profile Locked!** Welcome to the grid, {user_name}. You are the very first person here. Tell the next person to step up so we can find your match!")
                else:
                    best_match = None
                    highest_score = -1
                    
                    for person in st.session_state.crowd_data:
                        if person['name'].lower() == user_name.lower():
                            continue
                            
                        crowd_vector = person['vector'].reshape(1, -1)
                        similarity = cosine_similarity(user_vector, crowd_vector)[0][0]
                        
                        if similarity > highest_score:
                            highest_score = similarity
                            best_match = person
                    
                    # --- DISPLAY RESULTS ---
                    if best_match:
                        match_percentage = round(highest_score * 100, 1)
                        st.balloons()
                        
                        st.markdown(f"### 🎉 We found them!")
                        
                        # Clean metric display
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Your Crowd Twin is", value=best_match['name'])
                        with col2:
                            st.metric(label="Vibe Alignment", value=f"{match_percentage}%")
                        
                        st.caption(f"**{best_match['name']}'s Anthem:** {best_match['songs'][0]}")
                        
                        # Radar Chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=user_vector[0],
                            theta=audio_features,
                            fill='toself',
                            name=user_name,
                            line_color='#1DB954'
                        ))
                        
                        fig.add_trace(go.Scatterpolar(
                            r=best_match['vector'],
                            theta=audio_features,
                            fill='toself',
                            name=best_match['name'],
                            line_color='#FF5722' # Contrast color for the twin
                        ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                            showlegend=True,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=30, b=30, l=30, r=30)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                                
                    else:
                        st.info("No matches found yet. Let the next person jump in!")

            # Add to DB after showing results
            st.session_state.crowd_data.append({
                'name': user_name,
                'songs': selected_songs,
                'vector': user_vector[0]
            })

# --- 6. HIDDEN ADMIN CONTROLS ---
# Using an expander at the bottom instead of a sidebar keeps the app looking like a single-page site
with st.expander("⚙️ Admin Console (Crowd Roster)"):
    st.write(f"Total Profiles Registered: {len(st.session_state.crowd_data)}")
    for p in st.session_state.crowd_data:
        st.write(f"• {p['name']}")
    if st.button("Reset Crowd Database", type="secondary"):
        st.session_state.crowd_data = []
        st.rerun()