import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- PAGE SETUP ---
st.set_page_config(page_title="Crowd Song Twin Matcher", page_icon="🎵", layout="centered")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Load your specific dataset
    try:
        df = pd.read_csv("/Users/shaguntembhurne/gdg_event/cleaned_dataset.csv")
    except FileNotFoundError:
        st.error("⚠️ Please ensure 'spotify_dataset.csv' is in the same directory as this script.")
        return pd.DataFrame()

    # Create a display column combining Artist and Track
    if 'Artist' in df.columns and 'Track' in df.columns:
        df['Display_Name'] = df['Artist'] + " - " + df['Track']
    else:
        df['Display_Name'] = df['Title'] # Fallback if columns differ slightly
        
    # Drop rows missing crucial audio features to prevent errors
    features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    df = df.dropna(subset=features)
    
    return df, features

df, audio_features = load_data()

# --- SESSION STATE INITIALIZATION ---
# This acts as our temporary database for the crowd
if "crowd_data" not in st.session_state:
    st.session_state.crowd_data = []

# --- APP UI ---
st.title("🎧 Find Your Crowd Song Twin!")
st.markdown("Enter your name, pick your anthem, and we'll match you with someone in the room who shares your sonic vibe.")

if df.empty:
    st.stop()

# --- USER INPUT ---
with st.container():
    st.subheader("1. Who are you?")
    user_name = st.text_input("Enter your name or nickname:", placeholder="e.g., DJ Data")
    
    st.subheader("2. Pick your song")
    # Streamlit's selectbox is searchable, which is perfect for a crowd finding their song
    selected_song = st.selectbox("Search for a track from the dataset:", df['Display_Name'].unique())

# --- MATCHING LOGIC ---
if st.button("Find My Twin! 👯‍♂️", type="primary"):
    if not user_name:
        st.warning("Please enter your name first!")
    else:
        # 1. Get the features of the selected song
        song_row = df[df['Display_Name'] == selected_song].iloc[0]
        user_vector = song_row[audio_features].values.reshape(1, -1)
        
        # 2. Check if we have enough people in the crowd to make a match
        if len(st.session_state.crowd_data) == 0:
            st.success(f"Welcome to the party, {user_name}! You're the first one here. Hang tight, your twin is coming soon.")
        else:
            # 3. Calculate similarity against the crowd
            best_match_name = None
            best_match_song = None
            highest_score = -1
            
            # Normalize vectors for fair comparison using Min-Max scaling logic
            scaler = MinMaxScaler()
            
            for person in st.session_state.crowd_data:
                # Prevent matching with yourself if you try twice
                if person['name'] == user_name:
                    continue
                    
                crowd_vector = person['vector'].reshape(1, -1)
                
                # Stack, scale, and compare
                stacked = np.vstack((user_vector, crowd_vector))
                scaled = scaler.fit_transform(stacked)
                
                # Cosine similarity returns a matrix; we want the similarity between item 0 and 1
                similarity = cosine_similarity(scaled[0:1], scaled[1:2])[0][0]
                
                if similarity > highest_score:
                    highest_score = similarity
                    best_match_name = person['name']
                    best_match_song = person['song']
            
            # 4. Display the Result
            if best_match_name:
                match_percentage = round(highest_score * 100, 1)
                st.divider()
                st.snow() # Fun visual effect for the crowd
                st.header(f"🎉 We found a match, {user_name}!")
                st.subheader(f"Your Song Twin is **{best_match_name}**")
                st.markdown(f"**Their track:** {best_match_song}")
                st.metric(label="Vibe Match Score", value=f"{match_percentage}%")
                
                # Optional: Show a bar chart comparing their vibes
                st.markdown("### Vibe Comparison")
                crowd_row = df[df['Display_Name'] == best_match_song].iloc[0]
                comparison_df = pd.DataFrame({
                    user_name: song_row[audio_features].values,
                    best_match_name: crowd_row[audio_features].values
                }, index=audio_features)
                st.bar_chart(comparison_df)
            else:
                st.info("You're the only one with this unique vibe so far! Let's get more people in the system.")

        # 5. Add the current user to the crowd database AFTER checking for matches
        # This ensures they don't match with themselves
        st.session_state.crowd_data.append({
            'name': user_name,
            'song': selected_song,
            'vector': user_vector[0]
        })

# --- SIDEBAR ---
# Show who is currently in the "database"
with st.sidebar:
    st.header("👥 Crowd Roster")
    st.write(f"Total people registered: {len(st.session_state.crowd_data)}")
    for p in st.session_state.crowd_data:
        st.caption(f"• {p['name']} ({p['song'].split(' - ')[0]})")
    
    if st.button("Clear Crowd Data"):
        st.session_state.crowd_data = []
        st.rerun()