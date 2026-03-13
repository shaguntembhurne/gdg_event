import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Few-Shot Object Recognition",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Few-Shot Object Recognition Demo")

st.write(
"""
Teach the system a new object using a few photos and it will recognize it later.
Works directly in the browser.
"""
)

# -----------------------------
# LOAD FEATURE MODEL
# -----------------------------

@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    return model

model = load_model()

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------

def get_embedding(image):

    image = image.resize((224,224))

    img = np.array(image)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    emb = model.predict(img, verbose=0)

    return emb

# -----------------------------
# DATABASE
# -----------------------------

def load_db():
    try:
        return pickle.load(open("objects.pkl","rb"))
    except:
        return {}

def save_db(db):
    pickle.dump(db, open("objects.pkl","wb"))

db = load_db()

# -----------------------------
# SIDEBAR
# -----------------------------

mode = st.sidebar.radio(
    "Select Mode",
    ["Add Object","Recognize","Object Database"]
)

st.sidebar.info(
"""
How to use

1️⃣ Add object  
2️⃣ Take 3-5 pictures  
3️⃣ Try recognition
"""
)

# -----------------------------
# ADD OBJECT
# -----------------------------

if mode == "Add Object":

    st.header("➕ Teach a new object")

    label = st.text_input("Object name")

    img_file = st.camera_input("Take picture")

    if img_file and label:

        image = Image.open(img_file)

        emb = get_embedding(image)

        if label not in db:
            db[label] = []

        db[label].append(emb)

        save_db(db)

        st.success("Image stored")

        st.write("Images stored:", len(db[label]))

# -----------------------------
# RECOGNITION
# -----------------------------

if mode == "Recognize":

    st.header("🔎 Object Recognition")

    img_file = st.camera_input("Show object to camera")

    if img_file:

        image = Image.open(img_file)

        emb = get_embedding(image)

        best_label = "Unknown"
        best_score = 0

        for label in db:

            stored = np.vstack(db[label])

            score = cosine_similarity(emb, stored).mean()

            if score > best_score:
                best_score = score
                best_label = label

        THRESHOLD = 0.75

        if best_score < THRESHOLD:
            best_label = "Unknown"

        st.subheader("Prediction")

        st.metric("Object", best_label)

        st.metric("Confidence", round(best_score,2))

# -----------------------------
# DATABASE VIEW
# -----------------------------

if mode == "Object Database":

    st.header("📚 Stored Objects")

    if len(db) == 0:
        st.info("No objects added yet")

    for obj in db:

        st.write(f"**{obj}** — {len(db[obj])} images")