import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Object Recognition",
    layout="centered"
)

# -----------------------------
# HEADER
# -----------------------------

st.markdown("""
# Object Recognition Demo
Teach the system a new object using a few photos and it will recognize it later.
""")

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

model = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# EMBEDDING
# -----------------------------

def get_embedding(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        emb = model(img)

    emb = emb.numpy().reshape(1,-1)

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

st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Mode",
    ["Add Object","Recognize","Objects"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works**

1. Add a new object  
2. Capture a few images  
3. Try recognition
""")

# -----------------------------
# ADD OBJECT
# -----------------------------

if mode == "Add Object":

    st.subheader("Add New Object")

    col1, col2 = st.columns([1,1])

    with col1:

        label = st.text_input("Object Name")

        img_file = st.camera_input("Capture Image")

    with col2:

        st.markdown("**Tips**")
        st.write("• Take images from different angles")
        st.write("• Ensure good lighting")
        st.write("• Fill most of the frame")

    if img_file and label:

        image = Image.open(img_file)

        emb = get_embedding(image)

        if label not in db:
            db[label] = []

        db[label].append(emb)

        save_db(db)

        st.success("Image stored")

        st.write("Total images:", len(db[label]))


# -----------------------------
# RECOGNITION
# -----------------------------

if mode == "Recognize":

    st.subheader("Recognition")

    img_file = st.camera_input("Show object")

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

        threshold = 0.75

        if best_score < threshold:
            best_label = "Unknown"

        st.markdown("---")

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:

            st.metric("Prediction", best_label)

            st.progress(float(best_score))

            st.write("Confidence:", round(best_score,2))


# -----------------------------
# OBJECT DATABASE
# -----------------------------

if mode == "Objects":

    st.subheader("Stored Objects")

    if len(db) == 0:

        st.info("No objects stored yet.")

    else:

        for obj in db:

            col1, col2 = st.columns([3,1])

            with col1:
                st.write(obj)

            with col2:
                st.write(len(db[obj]), "images")