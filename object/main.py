import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import pickle
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Object AI System", layout="centered")

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
img {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("Object Recognition System")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_detector():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_embed_model():
    model = models.mobilenet_v2(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

detector = load_detector()
embed_model = load_embed_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def get_embedding(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        emb = embed_model(img)
    return emb.numpy().reshape(1,-1)

# -----------------------------
# DETECTION FUNCTION
# -----------------------------
def detect_objects(image):
    img_tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        preds = detector([img_tensor])[0]

    draw = ImageDraw.Draw(image)

    for box, score in zip(preds['boxes'], preds['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    return image

# -----------------------------
# DB
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
# MODE
# -----------------------------
mode = st.selectbox(
    "Select Mode",
    ["Detect Objects", "Teach New Object", "Recognize Object"]
)

# -----------------------------
# CAMERA
# -----------------------------
def capture_image(label):
    img_file = st.camera_input(label)
    if img_file:
        try:
            image = Image.open(img_file).convert("RGB")
            return image
        except:
            return None
    return None

# -----------------------------
# DETECT
# -----------------------------
if mode == "Detect Objects":

    st.subheader("Object Detection")

    image = capture_image("Take a picture")

    if image is not None:
        try:
            result_img = detect_objects(image)
            st.image(result_img, use_column_width=True)
        except:
            st.warning("Detection failed")

# -----------------------------
# TEACH
# -----------------------------
elif mode == "Teach New Object":

    st.subheader("Teach Object")

    label = st.text_input("Object Name")
    image = capture_image("Capture Image")

    if image is not None and label:
        try:
            emb = get_embedding(image)

            if label not in db:
                db[label] = []

            db[label].append(emb)
            save_db(db)

            st.success("Image saved")
            st.write("Total samples:", len(db[label]))

        except:
            st.warning("Failed")

# -----------------------------
# RECOGNIZE
# -----------------------------
elif mode == "Recognize Object":

    st.subheader("Recognize")

    image = capture_image("Show object")

    if image is not None:
        try:
            emb = get_embedding(image)

            best_label = "Unknown"
            best_score = 0

            for label in db:
                stored = np.vstack(db[label])
                score = cosine_similarity(emb, stored).mean()

                if score > best_score:
                    best_score = score
                    best_label = label

            if best_score < 0.7:
                best_label = "Unknown"

            st.image(image, use_column_width=True)

            st.markdown("### Result")
            st.write("Prediction:", best_label)
            st.progress(float(best_score))
            st.write("Confidence:", round(best_score,2))

        except:
            st.warning("Recognition failed")
