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
# COCO LABELS (Fixed for PyTorch 91-class output)
# -----------------------------
COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_detector():
    # 'pretrained' is deprecated in newer torchvision, but still works. 
    # Using 'weights="DEFAULT"' is the modern approach if you ever update.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_embed_model():
    model = models.mobilenet_v2(pretrained=True)
    # Strip the classifier layer
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
        # GAP (Global Average Pooling) to reduce the 1280x7x7 tensor to a 1280 vector
        emb = emb.mean([2, 3]) 
    return emb.numpy().reshape(1,-1)

# -----------------------------
# DETECTION (YOLO STYLE)
# -----------------------------
def detect_objects(image):
    img_tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        preds = detector([img_tensor])[0]

    draw = ImageDraw.Draw(image)

    for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
        if score > 0.5:
            x1, y1, x2, y2 = box.tolist()
            name = COCO_LABELS[label.item()] # Ensure label is an int

            # Skip the 'N/A' placeholders
            if name == 'N/A':
                continue

            # Box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Text
            text = f"{name} {score:.2f}"
            try:
                # Modern Pillow
                bbox = draw.textbbox((0, 0), text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # Fallback for older Pillow versions
                text_width, text_height = draw.textsize(text)

            # Background
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill="red"
            )

            # Text
            draw.text((x1 + 2, y1 - text_height - 2), text, fill="white")

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
    # Give the user an option to either upload a file or use the camera
    upload_method = st.radio("Choose image source:", ["Upload File", "Use Camera"], horizontal=True)
    
    img_file = None
    if upload_method == "Upload File":
        # This creates the drag-and-drop upload box
        img_file = st.file_uploader(f"{label} (Upload)", type=["jpg", "jpeg", "png"])
    else:
        # This opens the live webcam
        img_file = st.camera_input(f"{label} (Camera)")

    if img_file is not None:
        try:
            return Image.open(img_file).convert("RGB")
        except Exception as e:
            st.error(f"Error opening image: {e}")
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
        except Exception as e:
            st.error(f"Detection failed: {e}")

# -----------------------------
# TEACH
# -----------------------------
elif mode == "Teach New Object":
    st.subheader("Teach Object")
    label = st.text_input("Object Name")
    image = capture_image("Capture Image")

    if image is not None and label:
        # Require a button click so it doesn't duplicate saves on every Streamlit rerun
        if st.button("Save to Database"):
            try:
                emb = get_embedding(image)

                if label not in db:
                    db[label] = []

                db[label].append(emb)
                save_db(db)

                st.success(f"Image saved for '{label}'!")
                st.write("Total samples:", len(db[label]))

            except Exception as e:
                st.error(f"Failed to save: {e}")

# -----------------------------
# RECOGNIZE
# -----------------------------
elif mode == "Recognize Object":
    st.subheader("Recognize")
    image = capture_image("Show object")

    if image is not None:
        try:
            if not db:
                st.warning("Database is empty. Please teach some objects first!")
            else:
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
                st.progress(float(min(best_score, 1.0))) # Ensure progress bar doesn't exceed 1.0
                st.write("Confidence:", round(best_score, 2))

        except Exception as e:
            st.error(f"Recognition failed: {e}")
            
