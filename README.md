# GDG Event — AI Showcase Apps

Two interactive Streamlit demos built for the GDG event:

| App | File | What it does |
|-----|------|--------------|
| 🎧 Sonic Twin Matcher | `file.py` | Pick songs → get matched with someone who has the same music vibe |
| 🔍 Object Recognition | `object/main.py` | Detect, teach, and recognise objects from photos using AI |

---

## Requirements

- Python **3.10**
- pip packages listed in `requirements.txt`

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/shaguntembhurne/gdg_event.git
cd gdg_event
```

### 2. Create a virtual environment (recommended)

```bash
python3.10 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Apps

### 🎧 Sonic Twin Matcher

Matches attendees based on their favourite songs using audio features and cosine similarity.

```bash
streamlit run file.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

**How to use:**
1. Search for 1–3 songs that match your current mood.
2. Enter your name and click **Analyze & Find Match**.
3. See your "Sonic DNA" radar chart and your crowd twin!

> **Note:** The app uses `cleaned_dataset.csv` (included in the repo). No external API key is needed.

---

### 🔍 Object Recognition System

Detect objects in images, teach the AI new objects, and recognise them later.

```bash
streamlit run object/main.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

**Modes:**
- **Detect Objects** – Upload or capture an image; bounding boxes are drawn around detected items.
- **Teach New Object** – Give an object a name and save a photo of it to the database.
- **Recognize Object** – Show an object and the app predicts what it is from the saved database.

> **Note:** The first run downloads pre-trained PyTorch models (~100 MB). An internet connection is required for the initial download only.

---

## Project Structure

```
gdg_event/
├── file.py                  # Sonic Twin Matcher app
├── cleaned_dataset.csv      # Spotify audio-feature dataset
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version (3.10)
└── object/
    ├── main.py              # Object Recognition app
    ├── objects.pkl          # Saved object database (auto-created)
    ├── yolov8n.pt           # Model weights
    └── .streamlit/
        └── config.toml      # Streamlit theme config
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `cleaned_dataset.csv` not found | Make sure you are running from the repo root |
| Camera not working | Use the **Upload File** option instead |
| Slow first load | PyTorch models are downloading — wait a moment |
