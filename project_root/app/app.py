import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

st.set_page_config(page_title="Movie Poster → Genres", page_icon="🎬", layout="centered")

# Paths to exported assets
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "export_movie_genre_app")
MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, "model.keras"))
LABELS_PATH = os.path.abspath(os.path.join(MODEL_DIR, "labels.json"))

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return model, labels

model, labels = load_model_and_labels()

st.title("Movie Poster → Genres 🎬")
st.caption("Upload a poster image to get multi‑label genre predictions (sigmoid head). Ensure labels.json matches training order.")

uploaded = st.file_uploader("Upload poster (JPG/PNG)", type=["jpg","jpeg","png"])
col1, col2 = st.columns(2)
with col1:
    topk = st.slider("Top‑k to show", 1, min(10, len(labels)), 5)
with col2:
    run = st.button("Predict")

placeholder = st.empty()

# Match training size and preprocess
INPUT_SIZE = (224, 224)  # change if you trained at a different size

def prepare(img: Image.Image):
    img = img.convert("RGB").resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Input poster")
else:
    img = None

if run:
    if img is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Running inference..."):
            x = prepare(img)
            probs = model.predict(x, verbose=0)[0]
        order = np.argsort(-probs)

        for i in range(min(topk, len(labels))):
            idx = int(order[i])
            genre = labels[idx]
            score = float(probs[idx])
            placeholder.info(f"{i+1}. {genre}: {score:.3f}")
            st.progress(min(1.0, score), text=genre)

        st.subheader("All genres (sorted)")
        st.dataframe(
            {"genre": [labels[i] for i in order], "prob": [float(probs[i]) for i in order]},
            use_container_width=True,
            hide_index=True,
        )
