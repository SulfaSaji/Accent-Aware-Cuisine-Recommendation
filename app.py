# ---------------------------------------------------------
# 🎙 Accent-Aware Cuisine Recommendation System (Streamlit)
# ---------------------------------------------------------

import streamlit as st
import numpy as np
import librosa
import torch
import joblib
from transformers import AutoFeatureExtractor, HubertModel

@st.cache_resource
def load_assets():
    model = joblib.load("hubert_model.pkl")
    states = joblib.load("hubert_states.pkl")        # list of folder names in training order
    extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()
    return model, states, extractor, hubert

model, states, extractor, hubert = load_assets()

accent_to_cuisine = {
    "Andhra Pradesh": "Pulihora, Pesarattu",
    "Gujarat": "Dhokla, Thepla",
    "Jharkhand": "Litti Chokha",
    "Karnataka": "Bisi Bele Bath, Neer Dosa",
    "Kerala": "Appam, Avial, Puttu",
    "Tamil Nadu": "Pongal, Idli, Dosa",
}

st.set_page_config(page_title="Accent-Aware Cuisine", page_icon="🎧")
st.title("🎙 Accent-Aware Cuisine Recommendation")
st.write("Upload an English speech sample. I’ll detect the accent and suggest cuisines from that region 🍛")

uploaded = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])

if uploaded is not None:
    st.audio(uploaded, format="audio/wav")
    with st.spinner("Extracting HuBERT features and predicting..."):
        try:
            y, sr = librosa.load(uploaded, sr=16000)
            y = librosa.util.normalize(y)
            # keep max 5s
            if len(y) > 16000*5:
                y = y[:16000*5]
            # basic silence check
            if np.mean(np.abs(y)) < 0.01:
                st.warning("Audio is too quiet or silent. Please upload a clearer clip.")
                st.stop()

            inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = hubert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            pred = model.predict([emb])[0]
            raw_state = states[pred]                     # e.g., 'kerala'
            accent = raw_state.replace("_", " ").title() # 'Kerala'
            cuisine = accent_to_cuisine.get(accent, "Cuisine recommendation not available 🍽")

            st.success(f"🗣 Detected Accent: *{accent}*")
            st.info(f"🍲 Recommended Cuisine: *{cuisine}*")

            if hasattr(model, "predict_proba"):
                conf = float(np.max(model.predict_proba([emb])))*100
                st.caption(f"Confidence: {conf:.2f}%")

            st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")