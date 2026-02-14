import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Akıllı Görsel Analiz Sistemi", layout="wide")

MODEL_PATH = "weights/best.pt"

@st.cache_resource 
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.sidebar.title("Ayarlar")
conf_threshold = st.sidebar.slider("Güven Eşiği (Confidence)", 0.0, 1.0, 0.25)

st.title("Akıllı Görsel Analiz Sistemi")
st.write("YOLOv8 ile eğitilmiş modelinizi test edin.")

uploaded_file = st.file_uploader("Bir resim seçin veya sürükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Orijinal Resim")
        st.image(image, use_container_width=True)
      
    results = model.predict(image, conf=conf_threshold)
    
    res_plotted = results[0].plot()
    
    with col2:
        st.subheader("Model Tahmini")
        st.image(res_plotted, channels="BGR", use_container_width=True)

    st.divider()
    st.subheader("📊 Analiz Sonuçları")
    
    names = model.names
    detected_indices = results[0].boxes.cls.cpu().numpy()
    
    if len(detected_indices) > 0:
        counts = {}
        for idx in detected_indices:
            label = names[int(idx)]
            counts[label] = counts.get(label, 0) + 1
            
        cols = st.columns(len(counts))
        for i, (label, count) in enumerate(counts.items()):
            cols[i].metric(label, count)
            
        st.success(f"Toplam {len(detected_indices)} nesne başarıyla tespit edildi.")
    else:
        st.warning("Bu güven eşiği ile herhangi bir nesne tespit edilemedi.")