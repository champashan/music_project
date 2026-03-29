import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'music_model.pkl')

if not os.path.exists(model_path):
    st.error(f"❌ Файл модели не найден по пути: {model_path}")
    st.info("Проверьте, что music_model.pkl лежит в той же папке, что и app.py")
    st.stop()
else:
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {e}")
        st.stop()

st.set_page_config(page_title="AI Music Producer", page_icon="🎵")
st.title("🎵 AI Music Hit Predictor")

st.sidebar.header("Параметры трека")
danc = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
ener = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
loud = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, -10.0)
acos = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.1)
vale = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)
temp = st.sidebar.slider("Tempo (BPM)", 50, 220, 120)

if st.button("🚀 Провести ИИ-экспертизу"):
    input_data = np.array([[danc, ener, loud, acos, temp, vale]])
    raw_prediction = model.predict(input_data)[0]
    
    if (danc + ener) < 0.1:
        final_score = 5 
    else:
        min_possible = 10
        max_possible = 55
           
        final_score = (raw_prediction - min_possible) / (max_possible - min_possible) * 100
        final_score = np.clip(final_score, 0, 100)

    st.divider()
    st.metric("Прогноз популярности (AI Index)", f"{int(final_score)} / 100")
    
    if final_score >= 75:
        st.success("🔥 СТАТУС: ПОТЕНЦИАЛЬНЫЙ ХИТ")
    elif final_score >= 40:
        st.info("📈 СТАТУС: СРЕДНИЙ ПОТЕНЦИАЛ")
    else:
        st.warning("🧊 СТАТУС: НИШЕВЫЙ ПРОДУКТ")
