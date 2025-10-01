import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Adicionar o src ao path
sys.path.append('../src')
from viz import gradcam_heatmap

st.set_page_config(
    page_title="Classificador Cats vs Dogs",
    page_icon="🐱🐶",
    layout="wide"
)

# Título
st.title("🐱 vs 🐶 - Classificador de Imagens")
st.markdown("Upload uma imagem para classificar entre Gato e Cachorro")

# Sidebar para configurações
st.sidebar.header("Configurações")
model_choice = st.sidebar.selectbox(
    "Escolha o modelo:",
    ["MLP", "CNN"]
)

# Carregar modelos (com cache)
@st.cache_resource
def load_models():
    mlp_model = tf.keras.models.load_model('models/mlp_model.h5')
    cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
    return mlp_model, cnn_model

mlp_model, cnn_model = load_models()

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem...",
                                 type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Classificação")

        # Pré-processamento
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Selecionar modelo e prever
        if model_choice == "CNN":
            model = cnn_model
        else:
            model = mlp_model
            
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        class_names = ['Gato 🐱', 'Cachorro 🐶']
        result = class_names[predicted_class]
        
        st.metric("Predição", result)
        st.metric("Confiança", f"{confidence:.2%}")
        
        # Mostrar probabilidades
        st.write("**Probabilidades:**")
        for i, class_name in enumerate(class_names):
            prob = predictions[0][i]
            st.write(f"{class_name}: {prob:.2%}")

# Info
st.sidebar.markdown("---")
st.sidebar.info("Projeto de Redes Neurais - Cats vs Dogs")
