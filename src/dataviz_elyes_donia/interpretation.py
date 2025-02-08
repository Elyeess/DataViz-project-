import streamlit as st
import pandas as pd
from PIL import Image
from dataviz_elyes_donia.ai_engine import interpret_data
from dataviz_elyes_donia.data_pipeline import load_data
from dataviz_elyes_donia.utils import display_dataframe_overview
from dotenv import load_dotenv
import os

# Charger la clé API depuis le fichier .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

def display_interpretation():
    st.title("🧠 Interprétation IA des Données et Images")

    uploaded_file = st.file_uploader("📂 Téléchargez votre fichier de données (CSV, Excel)", type=["csv", "xlsx"])
    uploaded_image = st.file_uploader("🖼️ Téléchargez une image (JPG, PNG)", type=["jpg", "png"])

    if uploaded_file and uploaded_image:
        try:
            df = load_data(uploaded_file)
            st.subheader("🔍 Aperçu des Donnees")
            display_dataframe_overview(df)

            st.subheader("🖼️ Aperçu de l'Image Téléchargée")
            image = Image.open(uploaded_image)
            st.image(image, caption="Image fournie", use_container_width=True)

            if st.button("🚀 Générer l'Interprétation IA"):
                with st.spinner("Analyse en cours..."):
                    interpretation = interpret_data(df, uploaded_image, API_KEY)
                    st.success("✅ Interprétation générée avec succès !")
                    st.markdown(interpretation)
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement des données : {e}")
    else:
        st.info("Veuillez télécharger un fichier de données et une image pour continuer.")

if __name__ == "__main__":
    display_interpretation()
