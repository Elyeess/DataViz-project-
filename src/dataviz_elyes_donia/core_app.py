import streamlit as st
import pandas as pd
import logging
import matplotlib.pyplot as plt

# Importation des modules développés
from ai_engine import generate_recommendations, detect_anomalies, call_llm_for_viz, exec_generated_code

# Chargement de la clé API depuis le fichier .env
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("Clé API manquante. Veuillez la définir dans le fichier .env")

# Configuration du journal de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title="📊 DataViz Project", layout="wide")
    st.sidebar.title("🚀 Menu de Navigation")

    pages = {
        "🏠 Accueil": "home",
        "💬 Génération IA Avancées": "ai_analytics"
    }

    selected_page = st.sidebar.radio("Choisissez une page :", list(pages.keys()))

    if selected_page == "🏠 Accueil":
        st.title("📊 DataViz Project Elyes-Donia")
        st.markdown("""
            **Cette application vous permet de :**
            - Générer des **recommandations IA** basées sur vos données.
            - Obtenir des **visualisations analytiques** de vos données.
        """)

    elif selected_page == "💬 Génération IA Avancées":
        uploaded_file = st.file_uploader("📂 Téléchargez votre fichier de données pour l'analyse IA :", type=["csv", "xlsx"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

            st.subheader("🔍 Génération de Summary et Recommandations IA")
            if st.button("🚀 Générer des Recommandations"):
                with st.spinner("Analyse en cours..."):
                    recommendations = generate_recommendations(df, API_KEY)
                    st.success("✅ Recommandations générées avec succès !")
                    st.markdown(recommendations)

            st.subheader("⚠️ Détection d'Anomalies")
            if st.button("🔎 Détecter des Anomalies"):
                with st.spinner("Analyse des anomalies en cours..."):
                    anomalies = detect_anomalies(df, API_KEY)
                    st.success("✅ Anomalies détectées avec succès !")
                    st.markdown(anomalies)

            # ✅ Nouvelle fonctionnalité : Génération de visualisations personnalisées
            st.subheader("📊 Génération de Visualisations")
            user_prompt = st.text_area("📝 Décrivez la visualisation souhaitée :", 
                                       placeholder="Exemple : Affiche un histogramme des ventes par mois")

            if st.button("🎨 Générer la Visualisation"):
                if user_prompt.strip():
                    with st.spinner("⏳ Génération du code de visualisation..."):
                        try:
                            generated_code = call_llm_for_viz(df, user_prompt, API_KEY)
                            st.subheader("🖥️ Code Généré par l'IA")
                            st.code(generated_code, language="python")

                            st.subheader("📈 Visualisation Générée")
                            exec_generated_code(generated_code, df)  # Exécuter le code généré pour afficher le graphique

                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération de la visualisation : {e}")
                            logger.error(f"❌ Erreur : {e}")
                else:
                    st.warning("⚠️ Veuillez décrire la visualisation souhaitée.")

if __name__ == "__main__":
    main()
