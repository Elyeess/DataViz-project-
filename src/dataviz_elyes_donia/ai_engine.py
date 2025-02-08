import anthropic
import streamlit as st
import logging
import base64
import pandas as pd
from io import BytesIO

# ✅ Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Initialisation du client Anthropic
def initialize_ai_client(api_key):
    return anthropic.Anthropic(api_key=api_key)

# ✅ Fonction générique pour envoyer des requêtes à Claude
def send_request_to_claude(client, prompt, max_tokens=2000):
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"❌ Erreur lors de la requête Claude : {e}")
        st.error(f"Erreur lors de la requête Claude : {str(e)}")
        return None

# ✅ Générer des recommandations IA
def generate_recommendations(df, api_key):
    client = initialize_ai_client(api_key)
    dataset_summary = f"""
    **Aperçu du Jeu de Données :**
    - Colonnes : {', '.join(df.columns)}
    - Statistiques descriptives :
    {df.describe(include='all').to_string()}
    """

    prompt = f"""
    Tu es un expert en analyse de données. À partir des informations suivantes :

    {dataset_summary}

    1. Identifie les principales tendances et anomalies dans les données.
    2. Suggère des actions basées sur ces insights pour améliorer les performances.
    3. Mets en évidence toute relation inattendue entre les variables.

    Génère un rapport concis sous forme de points clés.
    """
    return send_request_to_claude(client, prompt)

# ✅ Détecter des anomalies dans les données
def detect_anomalies(df, api_key):
    client = initialize_ai_client(api_key)
    dataset_summary = f"""
    **Aperçu des Données :**
    - Colonnes : {', '.join(df.columns)}
    - Statistiques descriptives :
    {df.describe(include='all').to_string()}
    """

    prompt = f"""
    Tu es un spécialiste en détection d'anomalies. Sur la base des données suivantes :

    {dataset_summary}

    Identifie les anomalies potentielles, en expliquant pourquoi elles pourraient être considérées comme telles.
    Donne des suggestions sur la manière de gérer ces anomalies.
    """
    return send_request_to_claude(client, prompt)

# ✅ Générer des visualisations personnalisées
def call_llm_for_viz(df, user_prompt, api_key):
    client = initialize_ai_client(api_key)
    dataset_summary = f"""
    Colonnes et types :
    {df.dtypes.to_string()}

    Description du jeu de données :
    {df.describe(include='all').to_string()}
    """

    prompt = f"""
    Tu es un expert en visualisation de données avec Python. En utilisant le DataFrame suivant :
    
    {dataset_summary}

    Crée un code Python pour générer la visualisation suivante :
    {user_prompt}

    Contraintes :
    - Utilise uniquement matplotlib, seaborn, ou plotly.
    - Donne uniquement le code Python entre balises ```python.
    - Le DataFrame est déjà chargé sous le nom 'df'.
    - Remplace plt.show() par st.pyplot(plt) pour compatibilité avec Streamlit.
    - Inclure des graphiques pertinents comme les histogrammes, heatmaps, diagrammes de corrélation, etc.
    """
    return send_request_to_claude(client, prompt, max_tokens=1500)

# ✅ Exécuter dynamiquement du code Python généré par l'IA
def exec_generated_code(code: str, df: pd.DataFrame):
    try:
        exec_globals = {
            "st": st,
            "pd": pd,
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn"),
            "px": __import__("plotly.express"),
            "df": df
        }
        exec(code, exec_globals)
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution du code généré : {e}")
        st.error(f"Erreur lors de l'exécution du code généré : {e}")
