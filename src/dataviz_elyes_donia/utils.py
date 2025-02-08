import pandas as pd
import streamlit as st
import logging

# Configuration du journal de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def read_uploaded_file(uploaded_file):
    """
    Lire un fichier CSV ou Excel et le convertir en DataFrame.
    
    Args:
        uploaded_file: Fichier téléchargé par l'utilisateur.

    Returns:
        pd.DataFrame ou None en cas d'erreur.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Format de fichier non supporté. Veuillez télécharger un fichier CSV ou Excel.")
            return None

        logger.info(f"✅ Fichier {uploaded_file.name} chargé avec succès.")
        return df

    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture du fichier : {e}")
        st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
        return None

def clean_dataframe(df):
    """
    Nettoyer un DataFrame : suppression des doublons, gestion des valeurs manquantes.

    Args:
        df: DataFrame à nettoyer.

    Returns:
        pd.DataFrame nettoyé.
    """
    try:
        df_cleaned = df.copy()

        # Nettoyage des noms de colonnes
        df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

        # Suppression des doublons
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        logger.info(f"✅ {initial_rows - len(df_cleaned)} doublons supprimés.")

        # Gestion des valeurs manquantes
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                else:
                    df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else "", inplace=True)

                logger.info(f"🔧 Valeurs manquantes remplies dans la colonne '{column}'.")

        return df_cleaned

    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage des données : {e}")
        st.error(f"Erreur lors du nettoyage des données : {str(e)}")
        return df

def display_dataframe_overview(df):
    """
    Afficher un aperçu des données : dimensions, colonnes et premières lignes.

    Args:
        df: DataFrame à afficher.
    """
    try:
        st.write("### 🗂️ Aperçu des données")
        st.write(f"**Dimensions :** {df.shape[0]} lignes × {df.shape[1]} colonnes")
        st.write(f"**Colonnes :** {', '.join(df.columns)}")

        st.write("### 📋Dataset head")
        st.dataframe(df.head())

        logger.info("✅ Aperçu des données affiché avec succès.")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'affichage des données : {e}")
        st.error(f"Erreur lors de l'affichage des données : {str(e)}")
