import pandas as pd
import numpy as np
import streamlit as st
import time
from dataviz_elyes_donia.utils import read_uploaded_file, clean_dataframe

# Pipeline de traitement des données
def load_data(file):
    """
    Charger les données à partir d'un fichier CSV/Excel.

    Args:
        file: Fichier téléchargé par l'utilisateur.

    Returns:
        DataFrame nettoyé et prêt à l'analyse.
    """
    df = read_uploaded_file(file)
    if df is not None:
        df_cleaned = clean_dataframe(df)
        return df_cleaned
    else:
        st.error("❌ Impossible de charger les données.")
        return None

def normalize_data(df):
    """
    Normaliser les données numériques entre 0 et 1.

    Args:
        df: DataFrame à normaliser.

    Returns:
        DataFrame avec les colonnes numériques normalisées.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_normalized = df.copy()
    
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0

    return df_normalized

def transform_variables(df):
    """
    Transformation des variables : création de nouvelles colonnes à partir des données existantes.

    Args:
        df: DataFrame à transformer.

    Returns:
        DataFrame enrichi de nouvelles variables.
    """
    df_transformed = df.copy()

    # Exemple : créer une variable d'interaction si deux colonnes numériques existent
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[:2]
        df_transformed['interaction_' + col1 + '_' + col2] = df[col1] * df[col2]

    return df_transformed

def simulate_data_stream(df, delay=1):
    """
    Simuler un flux de données en émettant des lignes une par une.

    Args:
        df: DataFrame source.
        delay: Délai (en secondes) entre chaque envoi de ligne.
    """
    st.write("### 🚀 Flux de Données en Temps Réel")
    progress_bar = st.progress(0)

    for i, row in df.iterrows():
        st.write(row.to_frame().T)
        progress_bar.progress((i + 1) / len(df))
        time.sleep(delay)

        if i >= 10:  # Limiter la simulation à 10 lignes pour la démo
            break
