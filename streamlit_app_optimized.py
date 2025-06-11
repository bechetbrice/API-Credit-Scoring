"""
Interface Streamlit optimisée pour API Railway
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time

# Configuration pour API en ligne
st.set_page_config(
    page_title="Credit Scoring - Railway", 
    page_icon="🌐", 
    layout="wide"
)

# URL API Railway
API_URL = "https://web-production-api-credit-scoring-production.up.railway.app"

# Traductions des variables
FEATURE_TRANSLATIONS = {
    "EXT_SOURCE_1": "Score Externe 1",
    "EXT_SOURCE_2": "Score Externe 2", 
    "EXT_SOURCE_3": "Score Externe 3",
    "CODE_GENDER": "Genre",
    "DAYS_EMPLOYED": "Ancienneté professionnelle",
    "INSTAL_DPD_MEAN": "Retards moyens",
    "PAYMENT_RATE": "Ratio d'endettement"
}

def test_railway_api():
    """Test de l'API Railway"""
    try:
        start = time.time()
        response = requests.get(f"{API_URL}/health", timeout=15)
        response_time = time.time() - start
        
        if response.status_code == 200:
            health_data = response.json()
            return True, response_time, health_data
        else:
            return False, response_time, None
    except Exception as e:
        return False, 999, str(e)

def call_railway_api(client_data):
    """Appel optimisé Railway"""
    
    # Test de connectivité
    st.write("Test de l'API Railway...")
    is_online, speed, health = test_railway_api()
    
    if not is_online:
        st.error("API Railway indisponible")
        if isinstance(health, str):
            st.error(f"Erreur: {health}")
        return None, None
    
    # Affichage du status
    if speed < 5:
        st.success(f"API Railway rapide ({speed:.1f}s)")
    elif speed < 15:
        st.warning(f"API Railway correcte ({speed:.1f}s)")
    else:
        st.error(f"API Railway lente ({speed:.1f}s)")
    
    # Informations API
    if health and isinstance(health, dict):
        st.info(f"Railway - Version: {health.get('version', 'N/A')} - Seuil: {health.get('threshold', 'N/A'):.3f}")
    
    # Appel de prédiction avec timeout adaptatif
    timeout = max(30, min(120, speed * 8))
    
    try:
        with st.spinner(f"Prédiction via Railway (timeout: {timeout}s)..."):
            start = time.time()
            response = requests.post(
                f"{API_URL}/predict",
                json=client_data,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            total_time = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Réponse Railway en {total_time:.1f}s")
            return result, f"Railway ({total_time:.1f}s)"
        else:
            st.error(f"Erreur Railway {response.status_code}")
            try:
                error_detail = response.json()
                st.error(f"Détail: {error_detail.get('error', 'Erreur inconnue')}")
            except:
                st.error(f"Réponse brute: {response.text[:200]}")
            return None, None
            
    except requests.exceptions.Timeout:
        st.error(f"Timeout Railway ({timeout}s)")
        st.info("Railway peut être surchargé. Réessayez dans quelques minutes.")
        return None, None
    except Exception as e:
        st.error(f"Erreur Railway: {str(e)}")
        return None, None

@st.cache_data
def load_feature_names():
    """Charger les features avec fallback"""
    try:
        with open('data/processed/final_features_list.json', 'r') as f:
            return json.load(f)['selected_features']
    except:
        return [f"feature_{i}" for i in range(234)]

def create_client_data_railway(ext_source_2, ext_source_3, ext_source_1, gender, days_employed, 
                               instal_dpd, payment_rate, amt_annuity, approved_cnt, instal_sum):
    """Créer les données client optimisées pour Railway"""
    
    feature_names = load_feature_names()
    client_data = {name: 0.0 for name in feature_names}
    
    # Mapping optimisé pour Railway
    feature_mapping = {
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3, 
        "EXT_SOURCE_1": ext_source_1,
        "CODE_GENDER": 1 if gender == "Homme" else 0,
        "DAYS_EMPLOYED": -abs(days_employed),
        "INSTAL_DPD_MEAN": instal_dpd,
        "PAYMENT_RATE": payment_rate,
        "AMT_ANNUITY": amt_annuity,
        "APPROVED_CNT_PAYMENT_MEAN": approved_cnt,
        "INSTAL_AMT_PAYMENT_SUM": instal_sum
    }
    
    # Mise à jour des features existantes
    for key, value in feature_mapping.items():
        if key in client_data:
            client_data[key] = float(value)
    
    return client_data

def display_railway_results(data, api_info):
    """Affichage des résultats Railway"""
    if not data:
        st.error("Aucun résultat Railway")
        return
    
    prob = data.get("probability", 0)
    decision = data.get("decision", "UNKNOWN")
    threshold = data.get("threshold", 0.1)
    platform = data.get("platform", "Railway")
    confidence = data.get("confidence", "MEDIUM")
    
    # Résultat principal
    if decision == "REFUSE":
        st.markdown(f"### CRÉDIT REFUSÉ (Railway)")
        st.markdown(f"**Risque de défaut: {prob:.1%}** (seuil: {threshold:.1%})")
    else:
        st.markdown(f"### CRÉDIT ACCORDÉ (Railway)") 
        st.markdown(f"**Risque de défaut: {prob:.1%}** (seuil: {threshold:.1%})")
    
    # Métriques Railway
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probabilité", f"{prob:.2%}")
    with col2:
        st.metric("Seuil Modèle", f"{threshold:.2%}")
    with col3:
        processing_time = data.get("processing_time", 0)
        st.metric("Temps Railway", f"{processing_time}s")
    with col4:
        st.metric("Confiance", confidence)
    
    # Facteurs explicatifs Railway
    if "top_features" in data:
        st.subheader("Facteurs clés")
        
        features = data["top_features"][:6]
        for i, feature in enumerate(features, 1):
            name = feature.get("feature", "Variable")
            importance = feature.get("importance", 0)
            impact = feature.get("impact", "Neutre")
            
            # Traduction si disponible
            display_name = FEATURE_TRANSLATIONS.get(name, name.replace("_", " ").title())
            
            st.write(f"**{i}. {display_name}** - Impact: {impact} ({importance:.3f})")
    
    # Infos techniques Railway
    with st.expander("Détails techniques Railway"):
        st.write(f"**Plateforme:** {platform}")
        st.write(f"**Version API:** {data.get('version', 'N/A')}")
        st.write(f"**Temps prédiction:** {data.get('prediction_time', 'N/A')}s")
        st.write(f"**Source:** {api_info}")

# Interface principale Railway
st.title("Credit Scoring - API Railway")
st.markdown("*Interface optimisée pour l'API déployée sur Railway*")

# Bannière Railway
st.info("**CONNEXION RAILWAY** - Interface connectée à l'API déployée sur Railway Cloud")

# Sidebar - Status Railway
with st.sidebar:
    st.header("Status Railway")
    
    if st.button("Tester Railway"):
        with st.spinner("Test Railway..."):
            is_ok, speed, health = test_railway_api()
        
        if is_ok:
            color = "Rapide" if speed < 5 else "Correct" if speed < 15 else "Lent"
            st.write(f"**Railway**: {speed:.1f}s ({color})")
            
            if health and isinstance(health, dict):
                st.json(health)
        else:
            st.write("**Railway**: Indisponible")
            if isinstance(health, str):
                st.error(health)
    
    # Infos Railway
    st.markdown("---")
    st.markdown("**Railway Info:**")
    st.caption(f"URL: {API_URL}")
    st.caption("Plateforme: Railway Cloud")
    st.caption("Optimisation: Sans SHAP")

st.markdown("---")

# Formulaire optimisé Railway
st.subheader("Analyse crédit via Railway")

# Instructions Railway
st.info("**Conseil Railway**: Les premières requêtes peuvent être plus lentes (cold start). Soyez patient!")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Scores externes (critiques)**")
    ext_source_2 = st.slider("Score Externe 2", 0.0, 1.0, 0.6, 0.01, 
                             help="Score principal - Impact majeur sur la décision")
    ext_source_3 = st.slider("Score Externe 3", 0.0, 1.0, 0.5, 0.01,
                             help="Score secondaire - Très important")
    ext_source_1 = st.slider("Score Externe 1", 0.0, 1.0, 0.4, 0.01,
                             help="Score tertiaire - Important")
    gender = st.selectbox("Genre", ["Femme", "Homme"], help="Genre du demandeur")
    days_employed = st.number_input("Ancienneté emploi (jours)", 0, 15000, 3000, 
                                   help="Nombre de jours depuis le début de l'emploi actuel")

with col2:
    st.markdown("**Profil financier**")
    instal_dpd = st.number_input("Retards moyens (jours)", 0.0, 50.0, 0.0, 0.1,
                                help="Nombre moyen de jours de retard sur les paiements")
    payment_rate = st.slider("Ratio endettement", 0.0, 1.0, 0.1, 0.01,
                            help="Ratio entre paiements et crédit")
    amt_annuity = st.number_input("Annuité prêt (€)", 0, 200000, 50000, 1000,
                                 help="Montant des paiements annuels")
    approved_cnt = st.number_input("Nb paiements approuvés", 0.0, 50.0, 10.0, 1.0,
                                  help="Nombre de paiements approuvés historiquement")
    instal_sum = st.number_input("Somme paiements (€)", 0, 1000000, 100000, 5000,
                                help="Somme totale des paiements effectués")

# Bouton Railway
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("ANALYSER VIA RAILWAY", type="primary", use_container_width=True):
        
        # Préparation données Railway
        client_data = create_client_data_railway(
            ext_source_2, ext_source_3, ext_source_1, gender, days_employed,
            instal_dpd, payment_rate, amt_annuity, approved_cnt, instal_sum
        )
        
        # Appel Railway
        start_total = time.time()
        result, api_info = call_railway_api(client_data)
        total_time = time.time() - start_total
        
        if result:
            st.success(f"Analyse Railway terminée en {total_time:.1f}s")
            display_railway_results(result, api_info)
        else:
            st.error("API Railway indisponible")
            st.markdown("""
            ### Solutions:
            1. **Réessayer** dans 2-3 minutes (Railway peut être en cold start)
            2. **Vérifier** votre connexion internet
            3. **Contacter** l'administrateur si le problème persiste
            """)

# Guide Railway
st.markdown("---")
st.markdown("### Guide Railway")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Profil faible risque:**
    - Score Externe 2 > 0.7
    - Score Externe 3 > 0.6  
    - Ancienneté > 3000 jours
    - Aucun retard (0 jours)
    """)

with col2:
    st.markdown("""
    **Profil haut risque:**
    - Scores externes < 0.3
    - Emploi récent < 1000 jours
    - Retards > 10 jours
    - Ratio endettement > 0.5
    """)

# Footer Railway
st.markdown("---")
st.caption("API déployée sur Railway Cloud")
st.caption("Optimisée pour performances en ligne")
st.caption("Brice Béchet - Projet Credit Scoring - 2025")