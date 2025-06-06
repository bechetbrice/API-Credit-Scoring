"""
Interface Streamlit pour tester l'API de scoring crédit Railway
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Scoring Crédit - Prêt à dépenser",
    page_icon="💳",
    layout="wide"
)

# Configuration
API_URL = "https://web-production-a0c3b.up.railway.app"

# Dictionnaire de traduction des variables
FEATURE_TRANSLATIONS = {
    "EXT_SOURCE_1": "Score Externe 1",
    "EXT_SOURCE_2": "Score Externe 2", 
    "EXT_SOURCE_3": "Score Externe 3",
    "CODE_GENDER": "Genre",
    "DAYS_EMPLOYED": "Ancienneté professionnelle",
    "INSTAL_DPD_MEAN": "Retards moyens (jours)",
    "PAYMENT_RATE": "Ratio d'endettement",
    "AMT_ANNUITY": "Annuité du prêt",
    "APPROVED_CNT_PAYMENT_MEAN": "Historique des paiements",
    "INSTAL_AMT_PAYMENT_SUM": "Total des remboursements",
    "OWN_CAR_AGE": "Ancienneté du véhicule",
    "NAME_EDUCATION_TYPE_Higher_education": "Études supérieures",
    "PREV_CNT_PAYMENT_MEAN": "Paiements antérieurs",
    "AMT_GOODS_PRICE": "Valeur du bien",
    "DAYS_BIRTH": "Âge du client",
    "NAME_FAMILY_STATUS_Married": "Marié(e)",
    "AMT_CREDIT": "Montant emprunté",
    "AMT_INCOME_TOTAL": "Revenus totaux",
    "REGION_RATING_CLIENT": "Note risque de la région",
    "DAYS_REGISTRATION": "Ancienneté du dossier"
}

@st.cache_data
def load_data():
    """Charger le dataset preprocessé"""
    try:
        return pd.read_csv('data/processed/train_preprocessed_sample.csv', index_col=0)
    except FileNotFoundError:
        st.error("Fichier de données non trouvé")
        return pd.DataFrame()

def call_api_predict(client_data):
    """Appel à l'API pour prédiction"""
    try:
        response = requests.post(
            f"{API_URL}/predict", 
            json=client_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            try:
                error_detail = response.json()
                st.text(f"Détail: {error_detail.get('error', 'Erreur inconnue')}")
            except:
                st.text(f"Réponse: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        st.error(" Timeout: L'API Railway met trop de temps à répondre")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Impossible de se connecter à l'API Railway")
        return None
    except Exception as e:
        st.error(f" Erreur inattendue: {str(e)}")
        return None

def display_prediction_results(data):
    """Affichage des résultats de prédiction"""
    if data is None:
        return
        
    prob = data["probability"]
    decision = data["decision"]
    threshold = data["threshold"]
    
    # Décision finale avec couleur du texte
    if prob >= threshold:
        st.markdown(f"### :red[🔴 CRÉDIT REFUSÉ - Le prêt ne peut pas être accordé. Le taux de risque de défaut de paiement est supérieur au seuil ({prob:.1%} ≥ {threshold:.1%})]")
    else:
        st.markdown(f"### :green[🟢 CRÉDIT ACCORDÉ - Le prêt peut être accordé. Le taux de risque de défaut de paiement est inférieur au seuil ({prob:.1%} < {threshold:.1%})]")
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilité de défaut", f"{prob:.2%}")
    with col2:
        st.metric("Seuil optimal", f"{threshold:.2%}")
    with col3:
        ecart = abs(prob - threshold) / threshold * 100
        st.metric("Écart au seuil", f"{ecart:.1f}%")
    
    # Graphique SHAP
    display_shap_chart(data.get("top_features", []))

def display_shap_chart(top_features):
    """Affichage du graphique SHAP simplifié"""
    if not top_features:
        return
        
    st.markdown("### Facteurs clés de la décision")
    
    # Préparation des données
    df_features = pd.DataFrame(top_features)
    df_features["feature_fr"] = df_features["feature"].map(
        lambda x: FEATURE_TRANSLATIONS.get(x, x.replace("_", " ").title())
    )
    df_features = df_features.sort_values("shap_value", key=abs, ascending=True)
    
    # Couleur selon le signe (rouge = augmente risque, vert = diminue risque)
    df_features["color"] = df_features["shap_value"].apply(lambda x: "red" if x > 0 else "green")
    
    # Graphique horizontal simplifié
    fig = px.bar(
        df_features, 
        x='shap_value', 
        y='feature_fr',
        orientation='h',
        title="Impact des variables sur la décision",
        labels={'shap_value': 'Impact', 'feature_fr': ''},
        color='color',
        color_discrete_map={"red": "#ff4444", "green": "#44aa44"}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Impact sur le risque (négatif = diminue, positif = augmente)"
    )
    
    fig.update_traces(
        hoverinfo='none',
        hovertemplate=None
    )
    
    st.plotly_chart(fig, use_container_width=True)

def validate_form_data(**fields):
    """Valide que tous les champs obligatoires sont remplis"""
    missing_fields = []
    for name, value in fields.items():
        if value is None or value == 0 or value == "":
            missing_fields.append(name)
    return missing_fields

# Interface principale
st.title("💳 API de Scoring Crédit")
st.markdown("*Connecté à Railway*")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choisir une option",
    ["Prédiction par variables", "Prédiction par identifiant"]
)

# Test de connectivité
with st.sidebar:
    st.markdown("---")
    st.header("Test API Railway")
    if st.button("Tester connexion"):
        try:
            with st.spinner("Test de connexion..."):
                response = requests.get(f"{API_URL}/health", timeout=10)
            
            if response.status_code == 200:
                st.success(" API Railway accessible")
                data = response.json()
                st.json(data)
            else:
                st.error(f" Erreur {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.error(" Timeout lors du test")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connexion impossible")
        except Exception as e:
            st.error(f" Erreur: {str(e)}")

# Prédiction par variables
if option == "Prédiction par variables":
    st.subheader("Test de prédiction par variables")
    st.markdown("### 10 variables du modèle")
    
    # Formulaire avec 10 features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Scores externes**")
        ext_source_2 = st.slider("Score Externe 2", 0.0, 1.0, 0.5, 0.01)
        ext_source_3 = st.slider("Score Externe 3", 0.0, 1.0, 0.5, 0.01)
        ext_source_1 = st.slider("Score Externe 1", 0.0, 1.0, 0.5, 0.01)
        code_gender = st.selectbox("Genre :red[*]", ["", "Femme", "Homme"])
        days_employed = st.number_input("Nombre de jours d'ancienneté professionnelle :red[*]", min_value=0, step=100)
        
    with col2:
        st.markdown("**Historique de paiement**")
        instal_dpd_mean = st.number_input("Retards moyens (jours)", min_value=0.0, step=0.1)
        payment_rate = st.slider("Ratio d'endettement", 0.0, 1.0, 0.0, 0.005)
        amt_annuity = st.number_input("Annuité du prêt :red[*]", min_value=0, step=1000)
        approved_cnt_payment_mean = st.number_input("Historique des paiements :red[*]", min_value=0.0, step=1.0)
        instal_amt_payment_sum = st.number_input("Somme des paiements (€) :red[*]", min_value=0, step=10000)

    # Validation des données
    missing_fields = validate_form_data(
        genre=code_gender,
        anciennete_emploi=days_employed,
        montant_annuite=amt_annuity,
        nb_paiements_approuves=approved_cnt_payment_mean,
        somme_paiements=instal_amt_payment_sum
    )
    
    # Traduction des champs manquants
    field_translations = {
        "genre": "Genre",
        "anciennete_emploi": "Ancienneté professionnelle",
        "montant_annuite": "Annuité du prêt",
        "nb_paiements_approuves": "Historique des paiements",
        "somme_paiements": "Somme des paiements"
    }
    
    # Afficher les champs manquants
    if missing_fields:
        translated_fields = [field_translations.get(field, field) for field in missing_fields]
        st.warning(f"Veuillez remplir les champs suivants : {', '.join(translated_fields)}")
    
    # Bouton d'analyse 
    if st.button("Analyser la demande du client", type="primary", disabled=bool(missing_fields)):
        # Préparation des données (234 features complètes)
        gender_value = 0 if code_gender == "Femme" else 1
        days_employed_negative = -abs(days_employed)  # Conversion en négatif pour le modèle
        
        client_data = {
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3,
            "EXT_SOURCE_1": ext_source_1,
            "CODE_GENDER": gender_value,
            "DAYS_EMPLOYED": days_employed_negative,
            "INSTAL_DPD_MEAN": instal_dpd_mean,
            "PAYMENT_RATE": payment_rate,
            "AMT_ANNUITY": amt_annuity,
            "APPROVED_CNT_PAYMENT_MEAN": approved_cnt_payment_mean,
            "INSTAL_AMT_PAYMENT_SUM": instal_amt_payment_sum,
            **{f"feature_{i}": 0.0 for i in range(224)}  # Compléter à 234
        }
        
        with st.spinner("Analyse en cours..."):
            data = call_api_predict(client_data)
            display_prediction_results(data)

# Prédiction par identifiant
elif option == "Prédiction par identifiant":
    st.subheader("Prédiction par identifiant client")
    
    # Charger le dataset
    train_data = load_data()
    
    if not train_data.empty:
        st.markdown("Exemples d'identifiants: de 100002 à 100060")
        
        # Champ et bouton sur la même ligne
        col1, col2 = st.columns([2, 2])
        
        with col1:
            client_id = st.text_input(
                "Identifiant client",
                value="",
                placeholder="Ex: 100002"
            )
        
        with col2:
            st.write("")  # Espace pour aligner avec le label du text_input
            analyze_button = st.button("Analyser le dossier du client", type="primary", disabled=not client_id)
        
        if analyze_button:
            if not client_id.strip():
                st.error("Veuillez saisir un identifiant client")
                st.stop()
                
            try:
                client_id_int = int(client_id)
            except ValueError:
                st.error("L'identifiant client doit être un nombre entier")
                st.stop()
                
            if client_id_int not in train_data.index:
                st.error(f"Client {client_id_int} non trouvé dans le dataset")
                st.info(f"Identifiants disponibles: {train_data.index.min()} à {train_data.index.max()}")
                st.stop()
            
            st.write(f"Client {client_id_int} trouvé dans le dataset")
            
            # Récupérer les données du client et préparer pour l'API
            client_row = train_data.loc[client_id_int]
            client_data_dict = client_row.drop('TARGET').to_dict()
            
            # Nettoyer les données pour l'API
            clean_data = {k: float(v) if pd.notna(v) else 0.0 for k, v in client_data_dict.items()}
            
            with st.spinner("Analyse en cours..."):
                data = call_api_predict(clean_data)
                display_prediction_results(data)
    else:
        st.error("Impossible de charger les données d'exemple")

# Footer
st.markdown("---")
st.markdown("**Implémenter un modèle de scoring**")
st.markdown("Brice Béchet - Juin 2025 - Master 2 Data Scientist - OpenClassRoom")