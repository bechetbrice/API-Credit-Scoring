"""
API Flask de scoring crédit - "Prêt à dépenser"
Version production avec Gunicorn
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
import json
import os
from pathlib import Path

app = Flask(__name__)

# Configuration des chemins
MODELS_DIR = Path("models") 
DATA_DIR = Path("data/processed")

# Chargement des artifacts de production
print("Chargement des artifacts...")

try:
    # Modèle et seuil optimal
    model = joblib.load(MODELS_DIR / "lightgbm_final_model.pkl")
    threshold = joblib.load(MODELS_DIR / "optimal_threshold_baseline.pkl")
    
    # Liste des features sélectionnées
    with open(DATA_DIR / "final_features_list.json", 'r') as f:
        features_data = json.load(f)
    feature_names = features_data['selected_features']
    
    # SHAP explainer pour l'importance locale
    explainer = shap.TreeExplainer(model)
    
    print(f"API prête")
    print(f"  - Modèle: LightGBM Final Production")
    print(f"  - Seuil optimal: {threshold:.4f}")
    print(f"  - Features: {len(feature_names)}")
    
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    raise

def prepare_client_data(data):
    """Prépare et valide les données client"""
    # Conversion en DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("Format de données invalide")
    
    # Validation du nombre de features
    if df.shape[1] != len(feature_names):
        raise ValueError(f'Nombre de features incorrect. Attendu: {len(feature_names)}, reçu: {df.shape[1]}')
    
    # Attribution des noms de features corrects
    df.columns = feature_names
    
    # Nettoyage des données
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)
    
    return df

def calculate_shap_contributions(df):
    """Calcule les contributions SHAP pour explicabilité"""
    shap_values = explainer.shap_values(df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Classe positive (défaut)
    
    # Construction du top 10 des features importantes
    feature_contributions = []
    for i, (name, shap_val) in enumerate(zip(feature_names, shap_values[0])):
        feature_contributions.append({
            'feature': name,
            'shap_value': float(shap_val),
            'feature_value': float(df.iloc[0, i]) if pd.notna(df.iloc[0, i]) else None,
            'impact': 'increase_risk' if shap_val > 0 else 'decrease_risk'
        })
    
    # Tri par importance absolue
    feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return feature_contributions[:10]

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé - Vérification du statut de l'API"""
    return jsonify({
        'status': 'healthy',
        'service': 'Credit Scoring API',
        'model_type': 'LightGBM',
        'threshold': float(threshold),
        'features_count': len(feature_names),
        'version': '1.0',
        'environment': 'production'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction principal"""
    try:
        # Récupération et validation des données
        data = request.json
        if data is None:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400
        
        # Préparation des données
        df = prepare_client_data(data)
        
        # Prédiction
        probabilities = model.predict_proba(df)
        probability = float(probabilities[0, 1])  # Probabilité de défaut
        
        # Décision basée sur le seuil optimal
        decision = "REFUSE" if probability >= threshold else "ACCORDE"
        
        # Calcul des SHAP values pour explicabilité
        top_features = calculate_shap_contributions(df)
        
        # Réponse structurée
        response = {
            'probability': probability,
            'decision': decision,
            'threshold': float(threshold),
            'top_features': top_features
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({
            'error': f'Erreur lors de la prédiction: {str(e)}',
            'details': 'Vérifiez le format des données d\'entrée'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Gestionnaire d'erreur 404"""
    return jsonify({
        'error': 'Endpoint non trouvé',
        'available_endpoints': ['/health', '/predict']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Gestionnaire d'erreur 500"""
    return jsonify({
        'error': 'Erreur interne du serveur',
        'message': 'Contactez l\'administrateur'
    }), 500

# Configuration pour production
if __name__ == '__main__':
    # Mode développement local uniquement
    app.run(host='0.0.0.0', port=5001, debug=False)
else:
    # Mode production avec Gunicorn
    app.config['ENV'] = 'production'
    app.config['DEBUG'] = False