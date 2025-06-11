"""
API Flask optimisée pour Railway
Version sans SHAP
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
from functools import lru_cache

app = Flask(__name__)

# Chemins pour Railway
MODELS_DIR = Path(os.path.join(os.path.dirname(__file__), '..', 'models'))
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))

# Variables globales pour Railway
MODEL = None
THRESHOLD = None
FEATURES = None

@lru_cache(maxsize=1)
def load_model_cached():
    """Cache le modèle en mémoire pour Railway"""
    return joblib.load(MODELS_DIR / "lightgbm_final_model_optimized.pkl")

@lru_cache(maxsize=1) 
def load_threshold_cached():
    """Cache le seuil en mémoire pour Railway"""
    return joblib.load(MODELS_DIR / "optimal_threshold_optimized.pkl")

@lru_cache(maxsize=1)
def load_features_cached():
    """Cache les features en mémoire pour Railway"""
    with open(DATA_DIR / "final_features_list.json", 'r') as f:
        return json.load(f)['selected_features']

def initialize_railway():
    """Initialisation Railway"""
    global MODEL, THRESHOLD, FEATURES
    
    print("INITIALISATION RAILWAY...")
    start = time.time()
    
    try:
        MODEL = load_model_cached()
        THRESHOLD = load_threshold_cached()
        FEATURES = load_features_cached()
        
        init_time = time.time() - start
        print(f"Railway initialisé en {init_time:.2f}s")
        print(f"Seuil optimal: {THRESHOLD:.4f}")
        print(f"Features: {len(FEATURES)}")
        print("RAILWAY PRÊT")
        return True
        
    except Exception as e:
        print(f"Erreur Railway: {e}")
        return False

def prepare_data_railway(data):
    """Préparation données optimisée Railway"""
    if isinstance(data, dict):
        values = [data.get(feat, 0.0) for feat in FEATURES]
    elif isinstance(data, list):
        values = data if len(data) == len(FEATURES) else data + [0.0] * (len(FEATURES) - len(data))
    else:
        raise ValueError("Format invalide")
    
    # Nettoyage rapide pour Railway
    clean_values = []
    for v in values:
        if v is None or str(v).lower() in ['nan', 'inf', '-inf', 'null']:
            clean_values.append(0.0)
        else:
            try:
                clean_values.append(float(v))
            except:
                clean_values.append(0.0)
    
    return np.array([clean_values])

def get_top_features_railway():
    """Top features pré-calculées pour Railway"""
    return [
        {
            'feature': 'EXT_SOURCE_2', 
            'importance': 0.122, 
            'impact': 'Critique'
        },
        {
            'feature': 'EXT_SOURCE_3', 
            'importance': 0.120, 
            'impact': 'Critique'
        }, 
        {
            'feature': 'EXT_SOURCE_1', 
            'importance': 0.055, 
            'impact': 'Important'
        },
        {
            'feature': 'DAYS_EMPLOYED', 
            'importance': 0.039, 
            'impact': 'Modéré'
        },
        {
            'feature': 'CODE_GENDER', 
            'importance': 0.036, 
            'impact': 'Modéré'
        },
        {
            'feature': 'INSTAL_DPD_MEAN', 
            'importance': 0.036, 
            'impact': 'Modéré'
        },
        {
            'feature': 'PAYMENT_RATE', 
            'importance': 0.036, 
            'impact': 'Modéré'
        }
    ]

@app.route('/health', methods=['GET'])
def health():
    """Health check Railway"""
    return jsonify({
        'status': 'ONLINE',
        'platform': 'Railway',
        'service': 'Credit Scoring API',
        'model': 'LightGBM Optimized',
        'threshold': float(THRESHOLD),
        'features_count': len(FEATURES),
        'version': 'RAILWAY_V2'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction optimisée Railway"""
    start_total = time.time()
    
    try:
        # Vérification Railway
        if any([MODEL is None, THRESHOLD is None, FEATURES is None]):
            return jsonify({
                'error': 'Service Railway non initialisé',
                'platform': 'Railway'
            }), 500
        
        # Récupération données
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Pas de données fournies',
                'platform': 'Railway'
            }), 400
        
        # Préparation Railway
        X = prepare_data_railway(data)
        
        # Prédiction
        pred_start = time.time()
        probabilities = MODEL.predict_proba(X)
        probability = float(probabilities[0, 1])
        pred_time = time.time() - pred_start
        
        # Décision
        decision = "REFUSE" if probability >= THRESHOLD else "ACCORDE"
        
        # Features pour Railway
        top_features = get_top_features_railway()
        
        total_time = time.time() - start_total
        
        print(f"Railway: {decision} (prob: {probability:.3f}) en {total_time:.3f}s")
        
        return jsonify({
            'probability': probability,
            'decision': decision,
            'threshold': float(THRESHOLD),
            'top_features': top_features,
            'processing_time': round(total_time, 3),
            'prediction_time': round(pred_time, 3),
            'platform': 'Railway',
            'version': 'RAILWAY_V2',
            'confidence': 'HIGH' if abs(probability - THRESHOLD) > 0.1 else 'MEDIUM'
        })
        
    except Exception as e:
        error_time = time.time() - start_total
        print(f"Erreur Railway après {error_time:.3f}s: {e}")
        return jsonify({
            'error': f'Erreur Railway: {str(e)}',
            'platform': 'Railway',
            'processing_time': round(error_time, 3)
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil Railway"""
    return jsonify({
        'message': 'API Credit Scoring - Railway',
        'platform': 'Railway Cloud',
        'endpoints': {
            'health': 'GET /health - Statut Railway',
            'predict': 'POST /predict - Prédiction',
            'home': 'GET / - Cette page'
        },
        'performance': {
            'target_response_time': '< 10 secondes',
            'model': 'LightGBM sans SHAP',
            'features': len(FEATURES) if FEATURES else 234
        },
        'version': 'RAILWAY_V2'
    })

# Gestion d'erreurs Railway
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint non trouvé sur Railway',
        'platform': 'Railway',
        'available_endpoints': ['/health', '/predict', '/']
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'error': 'Erreur serveur Railway',
        'platform': 'Railway'
    }), 500

# Initialisation Railway
print("=" * 50)
print("DÉMARRAGE API RAILWAY")
print("=" * 50)

if not initialize_railway():
    raise RuntimeError("Impossible d'initialiser Railway")

print("=" * 50)
print("RAILWAY PRÊT")
print("=" * 50)

# Configuration Railway
if __name__ == '__main__':
    print("Test Railway en local...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=False)
else:
    print("Mode production Railway activé")