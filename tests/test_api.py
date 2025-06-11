"""
Tests unitaires pour l'API Credit Scoring 

COMMANDES POUR EXECUTER LES TESTS :
===================================

1. ACTIVEZ VOTRE ENVIRONNEMENT VIRTUEL :
   conda activate nom_de_votre_env

2. INSTALLEZ PYTEST :
   pip install pytest

3. LANCEZ LES TESTS (depuis la racine du projet) :
   pytest tests/test_api.py -v
"""

import pytest
import json
import sys
import os
from pathlib import Path

# Ajouter le chemin de l'API pour les imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def get_test_client():
    """Crée un client de test Flask"""
    from api.app_production import app
    app.config['TESTING'] = True
    return app.test_client()

def get_valid_client_data():
    """Crée des données client réalistes pour les tests"""
    # Essayer de charger les vrais noms de features
    possible_paths = [
        project_root / 'data/processed/final_features_list.json',
        Path('data/processed/final_features_list.json')
    ]
    
    feature_names = None
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                feature_names = json.load(f)['selected_features']
            break
        except FileNotFoundError:
            continue
    
    if feature_names is None:
        # Fallback : créer 234 features génériques
        feature_names = [f"feature_{i}" for i in range(234)]
    
    # Créer les données avec toutes les features à 0
    client_data = {name: 0.0 for name in feature_names}
    
    # Utiliser l'exemple "Faible Risque" du notebook 3 (SK_ID_CURR: 288878)
    # Client avec probabilité 0.52% -> CRÉDIT ACCORDÉ
    realistic_values = {
        "EXT_SOURCE_2": 0.78,
        "EXT_SOURCE_3": 0.688,
        "EXT_SOURCE_1": 0.897,
        "INSTAL_DPD_MEAN": 0.0,
        "NAME_EDUCATION_TYPE_Higher_education": 1.0,
        "CODE_GENDER": 0,
        "DAYS_EMPLOYED": -3000,
        "PAYMENT_RATE": 0.05,
        "AMT_ANNUITY": 12000
    }
    
    # Appliquer les valeurs réalistes si les features existent
    for key, value in realistic_values.items():
        if key in client_data:
            client_data[key] = value
            
    return client_data

# Tests essentiels (7 tests)

def test_api_health():
    """Test que l'endpoint /health fonctionne"""
    client = get_test_client()
    response = client.get('/health')
    
    assert response.status_code == 200
    data = response.json
    assert data['status'] == 'OK'
    assert 'threshold' in data
    assert 'features_count' in data

def test_api_predict_structure():
    """Test structure de réponse de /predict avec données valides"""
    client = get_test_client()
    client_data = get_valid_client_data()
    
    response = client.post('/predict', 
                          json=client_data,
                          content_type='application/json')
    
    assert response.status_code == 200
    data = response.json
    
    # Vérifier structure complète de la réponse
    assert 'probability' in data
    assert 'decision' in data
    assert 'threshold' in data
    assert 'top_features' in data
    
    # Vérifier cohérence des valeurs
    assert 0 <= data['probability'] <= 1
    assert data['decision'] in ['ACCORDE', 'REFUSE']
    assert len(data['top_features']) == 10

def test_prediction_consistency():
    """Test que même input = même output (reproductibilité)"""
    client = get_test_client()
    client_data = get_valid_client_data()
    
    # Deux appels identiques
    response1 = client.post('/predict', json=client_data)
    response2 = client.post('/predict', json=client_data)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    # Doivent donner exactement le même résultat
    assert response1.json['probability'] == response2.json['probability']
    assert response1.json['decision'] == response2.json['decision']

def test_decision_logic():
    """Test cohérence seuil : probabilité vs décision"""
    client = get_test_client()
    client_data = get_valid_client_data()
    
    response = client.post('/predict', json=client_data)
    data = response.json
    
    assert response.status_code == 200
    
    # Vérifier cohérence seuil
    if data['probability'] >= data['threshold']:
        assert data['decision'] == 'REFUSE'
    else:
        assert data['decision'] == 'ACCORDE'

def test_model_artifacts_loaded():
    """Test que les artifacts du modèle sont chargés"""
    from api.app_production import model, threshold, feature_names
    
    assert model is not None
    assert threshold is not None
    assert feature_names is not None
    assert len(feature_names) == 234

def test_api_wrong_feature_count():
    """Test avec mauvais nombre de features"""
    client = get_test_client()
    
    # Seulement 3 features au lieu de 234
    wrong_data = {
        "feature_1": 0.0, 
        "feature_2": 0.0, 
        "feature_3": 0.0
    }
    
    response = client.post('/predict', json=wrong_data)
    assert response.status_code == 500
    
    # Vérifier message d'erreur informatif
    assert 'error' in response.json

if __name__ == "__main__":
    # Exécution directe
    pytest.main([__file__, "-v"])