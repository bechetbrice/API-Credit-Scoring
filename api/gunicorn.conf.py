import os

# Configuration optimisée pour Railway
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
workers = 1  # Railway a des limitations mémoire
worker_class = "sync"
timeout = 120  # Timeout long pour les prédictions
keepalive = 2
max_requests = 1000

# Optimisations mémoire
preload_app = True  # Charge l'app une seule fois au démarrage
worker_tmp_dir = "/dev/shm"  # Utilise la RAM pour les fichiers temporaires

# Logs pour debugging
loglevel = "info"
accesslog = "-"
errorlog = "-"

def on_starting(server):
    print("Démarrage Gunicorn optimisé pour Railway")

def when_ready(server):
    print("✅ Gunicorn prêt à recevoir des requêtes")

def on_exit(server):
    print("🛑 Arrêt Gunicorn")