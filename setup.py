#!/usr/bin/env python3
"""
Script d'installation pour le bot de trading Telegram-Binance Futures.
Ce script installe toutes les dépendances nécessaires au fonctionnement du bot.
"""

import os
import sys
import subprocess
import platform

# Liste des dépendances requises
DEPENDENCIES = [
    "python-binance>=1.0.16",
    "python-telegram-bot>=13.7",
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "tabulate>=0.8.9",
    "aiohttp>=3.7.4",
    "asyncio>=3.4.3",
    "python-dotenv>=0.19.0",
    "colorama>=0.4.4"
]

def check_python_version():
    """Vérifie que la version de Python est compatible."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Erreur: Python {required_version[0]}.{required_version[1]} ou supérieur est requis.")
        print(f"Version actuelle: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    return True

def install_dependencies():
    """Installe les dépendances requises."""
    print("Installation des dépendances...")
    
    try:
        # Mettre à jour pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Installer les dépendances
        for dependency in DEPENDENCIES:
            print(f"Installation de {dependency}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
        
        print("Toutes les dépendances ont été installées avec succès.")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'installation des dépendances: {str(e)}")
        return False

def create_env_file():
    """Crée un fichier .env pour les variables d'environnement."""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    if os.path.exists(env_file):
        print(f"Le fichier {env_file} existe déjà.")
        return
    
    print("Création du fichier .env pour les variables d'environnement...")
    
    with open(env_file, "w") as f:
        f.write("# Configuration du bot de trading Telegram-Binance Futures\n\n")
        f.write("# Clés API Binance\n")
        f.write("BINANCE_API_KEY=\n")
        f.write("BINANCE_API_SECRET=\n\n")
        f.write("# Token Telegram\n")
        f.write("TELEGRAM_TOKEN=\n\n")
        f.write("# Paramètres de risque\n")
        f.write("RISK_PER_TRADE=5.0\n")
        f.write("MAX_TOTAL_RISK=10.0\n\n")
        f.write("# Utiliser le testnet Binance (true/false)\n")
        f.write("TESTNET=true\n")
    
    print(f"Fichier {env_file} créé avec succès.")
    print("Veuillez éditer ce fichier pour configurer vos clés API et autres paramètres.")

def create_config_file():
    """Crée un fichier de configuration JSON."""
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
    if os.path.exists(config_file):
        print(f"Le fichier {config_file} existe déjà.")
        return
    
    print("Création du fichier de configuration JSON...")
    
    import json
    
    config = {
        "binance_api_key": "",
        "binance_api_secret": "",
        "telegram_token": "",
        "allowed_chats": [],
        "testnet": True,
        "risk_per_trade": 5.0,
        "max_total_risk": 10.0,
        "monitoring_interval": 60,
        "enable_dashboard": True
    }
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Fichier {config_file} créé avec succès.")
    print("Veuillez éditer ce fichier pour configurer vos clés API et autres paramètres.")

def main():
    """Fonction principale."""
    print("=== Installation du Bot de Trading Telegram-Binance Futures ===\n")
    
    # Vérifier la version de Python
    if not check_python_version():
        sys.exit(1)
    
    # Installer les dépendances
    if not install_dependencies():
        sys.exit(1)
    
    # Créer les fichiers de configuration
    create_env_file()
    create_config_file()
    
    print("\nInstallation terminée avec succès!")
    print("\nPour démarrer le bot, exécutez:")
    print("python src/bot.py --config config.json")
    print("\nOu avec des variables d'environnement:")
    print("python src/bot.py")

if __name__ == "__main__":
    main()
