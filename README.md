# Guide d'Utilisation - Bot de Trading Telegram-Binance Futures

## Introduction

Ce bot de trading automatisé est conçu pour lire les signaux de trading depuis Telegram et exécuter automatiquement des positions sur Binance Futures avec une gestion intelligente des risques. Le système surveille en permanence les positions ouvertes, ajuste les stop-loss au point d'équilibre après avoir atteint certains niveaux de profit, et gère le risque global du portefeuille.

## Fonctionnalités Principales

- **Lecture des Signaux Telegram** : Analyse automatique des messages de signaux de trading
- **Exécution Automatique des Trades** : Placement des ordres d'entrée, take profit et stop loss
- **Gestion Intelligente des Risques** : Calcul automatique de la taille des positions et du levier
- **Surveillance des Positions** : Monitoring en temps réel des positions ouvertes
- **Interface Utilisateur** : Interface en ligne de commande et tableau de bord de monitoring
- **Système de Notifications** : Alertes pour les événements importants

## Prérequis

- Python 3.8 ou supérieur
- Compte Binance Futures avec API activée
- Bot Telegram (pour la réception des signaux)

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-nom/telegram-binance-bot.git
cd telegram-binance-bot
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurez les variables d'environnement ou créez un fichier de configuration :
```bash
export BINANCE_API_KEY="votre_clé_api"
export BINANCE_API_SECRET="votre_secret_api"
export TELEGRAM_TOKEN="votre_token_telegram"
```

## Configuration

Le bot peut être configuré via un fichier JSON, des variables d'environnement ou des arguments en ligne de commande.

### Exemple de fichier de configuration (config.json) :

```json
{
  "binance_api_key": "votre_clé_api",
  "binance_api_secret": "votre_secret_api",
  "telegram_token": "votre_token_telegram",
  "allowed_chats": [123456789],
  "testnet": true,
  "risk_per_trade": 5.0,
  "max_total_risk": 10.0,
  "monitoring_interval": 60,
  "enable_dashboard": true
}
```

### Paramètres de Configuration :

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| binance_api_key | Clé API Binance | - |
| binance_api_secret | Secret API Binance | - |
| telegram_token | Token du bot Telegram | - |
| allowed_chats | Liste des IDs de chat Telegram autorisés | [] |
| testnet | Utiliser le testnet Binance | true |
| risk_per_trade | Pourcentage du portefeuille à risquer par trade | 5.0 |
| max_total_risk | Risque total maximum autorisé | 10.0 |
| monitoring_interval | Intervalle de vérification des positions (en secondes) | 60 |
| enable_dashboard | Activer le tableau de bord | true |

## Utilisation

### Démarrage du Bot :

```bash
python src/bot.py --config config.json
```

Ou avec des arguments en ligne de commande :

```bash
python src/bot.py --binance-api-key "votre_clé_api" --binance-api-secret "votre_secret_api" --telegram-token "votre_token_telegram" --risk-per-trade 5.0 --max-total-risk 10.0
```

### Commandes de l'Interface en Ligne de Commande :

| Commande | Description |
|----------|-------------|
| help | Affiche l'aide des commandes |
| status | Affiche le statut général du bot |
| positions | Affiche les positions ouvertes |
| trades | Affiche les trades actifs |
| performance | Affiche les statistiques de performance |
| risk | Affiche les paramètres de risque |
| cancel <signal_id> | Annule un trade actif |
| signal <message> | Traite un signal manuellement |
| config | Affiche ou modifie la configuration |
| exit | Quitte l'interface |

## Format des Signaux

Le bot est conçu pour analyser des signaux de trading au format suivant :

```
🟢 ETH LONG 

🎯Entry price: 2255 - 2373

TP:
2500  
2601  
2770  
3000  
3180  
3300  

🛑 SL 2150
```

Les éléments clés que le bot extrait sont :
- Le symbole (ETH)
- La direction (LONG ou SHORT)
- La plage de prix d'entrée (2255 - 2373)
- Les niveaux de take profit (2500, 2601, etc.)
- Le stop loss (2150)

## Gestion des Risques

Le bot utilise un système sophistiqué de gestion des risques :

1. **Calcul de la Taille des Positions** : Basé sur le pourcentage du portefeuille à risquer et la distance entre le prix d'entrée et le stop loss.

2. **Calcul du Levier** : Déterminé automatiquement en fonction de la distance au stop loss pour optimiser l'utilisation du capital.

3. **Gestion des Positions Multiples** : Le bot évite la surexposition en limitant le risque total du portefeuille.

4. **Déplacement Automatique du Stop Loss** : Après avoir atteint TP2 ou TP3, le stop loss est déplacé au point d'équilibre pour protéger les gains.

## Dépannage

### Problèmes Courants :

1. **Erreur d'API Binance** : Vérifiez que vos clés API sont correctes et ont les permissions nécessaires.

2. **Erreur de Connexion Telegram** : Assurez-vous que votre token Telegram est valide et que le bot a été démarré correctement.

3. **Erreur de Parsing des Signaux** : Vérifiez que le format des signaux correspond à celui attendu par le bot.

### Logs :

Les logs du bot sont enregistrés dans le fichier `trading_bot.log`. Consultez ce fichier pour obtenir des informations détaillées sur les erreurs.

## Sécurité

- Les clés API sont stockées localement et ne sont jamais partagées.
- Il est recommandé d'utiliser des clés API avec des permissions limitées (lecture et trading uniquement).
- Activez l'authentification à deux facteurs sur votre compte Binance.

## Avertissement

Le trading de cryptomonnaies comporte des risques significatifs. Ce bot est fourni à titre éducatif et expérimental. N'investissez que ce que vous êtes prêt à perdre.

## Support

Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt GitHub ou contacter le développeur.

---

Bonne chance avec vos trades automatisés !
