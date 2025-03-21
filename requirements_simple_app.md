# Analyse des Exigences - Application de Signaux d'Achat de Cryptomonnaies

## Objectif Principal
Développer une application légère qui surveille les principales cryptomonnaies sur Binance et génère des signaux d'achat basés sur des indicateurs techniques simples.

## Exigences Fonctionnelles

### 1. Surveillance des Cryptomonnaies
- Surveiller les 5 à 10 principales cryptomonnaies par capitalisation boursière sur Binance
- Récupérer les données de prix en temps réel via l'API Binance
- Mettre à jour les données à intervalles réguliers (toutes les 1 à 5 minutes)

### 2. Détection des Signaux d'Achat
- Implémenter les stratégies de trading suivantes :
  - Signal d'achat lorsque le RSI (Relative Strength Index) est inférieur à 30
  - Signal d'achat lorsque le prix croise au-dessus de la moyenne mobile à 50 jours
- Permettre l'ajout facile d'autres stratégies simples à l'avenir

### 3. Affichage des Alertes
- Afficher les signaux d'achat détectés dans une interface simple
- L'interface peut être une sortie console, une application web légère, ou une application mobile
- Rafraîchir automatiquement l'affichage à chaque mise à jour des données

### 4. Gestion des Clés API
- Permettre à l'utilisateur de fournir sa propre clé API Binance pour l'accès aux données en temps réel
- Stocker les clés API de manière sécurisée

## Fonctionnalités Bonus (Optionnelles)

### 1. Affichage des Indicateurs
- Afficher les valeurs actuelles du RSI pour chaque cryptomonnaie surveillée
- Afficher les tendances de prix récentes

### 2. Personnalisation des Seuils
- Permettre à l'utilisateur de définir des seuils personnalisés pour les signaux d'achat
- Par exemple, permettre de modifier le seuil du RSI de 30 à une autre valeur

## Exigences Non Fonctionnelles

### 1. Performance
- L'application doit être légère et consommer peu de ressources
- Les temps de réponse doivent être rapides, même lors de la surveillance de plusieurs cryptomonnaies

### 2. Fiabilité
- L'application doit gérer correctement les erreurs de connexion à l'API Binance
- Les données doivent être validées pour éviter les faux signaux

### 3. Facilité d'Utilisation
- L'interface doit être simple et intuitive
- Les signaux d'achat doivent être clairement visibles et compréhensibles

### 4. Sécurité
- Les clés API doivent être stockées de manière sécurisée
- L'application ne doit pas avoir accès aux fonctionnalités de trading (lecture seule)

## Contraintes Techniques

### 1. Langages et Bibliothèques
- Utilisation de Python comme langage principal
- Utilisation de bibliothèques comme python-binance pour l'API Binance
- Utilisation de pandas-ta ou TA-Lib pour les calculs d'indicateurs techniques

### 2. Environnement d'Exécution
- L'application doit pouvoir fonctionner sur des systèmes Linux, Windows et macOS
- Possibilité de déploiement sur un serveur pour un accès continu

## Livrables Attendus

1. Code source de l'application
2. Documentation d'installation et d'utilisation
3. Guide de configuration des clés API
4. Instructions pour ajouter de nouvelles stratégies de trading (si applicable)

## Critères d'Acceptation

1. L'application peut se connecter à l'API Binance et récupérer les données des principales cryptomonnaies
2. Les signaux d'achat sont correctement détectés selon les stratégies définies
3. Les alertes sont affichées clairement dans l'interface
4. L'application fonctionne de manière fiable pendant une période prolongée
5. La configuration des clés API et des paramètres est simple et intuitive
