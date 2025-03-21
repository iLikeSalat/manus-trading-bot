# Recherche sur l'API Binance et les Indicateurs Techniques

## 1. API Binance

### Bibliothèques Python pour l'API Binance

#### python-binance
- **Description**: Wrapper non officiel pour l'API REST Binance v3
- **Fonctionnalités pertinentes pour notre application**:
  - Récupération des données de marché (prix, volumes, chandeliers)
  - Récupération des principales cryptomonnaies par capitalisation boursière
  - Support pour les websockets pour les mises à jour en temps réel
  - Gestion des limites de taux de l'API
- **Installation**: `pip install python-binance`
- **Documentation**: https://python-binance.readthedocs.io/
- **Exemple d'utilisation**:
```python
from binance import Client

client = Client(api_key, api_secret)
# Obtenir les données de chandeliers pour BTC/USDT
klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY)
```

#### binance-connector-python
- **Description**: Connecteur officiel et léger pour l'API publique de Binance
- **Fonctionnalités pertinentes**:
  - Support pour les endpoints `/api/*`
  - Flux de données de marché via WebSocket
- **Installation**: `pip install binance-connector`
- **Documentation**: https://binance-connector.readthedocs.io/

### Endpoints API Binance Utiles

#### Récupération des Principales Cryptomonnaies
- Pour obtenir les principales cryptomonnaies par volume d'échange:
  ```python
  # Avec python-binance
  tickers = client.get_ticker()
  # Trier par volume d'échange
  sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
  top_coins = sorted_tickers[:10]  # Top 10
  ```

#### Récupération des Données de Prix
- Pour obtenir les données OHLCV (Open, High, Low, Close, Volume):
  ```python
  # Avec python-binance
  klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY)
  ```

#### Limites de l'API
- L'API Binance impose des limites de taux (rate limits)
- Pour les comptes sans API key: 1200 requêtes par minute
- Pour les comptes avec API key: jusqu'à 6000 requêtes par minute selon le niveau de compte
- La bibliothèque python-binance gère automatiquement ces limites

## 2. Indicateurs Techniques

### RSI (Relative Strength Index)

#### Description
- Indicateur de momentum qui mesure la vitesse et le changement des mouvements de prix
- Oscille entre 0 et 100
- Valeurs typiques d'interprétation:
  - RSI < 30: Condition de survente (signal d'achat potentiel)
  - RSI > 70: Condition de surachat (signal de vente potentiel)

#### Calcul avec pandas-ta
```python
import pandas as pd
import pandas_ta as ta

# Supposons que df contient les données OHLCV
df['rsi'] = ta.rsi(df['close'], length=14)
```

#### Calcul avec TA-Lib
```python
import numpy as np
import talib

# Supposons que close_prices est un tableau numpy des prix de clôture
rsi = talib.RSI(close_prices, timeperiod=14)
```

### Moyennes Mobiles

#### Description
- Moyenne des prix sur une période spécifique
- Types courants:
  - SMA (Simple Moving Average): Moyenne arithmétique simple
  - EMA (Exponential Moving Average): Donne plus de poids aux prix récents

#### Calcul avec pandas-ta
```python
import pandas as pd
import pandas_ta as ta

# Supposons que df contient les données OHLCV
df['sma_50'] = ta.sma(df['close'], length=50)
df['ema_50'] = ta.ema(df['close'], length=50)
```

#### Calcul avec TA-Lib
```python
import numpy as np
import talib

# Supposons que close_prices est un tableau numpy des prix de clôture
sma_50 = talib.SMA(close_prices, timeperiod=50)
ema_50 = talib.EMA(close_prices, timeperiod=50)
```

### Détection de Croisement de Moyennes Mobiles

#### Description
- Signal d'achat: Prix croise au-dessus de la moyenne mobile
- Signal de vente: Prix croise en dessous de la moyenne mobile

#### Implémentation
```python
import numpy as np

def detect_price_crossover_ma(prices, ma):
    """
    Détecte quand le prix croise au-dessus ou en dessous de la moyenne mobile
    
    Args:
        prices: Série de prix
        ma: Série de moyenne mobile
        
    Returns:
        buy_signals: Liste des indices où le prix croise au-dessus de la MA
        sell_signals: Liste des indices où le prix croise en dessous de la MA
    """
    buy_signals = []
    sell_signals = []
    
    # Ignorer les valeurs NaN au début de la MA
    start_idx = np.where(~np.isnan(ma))[0][0]
    
    for i in range(start_idx + 1, len(prices)):
        # Croisement au-dessus (signal d'achat)
        if prices[i-1] <= ma[i-1] and prices[i] > ma[i]:
            buy_signals.append(i)
        
        # Croisement en dessous (signal de vente)
        elif prices[i-1] >= ma[i-1] and prices[i] < ma[i]:
            sell_signals.append(i)
    
    return buy_signals, sell_signals
```

## 3. Intégration des Données et des Indicateurs

### Flux de Travail Recommandé

1. **Collecte des Données**
   - Récupérer la liste des principales cryptomonnaies par volume d'échange
   - Pour chaque cryptomonnaie, récupérer les données OHLCV historiques
   - Mettre à jour les données à intervalles réguliers (1-5 minutes)

2. **Calcul des Indicateurs**
   - Calculer le RSI pour chaque cryptomonnaie
   - Calculer la moyenne mobile à 50 jours pour chaque cryptomonnaie
   - Stocker les résultats dans un DataFrame pandas

3. **Détection des Signaux**
   - Vérifier si le RSI est inférieur à 30 (ou au seuil personnalisé)
   - Vérifier si le prix a croisé au-dessus de la moyenne mobile à 50 jours
   - Générer des alertes pour les signaux détectés

4. **Affichage des Résultats**
   - Afficher les cryptomonnaies avec des signaux d'achat
   - Afficher les valeurs actuelles des indicateurs
   - Mettre à jour l'affichage à chaque nouvelle analyse

### Exemple de Structure de Données
```python
# Structure de données pour stocker les résultats
results = {
    'BTCUSDT': {
        'price': 50000.0,
        'rsi': 28.5,
        'sma_50': 48000.0,
        'buy_signal_rsi': True,  # RSI < 30
        'buy_signal_ma_crossover': False,
        'last_update': '2025-03-20 19:30:00'
    },
    'ETHUSDT': {
        'price': 3000.0,
        'rsi': 45.2,
        'sma_50': 2800.0,
        'buy_signal_rsi': False,
        'buy_signal_ma_crossover': True,  # Prix > SMA50
        'last_update': '2025-03-20 19:30:00'
    },
    # ... autres cryptomonnaies
}
```

## 4. Considérations Pratiques

### Gestion des Clés API
- Les clés API doivent être stockées de manière sécurisée
- Options de stockage:
  - Variables d'environnement
  - Fichier de configuration chiffré
  - Saisie manuelle au démarrage de l'application

### Optimisation des Performances
- Limiter le nombre de requêtes API en utilisant des websockets pour les mises à jour en temps réel
- Mettre en cache les données historiques pour éviter de les récupérer à chaque fois
- Utiliser des calculs vectorisés avec numpy et pandas pour les indicateurs techniques

### Gestion des Erreurs
- Gérer les erreurs de connexion à l'API Binance
- Gérer les cas où les données sont insuffisantes pour calculer les indicateurs
- Implémenter des mécanismes de retry avec backoff exponentiel pour les requêtes API échouées

## Conclusion

Cette recherche fournit les informations nécessaires pour implémenter l'application de signaux d'achat de cryptomonnaies. Les bibliothèques python-binance et pandas-ta/TA-Lib offrent toutes les fonctionnalités requises pour récupérer les données de marché et calculer les indicateurs techniques. La détection des signaux d'achat peut être implémentée en vérifiant les conditions spécifiées (RSI < 30, prix croise au-dessus de la MA50) sur les données traitées.
