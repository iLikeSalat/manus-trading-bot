# Architecture du Bot de Trading Telegram-Binance Futures

## Vue d'ensemble

Ce document présente l'architecture détaillée du bot de trading automatisé qui lit les signaux depuis Telegram et exécute des trades sur Binance Futures avec une gestion intelligente des risques.

L'architecture est conçue pour être modulaire, robuste et évolutive, permettant une maintenance facile et des extensions futures.

## Diagramme d'Architecture

```
+---------------------+      +----------------------+      +----------------------+
|                     |      |                      |      |                      |
| Module Telegram     |----->| Module de Parsing    |----->| Module de Validation |
| (Réception Signaux) |      | (Extraction Données) |      | (Vérification)       |
|                     |      |                      |      |                      |
+---------------------+      +----------------------+      +----------------------+
                                                                     |
                                                                     v
+---------------------+      +----------------------+      +----------------------+
|                     |      |                      |      |                      |
| Module Binance      |<-----| Module d'Exécution   |<-----| Module de Gestion   |
| (API Futures)       |      | (Placement Ordres)   |      | des Risques         |
|                     |      |                      |      |                      |
+---------------------+      +----------------------+      +----------------------+
        ^                             |                             ^
        |                             v                             |
        |                    +----------------------+               |
        |                    |                      |               |
        +--------------------| Module de Monitoring |---------------+
                             | (Suivi & Alertes)    |
                             |                      |
                             +----------------------+
```

## Modules Principaux

### 1. Module Telegram (Réception des Signaux)

**Objectif**: Recevoir et prétraiter les signaux de trading depuis Telegram.

**Composants**:
- **TelegramClient**: Gère la connexion à l'API Telegram et l'authentification.
- **MessageHandler**: Traite les messages entrants et les filtre.
- **SignalQueue**: File d'attente pour les signaux reçus à traiter.

**Fonctionnalités**:
- Connexion à Telegram via l'API Bot ou en tant que client utilisateur.
- Écoute des messages dans un canal spécifique ou en mode direct.
- Filtrage préliminaire des messages pour identifier les signaux potentiels.
- Mise en file d'attente des signaux pour traitement.

**Implémentation**:
```python
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

class TelegramModule:
    def __init__(self, token, signal_queue, allowed_chats=None):
        self.token = token
        self.signal_queue = signal_queue
        self.allowed_chats = allowed_chats or []
        self.application = Application.builder().token(token).build()
        
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Vérifier si le message provient d'une source autorisée
        if self.allowed_chats and update.effective_chat.id not in self.allowed_chats:
            return
            
        # Prétraiter le message
        message_text = update.message.text
        
        # Vérifier si c'est un signal potentiel (filtrage basique)
        if "LONG" in message_text or "SHORT" in message_text:
            # Ajouter à la file d'attente pour traitement
            self.signal_queue.put({
                'text': message_text,
                'chat_id': update.effective_chat.id,
                'message_id': update.message.message_id,
                'timestamp': update.message.date.timestamp()
            })
            
    def start(self):
        # Configurer les gestionnaires
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        # Démarrer le bot en mode polling
        self.application.run_polling()
```

### 2. Module de Parsing (Extraction des Données)

**Objectif**: Extraire les informations structurées des signaux de trading.

**Composants**:
- **SignalParser**: Analyse le texte des signaux et en extrait les données.
- **RegexPatterns**: Définitions des modèles d'expressions régulières pour l'extraction.

**Fonctionnalités**:
- Extraction du symbole de la cryptomonnaie.
- Identification de la direction du trade (LONG/SHORT).
- Extraction de la plage de prix d'entrée.
- Extraction des niveaux de Take Profit.
- Extraction du niveau de Stop Loss.

**Implémentation**:
```python
import re
from dataclasses import dataclass

@dataclass
class TradingSignal:
    symbol: str
    direction: str  # "LONG" ou "SHORT"
    entry_min: float
    entry_max: float
    take_profit_levels: list[float]
    stop_loss: float
    raw_text: str
    
class SignalParser:
    def __init__(self):
        # Définir les patterns regex pour l'extraction
        self.direction_pattern = re.compile(r'(LONG|SHORT)', re.IGNORECASE)
        self.symbol_pattern = re.compile(r'🟢\s+(\w+)\s+', re.IGNORECASE)
        self.entry_pattern = re.compile(r'Entry price:\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)')
        self.tp_pattern = re.compile(r'TP:(.*?)(?:🛑|$)', re.DOTALL)
        self.sl_pattern = re.compile(r'SL\s+(\d+(?:\.\d+)?)')
        
    def parse(self, message_text):
        # Extraire la direction
        direction_match = self.direction_pattern.search(message_text)
        direction = direction_match.group(1).upper() if direction_match else None
        
        # Extraire le symbole
        symbol_match = self.symbol_pattern.search(message_text)
        symbol = symbol_match.group(1) if symbol_match else None
        
        # Extraire la plage de prix d'entrée
        entry_match = self.entry_pattern.search(message_text)
        entry_min = float(entry_match.group(1)) if entry_match else None
        entry_max = float(entry_match.group(2)) if entry_match else None
        
        # Extraire les niveaux de Take Profit
        tp_levels = []
        tp_section = self.tp_pattern.search(message_text)
        if tp_section:
            tp_text = tp_section.group(1)
            tp_levels = [float(level.strip()) for level in re.findall(r'(\d+(?:\.\d+)?)', tp_text)]
        
        # Extraire le Stop Loss
        sl_match = self.sl_pattern.search(message_text)
        stop_loss = float(sl_match.group(1)) if sl_match else None
        
        # Créer et retourner l'objet signal
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_min=entry_min,
            entry_max=entry_max,
            take_profit_levels=tp_levels,
            stop_loss=stop_loss,
            raw_text=message_text
        )
```

### 3. Module de Validation (Vérification)

**Objectif**: Vérifier la validité et la cohérence des signaux extraits.

**Composants**:
- **SignalValidator**: Vérifie la validité des données extraites.
- **MarketDataChecker**: Vérifie la cohérence avec les données de marché actuelles.

**Fonctionnalités**:
- Validation de la présence de toutes les informations requises.
- Vérification de la cohérence des prix (entrée, TP, SL).
- Vérification de l'existence du symbole sur Binance Futures.
- Vérification des limites de prix et de quantité de Binance.

**Implémentation**:
```python
class SignalValidator:
    def __init__(self, binance_client):
        self.binance_client = binance_client
        
    def validate(self, signal):
        errors = []
        
        # Vérifier la présence des champs obligatoires
        if not signal.symbol:
            errors.append("Symbole manquant")
        
        if not signal.direction:
            errors.append("Direction (LONG/SHORT) manquante")
        
        if not signal.entry_min or not signal.entry_max:
            errors.append("Plage de prix d'entrée incomplète")
        
        if not signal.take_profit_levels or len(signal.take_profit_levels) == 0:
            errors.append("Niveaux de Take Profit manquants")
        
        if not signal.stop_loss:
            errors.append("Stop Loss manquant")
            
        # Si des erreurs de base sont trouvées, ne pas continuer
        if errors:
            return errors
            
        # Vérifier la cohérence des prix
        if signal.direction == "LONG":
            if signal.stop_loss >= signal.entry_min:
                errors.append("Stop Loss doit être inférieur au prix d'entrée pour un LONG")
            
            for tp in signal.take_profit_levels:
                if tp <= signal.entry_max:
                    errors.append(f"Take Profit {tp} doit être supérieur au prix d'entrée pour un LONG")
        else:  # SHORT
            if signal.stop_loss <= signal.entry_max:
                errors.append("Stop Loss doit être supérieur au prix d'entrée pour un SHORT")
            
            for tp in signal.take_profit_levels:
                if tp >= signal.entry_min:
                    errors.append(f"Take Profit {tp} doit être inférieur au prix d'entrée pour un SHORT")
        
        # Vérifier l'existence du symbole sur Binance
        try:
            symbol_info = self.binance_client.exchange_info()
            symbols = [info['symbol'] for info in symbol_info['symbols']]
            futures_symbol = f"{signal.symbol}USDT"
            
            if futures_symbol not in symbols:
                errors.append(f"Symbole {futures_symbol} non disponible sur Binance Futures")
        except Exception as e:
            errors.append(f"Erreur lors de la vérification du symbole: {str(e)}")
            
        return errors
```

### 4. Module de Gestion des Risques

**Objectif**: Calculer et gérer le risque pour chaque trade et pour l'ensemble du portefeuille.

**Composants**:
- **RiskCalculator**: Calcule la taille de position et le levier appropriés.
- **PortfolioManager**: Gère l'exposition globale et les positions multiples.

**Fonctionnalités**:
- Calcul de la taille de position basé sur le pourcentage de risque (5% par signal).
- Calcul du levier optimal basé sur la distance entre le prix d'entrée et le stop loss.
- Gestion des positions multiples pour éviter la surexposition.
- Ajustement automatique du stop loss au point d'équilibre après TP2/TP3.

**Implémentation**:
```python
class RiskManager:
    def __init__(self, binance_client, risk_per_trade=5.0, max_total_risk=10.0):
        self.binance_client = binance_client
        self.risk_per_trade = risk_per_trade  # Pourcentage du portefeuille à risquer par trade
        self.max_total_risk = max_total_risk  # Risque total maximum autorisé
        self.active_positions = {}  # Suivi des positions actives
        
    async def get_account_balance(self):
        try:
            account_info = self.binance_client.account()
            return float(account_info['totalWalletBalance'])
        except Exception as e:
            raise Exception(f"Erreur lors de la récupération du solde: {str(e)}")
            
    def calculate_position_size(self, signal, account_balance):
        # Calculer le montant à risquer
        risk_amount = account_balance * (self.risk_per_trade / 100)
        
        # Calculer la distance au stop loss
        if signal.direction == "LONG":
            entry_price = signal.entry_min  # Pour un LONG, on utilise le prix d'entrée minimum
            stop_distance = entry_price - signal.stop_loss
        else:  # SHORT
            entry_price = signal.entry_max  # Pour un SHORT, on utilise le prix d'entrée maximum
            stop_distance = signal.stop_loss - entry_price
            
        # Calculer le pourcentage de risque par unité
        risk_per_unit = stop_distance / entry_price
        
        # Calculer la taille de position
        position_size = risk_amount / (entry_price * risk_per_unit)
        
        return position_size, entry_price
        
    def calculate_leverage(self, signal, safety_factor=0.5, max_leverage=20):
        # Calculer la distance au stop loss en pourcentage
        if signal.direction == "LONG":
            entry_price = signal.entry_min
            stop_distance_percent = (entry_price - signal.stop_loss) / entry_price * 100
        else:  # SHORT
            entry_price = signal.entry_max
            stop_distance_percent = (signal.stop_loss - entry_price) / entry_price * 100
            
        # Calculer le levier nécessaire avec marge de sécurité
        required_leverage = 100 / (stop_distance_percent / safety_factor)
        
        # Limiter au levier maximum autorisé
        leverage = min(round(required_leverage), max_leverage)
        
        return leverage
        
    async def check_portfolio_risk(self, new_signal):
        # Calculer le risque actuel du portefeuille
        current_risk = sum(position['risk_percentage'] for position in self.active_positions.values())
        
        # Vérifier si l'ajout du nouveau signal dépasse le risque maximum
        if current_risk + self.risk_per_trade > self.max_total_risk:
            # Calculer le risque restant disponible
            remaining_risk = self.max_total_risk - current_risk
            if remaining_risk <= 0:
                return False, 0  # Pas de risque disponible
            return True, remaining_risk  # Risque partiel disponible
            
        return True, self.risk_per_trade  # Risque complet disponible
        
    def should_move_stop_loss(self, signal_id, current_price):
        # Vérifier si la position existe
        if signal_id not in self.active_positions:
            return False, None
            
        position = self.active_positions[signal_id]
        signal = position['signal']
        
        # Vérifier si le prix a atteint TP2 ou TP3
        if len(signal.take_profit_levels) >= 3:
            tp2_index = 1  # TP2 est le deuxième élément (index 1)
            
            if signal.direction == "LONG" and current_price >= signal.take_profit_levels[tp2_index]:
                return True, signal.entry_min  # Déplacer SL au prix d'entrée minimum
            elif signal.direction == "SHORT" and current_price <= signal.take_profit_levels[tp2_index]:
                return True, signal.entry_max  # Déplacer SL au prix d'entrée maximum
                
        return False, None
```

### 5. Module d'Exécution (Placement des Ordres)

**Objectif**: Exécuter les trades sur Binance Futures selon les signaux validés.

**Composants**:
- **OrderExecutor**: Gère la création et l'envoi des ordres à Binance.
- **OrderSplitter**: Divise les ordres pour les différents niveaux de TP.

**Fonctionnalités**:
- Placement d'ordres d'entrée (market ou limit selon la situation).
- Placement d'ordres take profit à plusieurs niveaux.
- Placement d'ordres stop loss.
- Modification des ordres existants (ex: ajustement du SL).

**Implémentation**:
```python
class OrderExecutor:
    def __init__(self, binance_client, risk_manager):
        self.binance_client = binance_client
        self.risk_manager = risk_manager
        self.active_orders = {}  # Suivi des ordres actifs
        
    async def execute_signal(self, signal, risk_percentage):
        try:
            # Récupérer le solde du compte
            account_balance = await self.risk_manager.get_account_balance()
            
            # Calculer la taille de position et le prix d'entrée
            position_size, entry_price = self.risk_manager.calculate_position_size(signal, account_balance)
            
            # Ajuster la taille de position en fonction du risque disponible
            position_size = position_size * (risk_percentage / self.risk_manager.risk_per_trade)
            
            # Calculer le levier approprié
            leverage = self.risk_manager.calculate_leverage(signal)
            
            # Définir le symbole complet
            symbol = f"{signal.symbol}USDT"
            
            # Définir le levier
            self.binance_client.change_leverage(symbol=symbol, leverage=leverage)
            
            # Déterminer le type d'ordre d'entrée
            current_price = float(self.binance_client.ticker_price(symbol=symbol)['price'])
            
            # Vérifier si le prix actuel est dans la plage d'entrée
            is_in_range = signal.entry_min <= current_price <= signal.entry_max
            
            # Créer l'ordre d'entrée
            entry_orders = []
            
            if is_in_range:
                # Prix actuel dans la plage - utiliser un ordre market
                entry_order = self.binance_client.new_order(
                    symbol=symbol,
                    side="BUY" if signal.direction == "LONG" else "SELL",
                    type="MARKET",
                    quantity=position_size
                )
                entry_orders.append(entry_order)
            else:
                # Prix hors plage - utiliser des ordres limit
                if signal.direction == "LONG" and current_price < signal.entry_min:
                    # Pour un LONG, placer un ordre limit à entry_min
                    entry_order = self.binance_client.new_order(
                        symbol=symbol,
                        side="BUY",
                        type="LIMIT",
                        timeInForce="GTC",
                        quantity=position_size,
                        price=signal.entry_min
                    )
                    entry_orders.append(entry_order)
                elif signal.direction == "SHORT" and current_price > signal.entry_max:
                    # Pour un SHORT, placer un ordre limit à entry_max
                    entry_order = self.binance_client.new_order(
                        symbol=symbol,
                        side="SELL",
                        type="LIMIT",
                        timeInForce="GTC",
                        quantity=position_size,
                        price=signal.entry_max
                    )
                    entry_orders.append(entry_order)
                else:
                    # Prix déjà dépassé la plage d'entrée - signal expiré
                    return {"status": "expired", "message": "Le prix a dépassé la plage d'entrée"}
            
            # Placer l'ordre stop loss
            sl_order = self.binance_client.new_order(
                symbol=symbol,
                side="SELL" if signal.direction == "LONG" else "BUY",
                type="STOP_MARKET",
                closePosition=True,
                stopPrice=signal.stop_loss
            )
            
            # Diviser la position pour les take profits
            tp_orders = self.place_take_profit_orders(signal, symbol, position_size)
            
            # Enregistrer les ordres actifs
            signal_id = f"{symbol}_{signal.direction}_{int(time.time())}"
            self.active_orders[signal_id] = {
                "signal": signal,
                "entry_orders": entry_orders,
                "sl_order": sl_order,
                "tp_orders": tp_orders,
                "position_size": position_size,
                "leverage": leverage,
                "risk_percentage": risk_percentage
            }
            
            # Ajouter à la liste des positions actives du gestionnaire de risque
            self.risk_manager.active_positions[signal_id] = {
                "signal": signal,
                "risk_percentage": risk_percentage
            }
            
            return {"status": "success", "signal_id": signal_id, "orders": self.active_orders[signal_id]}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    def place_take_profit_orders(self, signal, symbol, position_size):
        tp_orders = []
        
        # Nombre de niveaux TP
        num_levels = len(signal.take_profit_levels)
        
        if num_levels == 0:
            return tp_orders
            
        # Répartition de la position entre les niveaux TP
        # Stratégie: Répartition progressive (plus de volume sur les premiers TP)
        weights = []
        for i in range(num_levels):
            # Formule de pondération: plus de poids aux premiers TP
            weight = 1 / (i + 1)
            weights.append(weight)
            
        # Normaliser les poids pour qu'ils totalisent 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Placer les ordres TP
        for i, tp_price in enumerate(signal.take_profit_levels):
            # Calculer la quantité pour ce niveau TP
            tp_quantity = position_size * normalized_weights[i]
            
            # Placer l'ordre TP
            tp_order = self.binance_client.new_order(
                symbol=symbol,
                side="SELL" if signal.direction == "LONG" else "BUY",
                type="LIMIT",
                timeInForce="GTC",
                quantity=tp_quantity,
                price=tp_price
            )
            
            tp_orders.append(tp_order)
            
        return tp_orders
        
    async def update_stop_loss(self, signal_id, new_stop_price):
        if signal_id not in self.active_orders:
            return {"status": "error", "message": "Signal ID non trouvé"}
            
        try:
            order_info = self.active_orders[signal_id]
            symbol = f"{order_info['signal'].symbol}USDT"
            
            # Annuler l'ancien ordre SL
            self.binance_client.cancel_order(
                symbol=symbol,
                orderId=order_info['sl_order']['orderId']
            )
            
            # Placer le nouvel ordre SL
            new_sl_order = self.binance_client.new_order(
                symbol=symbol,
                side="SELL" if order_info['signal'].direction == "LONG" else "BUY",
                type="STOP_MARKET",
                closePosition=True,
                stopPrice=new_stop_price
            )
            
            # Mettre à jour l'information de l'ordre
            order_info['sl_order'] = new_sl_order
            
            return {"status": "success", "new_sl_order": new_sl_order}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
```

### 6. Module Binance (API Futures)

**Objectif**: Gérer la communication avec l'API Binance Futures.

**Composants**:
- **BinanceClient**: Wrapper autour de l'API Binance Futures.
- **APIRateLimiter**: Gère les limites de taux de l'API.

**Fonctionnalités**:
- Authentification sécurisée avec l'API Binance.
- Gestion des requêtes API avec respect des limites de taux.
- Récupération des données de marché.
- Exécution des ordres de trading.

**Implémentation**:
```python
from binance.cm_futures import CMFutures
import time
import asyncio
from functools import wraps

class BinanceModule:
    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialiser le client Binance Futures
        self.client = CMFutures(
            key=api_key,
            secret=api_secret,
            base_url="https://testnet.binancefuture.com" if testnet else None
        )
        
        # Limites de taux
        self.rate_limits = {
            "orders": {"limit": 50, "interval": 10, "current": 0, "last_reset": time.time()},
            "requests": {"limit": 2400, "interval": 60, "current": 0, "last_reset": time.time()}
        }
        
    def rate_limit(self, limit_type="requests"):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Vérifier et mettre à jour les limites de taux
                limit_info = self.rate_limits[limit_type]
                
                # Réinitialiser le compteur si l'intervalle est passé
                current_time = time.time()
                if current_time - limit_info["last_reset"] > limit_info["interval"]:
                    limit_info["current"] = 0
                    limit_info["last_reset"] = current_time
                
                # Vérifier si la limite est atteinte
                if limit_info["current"] >= limit_info["limit"]:
                    # Calculer le temps d'attente
                    wait_time = limit_info["interval"] - (current_time - limit_info["last_reset"])
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        # Réinitialiser après l'attente
                        limit_info["current"] = 0
                        limit_info["last_reset"] = time.time()
                
                # Incrémenter le compteur
                limit_info["current"] += 1
                
                # Exécuter la fonction
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    @rate_limit("requests")
    async def get_account_info(self):
        try:
            return self.client.account()
        except Exception as e:
            raise Exception(f"Erreur lors de la récupération des informations du compte: {str(e)}")
    
    @rate_limit("requests")
    async def get_symbol_price(self, symbol):
        try:
            return self.client.ticker_price(symbol=symbol)
        except Exception as e:
            raise Exception(f"Erreur lors de la récupération du prix: {str(e)}")
    
    @rate_limit("orders")
    async def place_order(self, **params):
        try:
            return self.client.new_order(**params)
        except Exception as e:
            raise Exception(f"Erreur lors du placement de l'ordre: {str(e)}")
    
    @rate_limit("orders")
    async def cancel_order(self, symbol, order_id):
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            raise Exception(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
    
    @rate_limit("requests")
    async def get_open_positions(self):
        try:
            return self.client.get_position_risk()
        except Exception as e:
            raise Exception(f"Erreur lors de la récupération des positions ouvertes: {str(e)}")
    
    @rate_limit("requests")
    async def change_leverage(self, symbol, leverage):
        try:
            return self.client.change_leverage(symbol=symbol, leverage=leverage)
        except Exception as e:
            raise Exception(f"Erreur lors du changement de levier: {str(e)}")
```

### 7. Module de Monitoring (Suivi et Alertes)

**Objectif**: Surveiller les positions ouvertes et générer des alertes.

**Composants**:
- **PositionMonitor**: Surveille l'état des positions ouvertes.
- **AlertGenerator**: Génère des alertes pour les événements importants.

**Fonctionnalités**:
- Suivi en temps réel des positions ouvertes.
- Détection des niveaux de TP atteints.
- Ajustement automatique du SL au point d'équilibre.
- Génération de notifications pour les événements importants.

**Implémentation**:
```python
class MonitoringModule:
    def __init__(self, binance_module, order_executor, risk_manager, telegram_module):
        self.binance_module = binance_module
        self.order_executor = order_executor
        self.risk_manager = risk_manager
        self.telegram_module = telegram_module
        self.running = False
        self.monitor_task = None
        
    async def start_monitoring(self, interval=60):
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        
    async def stop_monitoring(self):
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            
    async def _monitor_loop(self, interval):
        while self.running:
            try:
                await self._check_positions()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Erreur dans la boucle de monitoring: {str(e)}")
                await asyncio.sleep(interval)
                
    async def _check_positions(self):
        # Récupérer toutes les positions ouvertes
        positions = await self.binance_module.get_open_positions()
        
        # Pour chaque signal actif
        for signal_id, order_info in self.order_executor.active_orders.items():
            signal = order_info["signal"]
            symbol = f"{signal.symbol}USDT"
            
            # Vérifier si la position est toujours ouverte
            position_found = False
            for position in positions:
                if position["symbol"] == symbol:
                    position_found = True
                    
                    # Récupérer le prix actuel
                    current_price_info = await self.binance_module.get_symbol_price(symbol)
                    current_price = float(current_price_info["price"])
                    
                    # Vérifier si le SL doit être déplacé au point d'équilibre
                    should_move, new_sl = self.risk_manager.should_move_stop_loss(signal_id, current_price)
                    
                    if should_move and new_sl:
                        # Mettre à jour le stop loss
                        update_result = await self.order_executor.update_stop_loss(signal_id, new_sl)
                        
                        if update_result["status"] == "success":
                            # Envoyer une notification
                            await self._send_notification(
                                f"🔄 Stop Loss déplacé au point d'équilibre pour {symbol} {signal.direction}\n"
                                f"Nouveau Stop Loss: {new_sl}"
                            )
                    break
            
            # Si la position n'est plus ouverte, vérifier si elle a été fermée par TP ou SL
            if not position_found and signal_id in self.risk_manager.active_positions:
                # Supprimer de la liste des positions actives
                del self.risk_manager.active_positions[signal_id]
                
                # Envoyer une notification
                await self._send_notification(
                    f"✅ Position fermée pour {symbol} {signal.direction}\n"
                    f"Signal ID: {signal_id}"
                )
                
    async def _send_notification(self, message):
        # Si un module Telegram est disponible, envoyer la notification
        if hasattr(self.telegram_module, "send_message"):
            await self.telegram_module.send_message(message)
        
        # Toujours afficher dans les logs
        print(message)
```

## Flux de Données et Interactions

### Flux Principal

1. Le **Module Telegram** reçoit un message contenant un signal de trading.
2. Le message est mis en file d'attente et transmis au **Module de Parsing**.
3. Le **Module de Parsing** extrait les informations structurées du signal.
4. Le **Module de Validation** vérifie la validité et la cohérence du signal.
5. Le **Module de Gestion des Risques** calcule la taille de position et le levier.
6. Le **Module d'Exécution** place les ordres via le **Module Binance**.
7. Le **Module de Monitoring** surveille les positions ouvertes et génère des alertes.

### Gestion des Erreurs

- Chaque module implémente sa propre gestion des erreurs.
- Les erreurs sont propagées vers le haut avec des messages descriptifs.
- Les erreurs critiques sont notifiées à l'utilisateur via Telegram.
- Les erreurs non critiques sont journalisées pour analyse ultérieure.

### Persistance des Données

- Les signaux actifs et les positions ouvertes sont stockés en mémoire.
- Une sauvegarde périodique est effectuée pour permettre la reprise après un redémarrage.
- Les journaux détaillés sont maintenus pour l'audit et l'analyse.

## Considérations de Sécurité

### Gestion des Clés API

- Les clés API sont stockées de manière sécurisée (variables d'environnement ou fichier chiffré).
- Les clés API ont des permissions limitées (lecture et trading uniquement, pas de retrait).
- L'authentification utilise HMAC pour une sécurité maximale.

### Validation des Entrées

- Toutes les entrées utilisateur sont validées avant traitement.
- Les messages Telegram sont filtrés pour n'accepter que ceux provenant de sources autorisées.
- Les données extraites sont validées pour leur cohérence et leur conformité.

### Protection contre les Erreurs

- Des limites sont imposées sur la taille des positions et le levier.
- Des mécanismes de retry avec backoff exponentiel sont implémentés pour les appels API.
- Des timeouts sont définis pour éviter les blocages.

## Plan d'Implémentation

### Phase 1: Infrastructure de Base
1. Mise en place du Module Telegram pour la réception des signaux.
2. Implémentation du Module de Parsing pour l'extraction des données.
3. Création du Module Binance pour l'interaction avec l'API.

### Phase 2: Logique Métier
1. Développement du Module de Validation pour la vérification des signaux.
2. Implémentation du Module de Gestion des Risques.
3. Création du Module d'Exécution pour le placement des ordres.

### Phase 3: Monitoring et Améliorations
1. Développement du Module de Monitoring pour le suivi des positions.
2. Implémentation des notifications et alertes.
3. Optimisation des performances et de la fiabilité.

## Conclusion

Cette architecture modulaire permet de créer un bot de trading robuste qui répond à toutes les exigences spécifiées. La séparation claire des responsabilités entre les modules facilite la maintenance et les extensions futures. Les mécanismes de gestion des risques et de monitoring assurent une exécution sécurisée des trades, tandis que l'intégration avec Telegram offre une interface utilisateur intuitive.
