"""
Module d'intégration avec l'API Binance Futures.

Ce module contient les classes et fonctions nécessaires pour interagir
avec l'API Binance Futures, notamment pour récupérer les données de marché
et exécuter des ordres de trading.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps

from binance.cm_futures import CMFutures
from binance.error import ClientError

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Classe pour gérer l'interaction avec l'API Binance Futures.
    
    Cette classe encapsule les fonctionnalités de l'API Binance Futures
    et gère les limites de taux, l'authentification et les erreurs.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialise le client Binance Futures.
        
        Args:
            api_key: La clé API Binance.
            api_secret: Le secret API Binance.
            testnet: Indique si le client doit se connecter au testnet (environnement de test).
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialiser le client Binance Futures
        base_url = "https://testnet.binancefuture.com" if testnet else None
        self.client = CMFutures(
            key=api_key,
            secret=api_secret,
            base_url=base_url
        )
        
        # Limites de taux
        self.rate_limits = {
            "orders": {"limit": 50, "interval": 10, "current": 0, "last_reset": time.time()},
            "requests": {"limit": 2400, "interval": 60, "current": 0, "last_reset": time.time()}
        }
        
        logger.info(f"Client Binance Futures initialisé (testnet: {testnet})")
    
    def rate_limit(self, limit_type: str = "requests"):
        """
        Décorateur pour gérer les limites de taux de l'API.
        
        Args:
            limit_type: Le type de limite à appliquer ("orders" ou "requests").
        """
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
                        logger.warning(f"Limite de taux atteinte pour {limit_type}. Attente de {wait_time:.2f} secondes.")
                        await asyncio.sleep(wait_time)
                        # Réinitialiser après l'attente
                        limit_info["current"] = 0
                        limit_info["last_reset"] = time.time()
                
                # Incrémenter le compteur
                limit_info["current"] += 1
                
                try:
                    # Exécuter la fonction
                    return await func(*args, **kwargs)
                except ClientError as e:
                    # Gérer les erreurs spécifiques à Binance
                    error_code = e.error_code
                    error_message = e.error_message
                    
                    logger.error(f"Erreur Binance: {error_code} - {error_message}")
                    
                    # Gérer les erreurs de limite de taux
                    if error_code == -1003:  # TOO_MANY_REQUESTS
                        logger.warning("Limite de taux dépassée. Attente avant de réessayer.")
                        await asyncio.sleep(5)  # Attendre 5 secondes avant de réessayer
                        return await wrapper(*args, **kwargs)  # Réessayer
                    
                    # Propager l'erreur
                    raise
                except Exception as e:
                    logger.error(f"Erreur lors de l'appel à l'API Binance: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
    
    @rate_limit("requests")
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du compte.
        
        Returns:
            Un dictionnaire contenant les informations du compte.
        """
        try:
            return self.client.account()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations du compte: {str(e)}")
            raise
    
    @rate_limit("requests")
    async def get_symbol_price(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le prix actuel d'un symbole.
        
        Args:
            symbol: Le symbole dont on veut récupérer le prix.
            
        Returns:
            Un dictionnaire contenant le prix du symbole.
        """
        try:
            return self.client.ticker_price(symbol=symbol)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix de {symbol}: {str(e)}")
            raise
    
    @rate_limit("requests")
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur les symboles disponibles.
        
        Returns:
            Un dictionnaire contenant les informations sur les symboles.
        """
        try:
            return self.client.exchange_info()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations d'échange: {str(e)}")
            raise
    
    @rate_limit("orders")
    async def place_order(self, **params) -> Dict[str, Any]:
        """
        Place un ordre sur Binance Futures.
        
        Args:
            **params: Les paramètres de l'ordre.
            
        Returns:
            Un dictionnaire contenant les informations sur l'ordre placé.
        """
        try:
            return self.client.new_order(**params)
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre: {str(e)}")
            raise
    
    @rate_limit("orders")
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Annule un ordre existant.
        
        Args:
            symbol: Le symbole de l'ordre à annuler.
            order_id: L'ID de l'ordre à annuler.
            
        Returns:
            Un dictionnaire contenant les informations sur l'ordre annulé.
        """
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre {order_id} pour {symbol}: {str(e)}")
            raise
    
    @rate_limit("requests")
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Récupère les positions ouvertes.
        
        Returns:
            Une liste de dictionnaires contenant les informations sur les positions ouvertes.
        """
        try:
            return self.client.get_position_risk()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions ouvertes: {str(e)}")
            raise
    
    @rate_limit("requests")
    async def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Modifie le levier pour un symbole.
        
        Args:
            symbol: Le symbole pour lequel modifier le levier.
            leverage: La nouvelle valeur du levier.
            
        Returns:
            Un dictionnaire contenant les informations sur le changement de levier.
        """
        try:
            return self.client.change_leverage(symbol=symbol, leverage=leverage)
        except Exception as e:
            logger.error(f"Erreur lors du changement de levier pour {symbol}: {str(e)}")
            raise
    
    @rate_limit("requests")
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Le symbole pour lequel récupérer les ordres ouverts (optionnel).
            
        Returns:
            Une liste de dictionnaires contenant les informations sur les ordres ouverts.
        """
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            else:
                return self.client.get_open_orders()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            raise
    
    @rate_limit("requests")
    async def get_order_status(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Récupère le statut d'un ordre.
        
        Args:
            symbol: Le symbole de l'ordre.
            order_id: L'ID de l'ordre.
            
        Returns:
            Un dictionnaire contenant les informations sur l'ordre.
        """
        try:
            return self.client.query_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut de l'ordre {order_id} pour {symbol}: {str(e)}")
            raise


class OrderExecutor:
    """
    Classe pour exécuter les ordres de trading sur Binance Futures.
    
    Cette classe gère la création et l'envoi des ordres à Binance Futures
    en fonction des signaux de trading validés.
    """
    
    def __init__(self, binance_client: BinanceClient):
        """
        Initialise l'exécuteur d'ordres.
        
        Args:
            binance_client: Le client Binance Futures à utiliser.
        """
        self.binance_client = binance_client
        self.active_orders = {}  # Suivi des ordres actifs
        
    async def execute_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """
        Exécute un ordre au marché.
        
        Args:
            symbol: Le symbole sur lequel placer l'ordre.
            side: Le côté de l'ordre ("BUY" ou "SELL").
            quantity: La quantité à acheter ou vendre.
            
        Returns:
            Un dictionnaire contenant les informations sur l'ordre exécuté.
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity
        }
        
        logger.info(f"Exécution d'un ordre au marché: {params}")
        return await self.binance_client.place_order(**params)
    
    async def execute_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """
        Exécute un ordre limite.
        
        Args:
            symbol: Le symbole sur lequel placer l'ordre.
            side: Le côté de l'ordre ("BUY" ou "SELL").
            quantity: La quantité à acheter ou vendre.
            price: Le prix limite.
            
        Returns:
            Un dictionnaire contenant les informations sur l'ordre exécuté.
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price
        }
        
        logger.info(f"Exécution d'un ordre limite: {params}")
        return await self.binance_client.place_order(**params)
    
    async def execute_stop_market_order(self, symbol: str, side: str, stop_price: float, close_position: bool = True) -> Dict[str, Any]:
        """
        Exécute un ordre stop market.
        
        Args:
            symbol: Le symbole sur lequel placer l'ordre.
            side: Le côté de l'ordre ("BUY" ou "SELL").
            stop_price: Le prix de déclenchement du stop.
            close_position: Indique si l'ordre doit fermer la position entière.
            
        Returns:
            Un dictionnaire contenant les informations sur l'ordre exécuté.
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "closePosition": close_position
        }
        
        logger.info(f"Exécution d'un ordre stop market: {params}")
        return await self.binance_client.place_order(**params)
    
    async def execute_take_profit_orders(self, symbol: str, side: str, quantities: List[float], prices: List[float]) -> List[Dict[str, Any]]:
        """
        Exécute plusieurs ordres take profit.
        
        Args:
            symbol: Le symbole sur lequel placer les ordres.
            side: Le côté des ordres ("BUY" ou "SELL").
            quantities: Les quantités pour chaque niveau de TP.
            prices: Les prix pour chaque niveau de TP.
            
        Returns:
            Une liste de dictionnaires contenant les informations sur les ordres exécutés.
        """
        if len(quantities) != len(prices):
            raise ValueError("Les listes de quantités et de prix doivent avoir la même longueur")
        
        orders = []
        for i in range(len(prices)):
            params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": quantities[i],
                "price": prices[i]
            }
            
            logger.info(f"Exécution d'un ordre take profit: {params}")
            order = await self.binance_client.place_order(**params)
            orders.append(order)
        
        return orders
    
    async def cancel_order_by_id(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Annule un ordre par son ID.
        
        Args:
            symbol: Le symbole de l'ordre à annuler.
            order_id: L'ID de l'ordre à annuler.
            
        Returns:
            Un dictionnaire contenant les informations sur l'ordre annulé.
        """
        logger.info(f"Annulation de l'ordre {order_id} pour {symbol}")
        return await self.binance_client.cancel_order(symbol, order_id)
    
    async def update_stop_loss(self, symbol: str, old_order_id: int, new_stop_price: float, side: str) -> Dict[str, Any]:
        """
        Met à jour un ordre stop loss.
        
        Args:
            symbol: Le symbole de l'ordre.
            old_order_id: L'ID de l'ancien ordre stop loss.
            new_stop_price: Le nouveau prix stop loss.
            side: Le côté de l'ordre ("BUY" ou "SELL").
            
        Returns:
            Un dictionnaire contenant les informations sur le nouvel ordre stop loss.
        """
        # Annuler l'ancien ordre
        await self.cancel_order_by_id(symbol, old_order_id)
        
        # Créer un nouvel ordre stop loss
        return await self.execute_stop_market_order(symbol, side, new_stop_price)


def main():
    """Fonction principale pour tester le module."""
    import os
    import asyncio
    
    async def test_binance_client():
        # Récupérer les clés API depuis les variables d'environnement
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            print("Les clés API Binance ne sont pas définies dans les variables d'environnement.")
            return
        
        # Créer le client Binance
        client = BinanceClient(api_key, api_secret, testnet=True)
        
        try:
            # Récupérer les informations du compte
            account_info = await client.get_account_info()
            print("Informations du compte:")
            print(f"Balance totale: {account_info['totalWalletBalance']} USDT")
            
            # Récupérer le prix du BTC
            btc_price = await client.get_symbol_price("BTCUSDT")
            print(f"Prix du BTC: {btc_price['price']} USDT")
            
            # Récupérer les positions ouvertes
            positions = await client.get_open_positions()
            print("Positions ouvertes:")
            for position in positions:
                if float(position['positionAmt']) != 0:
                    print(f"{position['symbol']}: {position['positionAmt']} (PnL: {position['unRealizedProfit']})")
            
        except Exception as e:
            print(f"Erreur lors du test: {str(e)}")
    
    # Exécuter le test
    asyncio.run(test_binance_client())


if __name__ == "__main__":
    main()
