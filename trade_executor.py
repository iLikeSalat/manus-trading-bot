"""
Module d'exécution des trades pour le bot de trading.

Ce module contient les classes et fonctions nécessaires pour exécuter
les trades sur Binance Futures en fonction des signaux reçus.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple

from src.signal_parser import TradingSignal
from src.binance_client import BinanceClient, OrderExecutor
from src.risk_manager import RiskManager

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Classe pour exécuter les trades en fonction des signaux reçus.
    
    Cette classe intègre le client Binance, l'exécuteur d'ordres et le gestionnaire
    de risque pour exécuter les trades de manière complète et sécurisée.
    """
    
    def __init__(self, binance_client: BinanceClient, risk_manager: RiskManager):
        """
        Initialise l'exécuteur de trades.
        
        Args:
            binance_client: Le client Binance Futures à utiliser.
            risk_manager: Le gestionnaire de risque à utiliser.
        """
        self.binance_client = binance_client
        self.risk_manager = risk_manager
        self.order_executor = OrderExecutor(binance_client)
        self.active_trades = {}  # Suivi des trades actifs
        
        logger.info("Exécuteur de trades initialisé")
    
    async def execute_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Exécute un signal de trading complet.
        
        Cette méthode gère tout le processus d'exécution d'un signal:
        - Vérification du risque
        - Calcul de la taille de position et du levier
        - Placement des ordres d'entrée, de TP et de SL
        
        Args:
            signal: Le signal de trading à exécuter.
            
        Returns:
            Un dictionnaire contenant les informations sur l'exécution du signal.
        """
        try:
            # Générer un ID unique pour le signal
            signal_id = f"{signal.symbol}_{signal.direction}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Exécution du signal {signal_id}: {signal.symbol} {signal.direction}")
            
            # Vérifier si le symbole existe sur Binance Futures
            symbol = f"{signal.symbol}USDT"
            try:
                exchange_info = await self.binance_client.get_exchange_info()
                symbols = [info['symbol'] for info in exchange_info['symbols']]
                
                if symbol not in symbols:
                    logger.error(f"Symbole {symbol} non disponible sur Binance Futures")
                    return {
                        "status": "error",
                        "message": f"Symbole {symbol} non disponible sur Binance Futures"
                    }
            except Exception as e:
                logger.error(f"Erreur lors de la vérification du symbole: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Erreur lors de la vérification du symbole: {str(e)}"
                }
            
            # Vérifier le risque du portefeuille
            can_add, risk_percentage = await self.risk_manager.check_portfolio_risk(signal)
            
            if not can_add:
                logger.warning(f"Impossible d'ajouter le signal {signal_id}: risque maximum atteint")
                return {
                    "status": "rejected",
                    "message": "Risque maximum du portefeuille atteint"
                }
            
            # Récupérer le solde du compte
            account_balance = await self.risk_manager.get_account_balance()
            
            # Calculer la taille de position et le prix d'entrée
            position_size, entry_price = self.risk_manager.calculate_position_size(signal, account_balance)
            
            # Ajuster la taille de position en fonction du risque disponible
            if risk_percentage < self.risk_manager.risk_per_trade:
                position_size = position_size * (risk_percentage / self.risk_manager.risk_per_trade)
                logger.info(f"Taille de position ajustée à {position_size} {signal.symbol} (risque: {risk_percentage}%)")
            
            # Calculer le levier approprié
            leverage = self.risk_manager.calculate_leverage(signal)
            
            # Définir le levier
            await self.binance_client.change_leverage(symbol, leverage)
            
            # Récupérer le prix actuel
            current_price_info = await self.binance_client.get_symbol_price(symbol)
            current_price = float(current_price_info['price'])
            
            # Déterminer le type d'ordre d'entrée
            entry_orders = []
            is_in_range = signal.entry_min <= current_price <= signal.entry_max
            
            if is_in_range:
                # Prix actuel dans la plage - utiliser un ordre market
                logger.info(f"Prix actuel ({current_price}) dans la plage d'entrée, utilisation d'un ordre market")
                entry_order = await self.order_executor.execute_market_order(
                    symbol=symbol,
                    side="BUY" if signal.direction == "LONG" else "SELL",
                    quantity=position_size
                )
                entry_orders.append(entry_order)
            else:
                # Prix hors plage - utiliser des ordres limit
                if signal.direction == "LONG" and current_price < signal.entry_min:
                    # Pour un LONG, placer un ordre limit à entry_min
                    logger.info(f"Prix actuel ({current_price}) sous la plage d'entrée, utilisation d'un ordre limit à {signal.entry_min}")
                    entry_order = await self.order_executor.execute_limit_order(
                        symbol=symbol,
                        side="BUY",
                        quantity=position_size,
                        price=signal.entry_min
                    )
                    entry_orders.append(entry_order)
                elif signal.direction == "SHORT" and current_price > signal.entry_max:
                    # Pour un SHORT, placer un ordre limit à entry_max
                    logger.info(f"Prix actuel ({current_price}) au-dessus de la plage d'entrée, utilisation d'un ordre limit à {signal.entry_max}")
                    entry_order = await self.order_executor.execute_limit_order(
                        symbol=symbol,
                        side="SELL",
                        quantity=position_size,
                        price=signal.entry_max
                    )
                    entry_orders.append(entry_order)
                else:
                    # Prix déjà dépassé la plage d'entrée - signal expiré
                    logger.warning(f"Prix actuel ({current_price}) a dépassé la plage d'entrée, signal expiré")
                    return {
                        "status": "expired",
                        "message": "Le prix a dépassé la plage d'entrée"
                    }
            
            # Placer l'ordre stop loss
            sl_order = await self.order_executor.execute_stop_market_order(
                symbol=symbol,
                side="SELL" if signal.direction == "LONG" else "BUY",
                stop_price=signal.stop_loss,
                close_position=True
            )
            
            # Calculer la distribution des TP
            tp_quantities = self.risk_manager.calculate_tp_distribution(signal, position_size)
            
            # Placer les ordres take profit
            tp_orders = await self.order_executor.execute_take_profit_orders(
                symbol=symbol,
                side="SELL" if signal.direction == "LONG" else "BUY",
                quantities=tp_quantities,
                prices=signal.take_profit_levels
            )
            
            # Enregistrer les ordres actifs
            orders = {
                "entry_orders": entry_orders,
                "sl_order": sl_order,
                "tp_orders": tp_orders
            }
            
            # Ajouter à la liste des positions actives du gestionnaire de risque
            self.risk_manager.add_position(
                signal_id=signal_id,
                signal=signal,
                entry_price=entry_price,
                position_size=position_size,
                leverage=leverage,
                risk_percentage=risk_percentage,
                orders=orders
            )
            
            # Ajouter aux trades actifs
            self.active_trades[signal_id] = {
                "signal": signal,
                "orders": orders,
                "status": "active",
                "timestamp": time.time()
            }
            
            logger.info(f"Signal {signal_id} exécuté avec succès")
            return {
                "status": "success",
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": signal.direction,
                "position_size": position_size,
                "leverage": leverage,
                "risk_percentage": risk_percentage,
                "orders": orders
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur lors de l'exécution du signal: {str(e)}"
            }
    
    async def update_stop_loss(self, signal_id: str, new_stop_price: float) -> Dict[str, Any]:
        """
        Met à jour le stop loss d'un trade actif.
        
        Args:
            signal_id: L'ID du signal.
            new_stop_price: Le nouveau prix stop loss.
            
        Returns:
            Un dictionnaire contenant les informations sur la mise à jour.
        """
        try:
            # Vérifier si le trade existe
            if signal_id not in self.active_trades:
                logger.warning(f"Trade {signal_id} non trouvé pour la mise à jour du SL")
                return {
                    "status": "error",
                    "message": f"Trade {signal_id} non trouvé"
                }
            
            # Récupérer les informations du trade
            trade_info = self.active_trades[signal_id]
            signal = trade_info["signal"]
            symbol = f"{signal.symbol}USDT"
            
            # Récupérer l'ordre SL actuel
            sl_order = trade_info["orders"]["sl_order"]
            
            # Déterminer le côté de l'ordre SL
            sl_side = "SELL" if signal.direction == "LONG" else "BUY"
            
            # Mettre à jour le SL
            new_sl_order = await self.order_executor.update_stop_loss(
                symbol=symbol,
                old_order_id=sl_order["orderId"],
                new_stop_price=new_stop_price,
                side=sl_side
            )
            
            # Mettre à jour les informations du trade
            trade_info["orders"]["sl_order"] = new_sl_order
            
            logger.info(f"Stop loss mis à jour pour {signal_id}: {new_stop_price}")
            return {
                "status": "success",
                "signal_id": signal_id,
                "new_sl_order": new_sl_order
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du stop loss: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur lors de la mise à jour du stop loss: {str(e)}"
            }
    
    async def cancel_trade(self, signal_id: str) -> Dict[str, Any]:
        """
        Annule un trade actif.
        
        Args:
            signal_id: L'ID du signal à annuler.
            
        Returns:
            Un dictionnaire contenant les informations sur l'annulation.
        """
        try:
            # Vérifier si le trade existe
            if signal_id not in self.active_trades:
                logger.warning(f"Trade {signal_id} non trouvé pour l'annulation")
                return {
                    "status": "error",
                    "message": f"Trade {signal_id} non trouvé"
                }
            
            # Récupérer les informations du trade
            trade_info = self.active_trades[signal_id]
            signal = trade_info["signal"]
            symbol = f"{signal.symbol}USDT"
            
            # Annuler tous les ordres associés
            cancelled_orders = []
            
            # Annuler les ordres d'entrée (s'ils sont encore actifs)
            for entry_order in trade_info["orders"]["entry_orders"]:
                try:
                    cancelled = await self.binance_client.cancel_order(symbol, entry_order["orderId"])
                    cancelled_orders.append(cancelled)
                except Exception as e:
                    logger.warning(f"Erreur lors de l'annulation de l'ordre d'entrée: {str(e)}")
            
            # Annuler l'ordre SL
            try:
                cancelled = await self.binance_client.cancel_order(symbol, trade_info["orders"]["sl_order"]["orderId"])
                cancelled_orders.append(cancelled)
            except Exception as e:
                logger.warning(f"Erreur lors de l'annulation de l'ordre SL: {str(e)}")
            
            # Annuler les ordres TP
            for tp_order in trade_info["orders"]["tp_orders"]:
                try:
                    cancelled = await self.binance_client.cancel_order(symbol, tp_order["orderId"])
                    cancelled_orders.append(cancelled)
                except Exception as e:
                    logger.warning(f"Erreur lors de l'annulation de l'ordre TP: {str(e)}")
            
            # Supprimer le trade des listes actives
            del self.active_trades[signal_id]
            self.risk_manager.remove_position(signal_id)
            
            logger.info(f"Trade {signal_id} annulé avec succès")
            return {
                "status": "success",
                "signal_id": signal_id,
                "cancelled_orders": cancelled_orders
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation du trade: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur lors de l'annulation du trade: {str(e)}"
            }
    
    async def get_trade_status(self, signal_id: str) -> Dict[str, Any]:
        """
        Récupère le statut d'un trade.
        
        Args:
            signal_id: L'ID du signal.
            
        Returns:
            Un dictionnaire contenant les informations sur le statut du trade.
        """
        try:
            # Vérifier si le trade existe
            if signal_id not in self.active_trades:
                logger.warning(f"Trade {signal_id} non trouvé pour le statut")
                return {
                    "status": "error",
                    "message": f"Trade {signal_id} non trouvé"
                }
            
            # Récupérer les informations du trade
            trade_info = self.active_trades[signal_id]
            signal = trade_info["signal"]
            symbol = f"{signal.symbol}USDT"
            
            # Récupérer les positions ouvertes
            positions = await self.binance_client.get_open_positions()
            position_found = False
            position_info = None
            
            for position in positions:
                if position["symbol"] == symbol:
                    position_found = True
                    position_info = position
                    break
            
            # Récupérer les ordres ouverts
            open_orders = await self.binance_client.get_open_orders(symbol)
            
            # Préparer la réponse
            response = {
                "status": "active" if position_found else "pending",
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": signal.direction,
                "position": position_info,
                "open_orders": open_orders,
                "trade_info": trade_info
            }
            
            logger.info(f"Statut du trade {signal_id} récupéré: {response['status']}")
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut du trade: {str(e)}")
            return {
                "status": "error",
                "message": f"Erreur lors de la récupération du statut du trade: {str(e)}"
            }


def main():
    """Fonction principale pour tester le module."""
    import os
    import asyncio
    from src.binance_client import BinanceClient
    from src.risk_manager import RiskManager
    from src.signal_parser import TradingSignal
    
    async def test_trade_executor():
        # Récupérer les clés API depuis les variables d'environnement
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            print("Les clés API Binance ne sont pas définies dans les variables d'environnement.")
            return
        
        # Créer le client Binance, le gestionnaire de risque et l'exécuteur de trades
        binance_client = BinanceClient(api_key, api_secret, testnet=True)
        risk_manager = RiskManager(binance_client)
        trade_executor = TradeExecutor(binance_client, risk_manager)
        
        # Créer un signal de test
        signal = TradingSignal(
            symbol="BTC",
            direction="LONG",
            entry_min=50000.0,
            entry_max=51000.0,
            take_profit_levels=[52000.0, 53000.0, 54000.0, 55000.0],
            stop_loss=49000.0,
            raw_text="Signal de test"
        )
        
        try:
            # Exécuter le signal
            result = await trade_executor.execute_signal(signal)
            print(f"Résultat de l'exécution: {result}")
            
            if result["status"] == "success":
                # Récupérer le statut du trade
                signal_id = result["signal_id"]
                status = await trade_executor.get_trade_status(signal_id)
                print(f"Statut du trade: {status}")
                
                # Mettre à jour le stop loss
                update_result = await trade_executor.update_stop_loss(signal_id, 49500.0)
                print(f"Résultat de la mise à jour du SL: {update_result}")
                
                # Annuler le trade
                cancel_result = await trade_executor.cancel_trade(signal_id)
                print(f"Résultat de l'annulation: {cancel_result}")
            
        except Exception as e:
            print(f"Erreur lors du test: {str(e)}")
    
    # Exécuter le test
    asyncio.run(test_trade_executor())


if __name__ == "__main__":
    main()
