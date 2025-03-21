"""
Module de monitoring pour le bot de trading.

Ce module contient les classes et fonctions nécessaires pour surveiller
les positions ouvertes, détecter les événements importants et générer des alertes.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from src.signal_parser import TradingSignal
from src.binance_client import BinanceClient
from src.risk_manager import RiskManager
from src.trade_executor import TradeExecutor

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Classe pour surveiller les positions ouvertes et générer des alertes.
    
    Cette classe surveille en continu les positions ouvertes, détecte les
    événements importants (TP atteints, SL déclenché, etc.) et génère des alertes.
    """
    
    def __init__(self, binance_client: BinanceClient, risk_manager: RiskManager, 
                 trade_executor: TradeExecutor, notification_callback: Optional[Callable] = None):
        """
        Initialise le moniteur de positions.
        
        Args:
            binance_client: Le client Binance Futures à utiliser.
            risk_manager: Le gestionnaire de risque à utiliser.
            trade_executor: L'exécuteur de trades à utiliser.
            notification_callback: Fonction de callback pour envoyer des notifications.
        """
        self.binance_client = binance_client
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.notification_callback = notification_callback
        self.running = False
        self.monitor_task = None
        self.check_interval = 60  # Intervalle de vérification en secondes
        self.position_history = {}  # Historique des positions pour le suivi
        
        logger.info("Moniteur de positions initialisé")
    
    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Démarre la surveillance des positions.
        
        Args:
            interval: L'intervalle de vérification en secondes.
        """
        self.check_interval = interval
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Surveillance des positions démarrée (intervalle: {interval}s)")
    
    async def stop_monitoring(self) -> None:
        """Arrête la surveillance des positions."""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            logger.info("Surveillance des positions arrêtée")
    
    async def _monitor_loop(self) -> None:
        """Boucle principale de surveillance des positions."""
        while self.running:
            try:
                await self._check_positions()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                logger.info("Tâche de surveillance annulée")
                break
            except Exception as e:
                logger.error(f"Erreur dans la boucle de surveillance: {str(e)}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_positions(self) -> None:
        """Vérifie l'état des positions ouvertes."""
        try:
            # Récupérer toutes les positions ouvertes
            positions = await self.binance_client.get_open_positions()
            
            # Créer un dictionnaire des positions actuelles par symbole
            current_positions = {}
            for position in positions:
                if float(position["positionAmt"]) != 0:  # Position non nulle
                    current_positions[position["symbol"]] = position
            
            # Récupérer les trades actifs
            active_trades = self.trade_executor.active_trades
            
            # Pour chaque trade actif
            for signal_id, trade_info in list(active_trades.items()):
                signal = trade_info["signal"]
                symbol = f"{signal.symbol}USDT"
                
                # Vérifier si la position est toujours ouverte
                if symbol in current_positions:
                    position = current_positions[symbol]
                    
                    # Récupérer le prix actuel
                    current_price_info = await self.binance_client.get_symbol_price(symbol)
                    current_price = float(current_price_info["price"])
                    
                    # Vérifier si le SL doit être déplacé au point d'équilibre
                    should_move, new_sl = self.risk_manager.should_move_stop_loss(signal_id, current_price)
                    
                    if should_move and new_sl:
                        # Mettre à jour le stop loss
                        update_result = await self.trade_executor.update_stop_loss(signal_id, new_sl)
                        
                        if update_result["status"] == "success":
                            # Envoyer une notification
                            await self._send_notification(
                                f"🔄 Stop Loss déplacé au point d'équilibre pour {symbol} {signal.direction}\n"
                                f"Nouveau Stop Loss: {new_sl}"
                            )
                    
                    # Vérifier si des TP ont été atteints
                    self._check_tp_levels(signal_id, signal, current_price)
                    
                    # Enregistrer l'historique de la position pour le suivi
                    if signal_id not in self.position_history:
                        self.position_history[signal_id] = []
                    
                    self.position_history[signal_id].append({
                        "timestamp": time.time(),
                        "price": current_price,
                        "pnl": float(position["unRealizedProfit"])
                    })
                    
                else:
                    # La position n'est plus ouverte, vérifier si elle a été fermée par TP ou SL
                    if signal_id in self.risk_manager.active_positions:
                        # Récupérer l'historique de la position
                        history = self.position_history.get(signal_id, [])
                        
                        # Déterminer si la position a été fermée par TP ou SL
                        close_reason = self._determine_close_reason(signal, history)
                        
                        # Supprimer de la liste des positions actives
                        self.risk_manager.remove_position(signal_id)
                        
                        # Supprimer des trades actifs
                        if signal_id in active_trades:
                            del active_trades[signal_id]
                        
                        # Envoyer une notification
                        await self._send_notification(
                            f"✅ Position fermée pour {symbol} {signal.direction}\n"
                            f"Raison: {close_reason}\n"
                            f"Signal ID: {signal_id}"
                        )
                        
                        # Nettoyer l'historique
                        if signal_id in self.position_history:
                            del self.position_history[signal_id]
        
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des positions: {str(e)}")
    
    def _check_tp_levels(self, signal_id: str, signal: TradingSignal, current_price: float) -> None:
        """
        Vérifie si des niveaux de TP ont été atteints.
        
        Args:
            signal_id: L'ID du signal.
            signal: Le signal de trading.
            current_price: Le prix actuel.
        """
        # Récupérer l'historique de la position
        history = self.position_history.get(signal_id, [])
        
        # Si c'est la première vérification, initialiser l'historique des TP
        if not history or "tp_reached" not in history[-1]:
            tp_reached = [False] * len(signal.take_profit_levels)
        else:
            tp_reached = history[-1]["tp_reached"]
        
        # Vérifier chaque niveau de TP
        for i, tp_level in enumerate(signal.take_profit_levels):
            if not tp_reached[i]:  # Si ce niveau n'a pas encore été atteint
                if (signal.direction == "LONG" and current_price >= tp_level) or \
                   (signal.direction == "SHORT" and current_price <= tp_level):
                    # TP atteint
                    tp_reached[i] = True
                    
                    # Envoyer une notification
                    asyncio.create_task(self._send_notification(
                        f"🎯 Take Profit {i+1} atteint pour {signal.symbol}USDT {signal.direction}\n"
                        f"Niveau: {tp_level}\n"
                        f"Prix actuel: {current_price}\n"
                        f"Signal ID: {signal_id}"
                    ))
        
        # Mettre à jour l'historique
        if history:
            history[-1]["tp_reached"] = tp_reached
    
    def _determine_close_reason(self, signal: TradingSignal, history: List[Dict[str, Any]]) -> str:
        """
        Détermine la raison de la fermeture d'une position.
        
        Args:
            signal: Le signal de trading.
            history: L'historique de la position.
            
        Returns:
            La raison de la fermeture.
        """
        if not history:
            return "Inconnue"
        
        last_price = history[-1]["price"]
        
        # Vérifier si le dernier prix est proche du SL
        sl_threshold = 0.2  # 0.2% de marge
        sl_distance_percent = abs(last_price - signal.stop_loss) / signal.stop_loss * 100
        
        if sl_distance_percent <= sl_threshold:
            return "Stop Loss déclenché"
        
        # Vérifier si le dernier prix est proche d'un TP
        for i, tp_level in enumerate(signal.take_profit_levels):
            tp_threshold = 0.2  # 0.2% de marge
            tp_distance_percent = abs(last_price - tp_level) / tp_level * 100
            
            if tp_distance_percent <= tp_threshold:
                return f"Take Profit {i+1} atteint"
        
        return "Fermeture manuelle ou autre raison"
    
    async def _send_notification(self, message: str) -> None:
        """
        Envoie une notification.
        
        Args:
            message: Le message à envoyer.
        """
        logger.info(f"Notification: {message}")
        
        # Si un callback de notification est disponible, l'utiliser
        if self.notification_callback:
            try:
                await self.notification_callback(message)
            except Exception as e:
                logger.error(f"Erreur lors de l'envoi de la notification: {str(e)}")
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """
        Génère un rapport de performance des trades.
        
        Returns:
            Un dictionnaire contenant les statistiques de performance.
        """
        try:
            # Récupérer l'historique des trades
            trade_history = await self._get_trade_history()
            
            # Calculer les statistiques
            total_trades = len(trade_history)
            winning_trades = sum(1 for trade in trade_history if trade["pnl"] > 0)
            losing_trades = sum(1 for trade in trade_history if trade["pnl"] < 0)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            total_profit = sum(trade["pnl"] for trade in trade_history if trade["pnl"] > 0)
            total_loss = sum(trade["pnl"] for trade in trade_history if trade["pnl"] < 0)
            
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            # Créer le rapport
            report = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_profit": total_profit + total_loss,
                "profit_factor": profit_factor,
                "trades": trade_history,
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport de performance: {str(e)}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    async def _get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des trades.
        
        Returns:
            Une liste de dictionnaires contenant les informations des trades.
        """
        # Cette méthode devrait être implémentée pour récupérer l'historique des trades
        # depuis Binance ou une base de données locale
        
        # Pour l'instant, nous retournons un exemple d'historique
        return [
            {
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 50000.0,
                "exit_price": 52000.0,
                "pnl": 2000.0,
                "close_reason": "Take Profit 1 atteint",
                "timestamp": time.time() - 3600
            },
            {
                "symbol": "ETHUSDT",
                "direction": "SHORT",
                "entry_price": 3000.0,
                "exit_price": 3100.0,
                "pnl": -100.0,
                "close_reason": "Stop Loss déclenché",
                "timestamp": time.time() - 7200
            }
        ]


class RiskAnalyzer:
    """
    Classe pour analyser et optimiser les paramètres de risque.
    
    Cette classe analyse les performances passées et ajuste les paramètres
    de risque pour optimiser les résultats futurs.
    """
    
    def __init__(self, risk_manager: RiskManager, position_monitor: PositionMonitor):
        """
        Initialise l'analyseur de risque.
        
        Args:
            risk_manager: Le gestionnaire de risque à utiliser.
            position_monitor: Le moniteur de positions à utiliser.
        """
        self.risk_manager = risk_manager
        self.position_monitor = position_monitor
        
        logger.info("Analyseur de risque initialisé")
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyse les performances des trades et génère des recommandations.
        
        Returns:
            Un dictionnaire contenant les recommandations.
        """
        try:
            # Récupérer le rapport de performance
            performance = await self.position_monitor.generate_performance_report()
            
            # Analyser les performances
            recommendations = {}
            
            # Analyser le taux de réussite
            win_rate = performance.get("win_rate", 0)
            if win_rate < 40:
                recommendations["risk_per_trade"] = "Réduire le risque par trade à 3%"
            elif win_rate > 60:
                recommendations["risk_per_trade"] = "Maintenir le risque par trade à 5%"
            
            # Analyser le facteur de profit
            profit_factor = performance.get("profit_factor", 0)
            if profit_factor < 1.5:
                recommendations["tp_distribution"] = "Ajuster la distribution des TP pour prendre des profits plus tôt"
            elif profit_factor > 2.5:
                recommendations["tp_distribution"] = "Maintenir la distribution actuelle des TP"
            
            # Analyser le ratio profit/perte
            total_profit = performance.get("total_profit", 0)
            total_loss = performance.get("total_loss", 0)
            
            if total_loss != 0:
                avg_profit_loss_ratio = (total_profit / winning_trades) / (abs(total_loss) / losing_trades) \
                    if winning_trades > 0 and losing_trades > 0 else 0
                
                if avg_profit_loss_ratio < 1.5:
                    recommendations["stop_loss"] = "Réduire la distance du SL pour limiter les pertes"
                elif avg_profit_loss_ratio > 2.5:
                    recommendations["stop_loss"] = "Maintenir la stratégie actuelle de SL"
            
            return {
                "performance": performance,
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des performances: {str(e)}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    async def optimize_risk_parameters(self) -> Dict[str, Any]:
        """
        Optimise les paramètres de risque en fonction des performances passées.
        
        Returns:
            Un dictionnaire contenant les paramètres optimisés.
        """
        try:
            # Analyser les performances
            analysis = await self.analyze_performance()
            
            # Récupérer les recommandations
            recommendations = analysis.get("recommendations", {})
            
            # Paramètres actuels
            current_params = {
                "risk_per_trade": self.risk_manager.risk_per_trade,
                "max_total_risk": self.risk_manager.max_total_risk
            }
            
            # Paramètres optimisés
            optimized_params = current_params.copy()
            
            # Appliquer les recommandations
            if "risk_per_trade" in recommendations:
                if "Réduire" in recommendations["risk_per_trade"]:
                    optimized_params["risk_per_trade"] = 3.0
                elif "Augmenter" in recommendations["risk_per_trade"]:
                    optimized_params["risk_per_trade"] = 7.0
            
            # Autres optimisations possibles...
            
            return {
                "current_params": current_params,
                "optimized_params": optimized_params,
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation des paramètres de risque: {str(e)}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }


def main():
    """Fonction principale pour tester le module."""
    import os
    import asyncio
    from src.binance_client import BinanceClient
    from src.risk_manager import RiskManager
    from src.trade_executor import TradeExecutor
    
    async def test_position_monitor():
        # Récupérer les clés API depuis les variables d'environnement
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            print("Les clés API Binance ne sont pas définies dans les variables d'environnement.")
            return
        
        # Créer les composants nécessaires
        binance_client = BinanceClient(api_key, api_secret, testnet=True)
        risk_manager = RiskManager(binance_client)
        trade_executor = TradeExecutor(binance_client, risk_manager)
        
        # Fonction de callback pour les notifications
        async def notification_callback(message):
            print(f"NOTIFICATION: {message}")
        
        # Créer le moniteur de positions
        position_monitor = PositionMonitor(
            binance_client=binance_client,
            risk_manager=risk_manager,
            trade_executor=trade_executor,
            notification_callback=notification_callback
        )
        
        # Créer l'analyseur de risque
        risk_analyzer = RiskAnalyzer(risk_manager, position_monitor)
        
        try:
            # Démarrer la surveillance
            await position_monitor.start_monitoring(interval=10)
            
            # Attendre un peu pour voir les résultats
            print("Surveillance démarrée. Attente de 30 secondes...")
            await asyncio.sleep(30)
            
            # Générer un rapport de performance
            performance = await position_monitor.generate_performance_report()
            print("Rapport de performance:")
            print(performance)
            
            # Analyser les performances
            analysis = await risk_analyzer.analyze_performance()
            print("Analyse des performances:")
            print(analysis)
            
            # Optimiser les paramètres de risque
            optimization = await risk_analyzer.optimize_risk_parameters()
            print("Optimisation des paramètres de risque:")
            print(optimization)
            
            # Arrêter la surveillance
            await position_monitor.stop_monitoring()
            
        except Exception as e:
            print(f"Erreur lors du test: {str(e)}")
    
    # Exécuter le test
    asyncio.run(test_position_monitor())


if __name__ == "__main__":
    main()
