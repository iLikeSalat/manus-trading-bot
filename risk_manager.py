"""
Module de gestion des risques pour le bot de trading.

Ce module contient les classes et fonctions nécessaires pour calculer
la taille des positions, le levier approprié, et gérer le risque global
du portefeuille.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from src.signal_parser import TradingSignal
from src.binance_client import BinanceClient

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Représente les informations d'une position active."""
    
    signal_id: str
    signal: TradingSignal
    entry_price: float
    position_size: float
    leverage: int
    risk_percentage: float
    orders: Dict[str, Any]
    timestamp: float


class RiskManager:
    """
    Classe pour gérer le risque des trades.
    
    Cette classe calcule la taille des positions, le levier approprié,
    et gère le risque global du portefeuille.
    """
    
    def __init__(self, binance_client: BinanceClient, risk_per_trade: float = 5.0, max_total_risk: float = 10.0):
        """
        Initialise le gestionnaire de risque.
        
        Args:
            binance_client: Le client Binance Futures à utiliser.
            risk_per_trade: Le pourcentage du portefeuille à risquer par trade.
            max_total_risk: Le risque total maximum autorisé.
        """
        self.binance_client = binance_client
        self.risk_per_trade = risk_per_trade
        self.max_total_risk = max_total_risk
        self.active_positions = {}  # Suivi des positions actives
        
        logger.info(f"Gestionnaire de risque initialisé (risque par trade: {risk_per_trade}%, risque total max: {max_total_risk}%)")
    
    async def get_account_balance(self) -> float:
        """
        Récupère le solde du compte.
        
        Returns:
            Le solde total du portefeuille en USDT.
        """
        try:
            account_info = await self.binance_client.get_account_info()
            balance = float(account_info['totalWalletBalance'])
            logger.info(f"Solde du compte: {balance} USDT")
            return balance
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde: {str(e)}")
            raise
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> Tuple[float, float]:
        """
        Calcule la taille de position appropriée basée sur le risque.
        
        Args:
            signal: Le signal de trading.
            account_balance: Le solde du compte.
            
        Returns:
            Un tuple contenant la taille de position et le prix d'entrée.
        """
        # Calculer le montant à risquer
        risk_amount = account_balance * (self.risk_per_trade / 100)
        logger.info(f"Montant à risquer: {risk_amount} USDT ({self.risk_per_trade}% du solde)")
        
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
        
        logger.info(f"Taille de position calculée: {position_size} {signal.symbol} (prix d'entrée: {entry_price})")
        return position_size, entry_price
    
    def calculate_leverage(self, signal: TradingSignal, safety_factor: float = 0.5, max_leverage: int = 20) -> int:
        """
        Calcule le levier approprié basé sur la distance entre le prix d'entrée et le stop loss.
        
        Args:
            signal: Le signal de trading.
            safety_factor: Le facteur de sécurité (0.5 = 50% de marge).
            max_leverage: Le levier maximum autorisé.
            
        Returns:
            Le levier calculé.
        """
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
        
        logger.info(f"Levier calculé: {leverage}x (distance au SL: {stop_distance_percent:.2f}%)")
        return leverage
    
    async def check_portfolio_risk(self, new_signal: TradingSignal) -> Tuple[bool, float]:
        """
        Vérifie si l'ajout d'un nouveau signal dépasse le risque maximum autorisé.
        
        Args:
            new_signal: Le nouveau signal à évaluer.
            
        Returns:
            Un tuple contenant un booléen indiquant si le signal peut être ajouté
            et le pourcentage de risque disponible.
        """
        # Calculer le risque actuel du portefeuille
        current_risk = sum(position.risk_percentage for position in self.active_positions.values())
        logger.info(f"Risque actuel du portefeuille: {current_risk}%")
        
        # Vérifier si l'ajout du nouveau signal dépasse le risque maximum
        if current_risk + self.risk_per_trade > self.max_total_risk:
            # Calculer le risque restant disponible
            remaining_risk = self.max_total_risk - current_risk
            if remaining_risk <= 0:
                logger.warning("Pas de risque disponible pour un nouveau signal")
                return False, 0  # Pas de risque disponible
            
            logger.warning(f"Risque partiel disponible: {remaining_risk}%")
            return True, remaining_risk  # Risque partiel disponible
        
        logger.info(f"Risque complet disponible: {self.risk_per_trade}%")
        return True, self.risk_per_trade  # Risque complet disponible
    
    def should_move_stop_loss(self, signal_id: str, current_price: float) -> Tuple[bool, Optional[float]]:
        """
        Détermine si le stop loss doit être déplacé au point d'équilibre.
        
        Args:
            signal_id: L'ID du signal.
            current_price: Le prix actuel.
            
        Returns:
            Un tuple contenant un booléen indiquant si le SL doit être déplacé
            et le nouveau prix SL (ou None).
        """
        # Vérifier si la position existe
        if signal_id not in self.active_positions:
            return False, None
        
        position = self.active_positions[signal_id]
        signal = position.signal
        
        # Vérifier si le prix a atteint TP2 ou TP3
        if len(signal.take_profit_levels) >= 3:
            tp2_index = 1  # TP2 est le deuxième élément (index 1)
            
            if signal.direction == "LONG" and current_price >= signal.take_profit_levels[tp2_index]:
                logger.info(f"Prix ({current_price}) a atteint TP2 ({signal.take_profit_levels[tp2_index]}), déplacement du SL au point d'équilibre")
                return True, position.entry_price  # Déplacer SL au prix d'entrée
            elif signal.direction == "SHORT" and current_price <= signal.take_profit_levels[tp2_index]:
                logger.info(f"Prix ({current_price}) a atteint TP2 ({signal.take_profit_levels[tp2_index]}), déplacement du SL au point d'équilibre")
                return True, position.entry_price  # Déplacer SL au prix d'entrée
        
        return False, None
    
    def add_position(self, signal_id: str, signal: TradingSignal, entry_price: float, position_size: float, 
                    leverage: int, risk_percentage: float, orders: Dict[str, Any]) -> None:
        """
        Ajoute une position à la liste des positions actives.
        
        Args:
            signal_id: L'ID unique du signal.
            signal: Le signal de trading.
            entry_price: Le prix d'entrée.
            position_size: La taille de la position.
            leverage: Le levier utilisé.
            risk_percentage: Le pourcentage de risque alloué.
            orders: Les ordres associés à la position.
        """
        self.active_positions[signal_id] = PositionInfo(
            signal_id=signal_id,
            signal=signal,
            entry_price=entry_price,
            position_size=position_size,
            leverage=leverage,
            risk_percentage=risk_percentage,
            orders=orders,
            timestamp=time.time()
        )
        
        logger.info(f"Position ajoutée: {signal_id} ({signal.symbol} {signal.direction})")
    
    def remove_position(self, signal_id: str) -> Optional[PositionInfo]:
        """
        Supprime une position de la liste des positions actives.
        
        Args:
            signal_id: L'ID du signal à supprimer.
            
        Returns:
            Les informations de la position supprimée, ou None si non trouvée.
        """
        if signal_id in self.active_positions:
            position = self.active_positions.pop(signal_id)
            logger.info(f"Position supprimée: {signal_id} ({position.signal.symbol} {position.signal.direction})")
            return position
        
        logger.warning(f"Position non trouvée pour suppression: {signal_id}")
        return None
    
    def calculate_tp_distribution(self, signal: TradingSignal, position_size: float) -> List[float]:
        """
        Calcule la distribution de la taille de position entre les différents niveaux de TP.
        
        Args:
            signal: Le signal de trading.
            position_size: La taille totale de la position.
            
        Returns:
            Une liste contenant les quantités pour chaque niveau de TP.
        """
        # Nombre de niveaux TP
        num_levels = len(signal.take_profit_levels)
        
        if num_levels == 0:
            return []
        
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
        
        # Calculer les quantités pour chaque niveau TP
        quantities = [position_size * w for w in normalized_weights]
        
        logger.info(f"Distribution TP calculée: {quantities} (poids: {normalized_weights})")
        return quantities


def main():
    """Fonction principale pour tester le module."""
    import os
    import asyncio
    from src.binance_client import BinanceClient
    
    async def test_risk_manager():
        # Récupérer les clés API depuis les variables d'environnement
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            print("Les clés API Binance ne sont pas définies dans les variables d'environnement.")
            return
        
        # Créer le client Binance et le gestionnaire de risque
        binance_client = BinanceClient(api_key, api_secret, testnet=True)
        risk_manager = RiskManager(binance_client)
        
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
            # Récupérer le solde du compte
            balance = await risk_manager.get_account_balance()
            print(f"Solde du compte: {balance} USDT")
            
            # Calculer la taille de position
            position_size, entry_price = risk_manager.calculate_position_size(signal, balance)
            print(f"Taille de position: {position_size} BTC (prix d'entrée: {entry_price})")
            
            # Calculer le levier
            leverage = risk_manager.calculate_leverage(signal)
            print(f"Levier: {leverage}x")
            
            # Calculer la distribution TP
            tp_distribution = risk_manager.calculate_tp_distribution(signal, position_size)
            print(f"Distribution TP: {tp_distribution}")
            
            # Vérifier le risque du portefeuille
            can_add, risk_available = await risk_manager.check_portfolio_risk(signal)
            print(f"Peut ajouter le signal: {can_add} (risque disponible: {risk_available}%)")
            
        except Exception as e:
            print(f"Erreur lors du test: {str(e)}")
    
    # Exécuter le test
    asyncio.run(test_risk_manager())


if __name__ == "__main__":
    main()
