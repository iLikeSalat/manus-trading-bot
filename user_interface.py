"""
Module d'interface utilisateur pour le bot de trading.

Ce module contient les classes et fonctions nécessaires pour créer
une interface utilisateur en ligne de commande et un tableau de bord
de monitoring pour le bot de trading.
"""

import asyncio
import logging
import time
import os
import sys
import argparse
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import threading
import curses
from tabulate import tabulate

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_parser import TradingSignal
from src.binance_client import BinanceClient
from src.risk_manager import RiskManager
from src.trade_executor import TradeExecutor
from src.position_monitor import PositionMonitor, RiskAnalyzer

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='trading_bot.log'  # Rediriger les logs vers un fichier
)
logger = logging.getLogger(__name__)


class CommandLineInterface:
    """
    Interface en ligne de commande pour le bot de trading.
    
    Cette classe fournit une interface utilisateur en ligne de commande
    pour interagir avec le bot de trading.
    """
    
    def __init__(self, trade_executor: TradeExecutor, position_monitor: PositionMonitor,
                 risk_analyzer: RiskAnalyzer):
        """
        Initialise l'interface en ligne de commande.
        
        Args:
            trade_executor: L'exécuteur de trades à utiliser.
            position_monitor: Le moniteur de positions à utiliser.
            risk_analyzer: L'analyseur de risque à utiliser.
        """
        self.trade_executor = trade_executor
        self.position_monitor = position_monitor
        self.risk_analyzer = risk_analyzer
        self.running = False
        self.commands = {
            "help": self._cmd_help,
            "status": self._cmd_status,
            "positions": self._cmd_positions,
            "trades": self._cmd_trades,
            "performance": self._cmd_performance,
            "risk": self._cmd_risk,
            "cancel": self._cmd_cancel,
            "signal": self._cmd_signal,
            "config": self._cmd_config,
            "exit": self._cmd_exit
        }
        
        logger.info("Interface en ligne de commande initialisée")
    
    async def start(self) -> None:
        """Démarre l'interface en ligne de commande."""
        self.running = True
        
        print("\n=== Bot de Trading Telegram-Binance Futures ===")
        print("Tapez 'help' pour voir les commandes disponibles.")
        
        while self.running:
            try:
                command = input("\n> ").strip()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                args = parts[1:]
                
                if cmd in self.commands:
                    await self.commands[cmd](args)
                else:
                    print(f"Commande inconnue: {cmd}")
                    print("Tapez 'help' pour voir les commandes disponibles.")
            
            except KeyboardInterrupt:
                print("\nArrêt de l'interface...")
                self.running = False
            except Exception as e:
                print(f"Erreur: {str(e)}")
                logger.error(f"Erreur dans l'interface: {str(e)}")
    
    async def _cmd_help(self, args: List[str]) -> None:
        """Affiche l'aide des commandes."""
        print("\nCommandes disponibles:")
        print("  help                  - Affiche cette aide")
        print("  status                - Affiche le statut général du bot")
        print("  positions             - Affiche les positions ouvertes")
        print("  trades                - Affiche les trades actifs")
        print("  performance           - Affiche les statistiques de performance")
        print("  risk                  - Affiche les paramètres de risque")
        print("  cancel <signal_id>    - Annule un trade actif")
        print("  signal <message>      - Traite un signal manuellement")
        print("  config                - Affiche ou modifie la configuration")
        print("  exit                  - Quitte l'interface")
    
    async def _cmd_status(self, args: List[str]) -> None:
        """Affiche le statut général du bot."""
        try:
            # Récupérer le solde du compte
            account_balance = await self.trade_executor.risk_manager.get_account_balance()
            
            # Récupérer les positions ouvertes
            positions = await self.trade_executor.binance_client.get_open_positions()
            active_positions = [p for p in positions if float(p["positionAmt"]) != 0]
            
            # Récupérer les trades actifs
            active_trades = len(self.trade_executor.active_trades)
            
            # Afficher le statut
            print("\n=== Statut du Bot ===")
            print(f"Solde du compte: {account_balance} USDT")
            print(f"Positions ouvertes: {len(active_positions)}")
            print(f"Trades actifs: {active_trades}")
            print(f"Risque par trade: {self.trade_executor.risk_manager.risk_per_trade}%")
            print(f"Risque total maximum: {self.trade_executor.risk_manager.max_total_risk}%")
            
            # Calculer le risque actuel
            current_risk = sum(position.risk_percentage for position in self.trade_executor.risk_manager.active_positions.values())
            print(f"Risque actuel: {current_risk:.2f}%")
            
        except Exception as e:
            print(f"Erreur lors de la récupération du statut: {str(e)}")
            logger.error(f"Erreur lors de la récupération du statut: {str(e)}")
    
    async def _cmd_positions(self, args: List[str]) -> None:
        """Affiche les positions ouvertes."""
        try:
            # Récupérer les positions ouvertes
            positions = await self.trade_executor.binance_client.get_open_positions()
            active_positions = [p for p in positions if float(p["positionAmt"]) != 0]
            
            if not active_positions:
                print("\nAucune position ouverte.")
                return
            
            # Préparer les données pour l'affichage
            table_data = []
            for position in active_positions:
                symbol = position["symbol"]
                side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
                entry_price = float(position["entryPrice"])
                mark_price = float(position["markPrice"])
                position_amt = abs(float(position["positionAmt"]))
                leverage = int(position["leverage"])
                unrealized_pnl = float(position["unRealizedProfit"])
                roe = unrealized_pnl / (entry_price * position_amt / leverage) * 100 if position_amt > 0 else 0
                
                table_data.append([
                    symbol,
                    side,
                    f"{entry_price:.2f}",
                    f"{mark_price:.2f}",
                    f"{position_amt:.6f}",
                    f"{leverage}x",
                    f"{unrealized_pnl:.2f} USDT",
                    f"{roe:.2f}%"
                ])
            
            # Afficher le tableau
            headers = ["Symbole", "Direction", "Prix d'entrée", "Prix actuel", "Quantité", "Levier", "PnL", "ROE"]
            print("\n=== Positions Ouvertes ===")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
        except Exception as e:
            print(f"Erreur lors de la récupération des positions: {str(e)}")
            logger.error(f"Erreur lors de la récupération des positions: {str(e)}")
    
    async def _cmd_trades(self, args: List[str]) -> None:
        """Affiche les trades actifs."""
        try:
            # Récupérer les trades actifs
            active_trades = self.trade_executor.active_trades
            
            if not active_trades:
                print("\nAucun trade actif.")
                return
            
            # Préparer les données pour l'affichage
            table_data = []
            for signal_id, trade_info in active_trades.items():
                signal = trade_info["signal"]
                symbol = f"{signal.symbol}USDT"
                direction = signal.direction
                entry_min = signal.entry_min
                entry_max = signal.entry_max
                stop_loss = signal.stop_loss
                take_profits = ", ".join([f"{tp:.2f}" for tp in signal.take_profit_levels])
                status = trade_info["status"]
                timestamp = datetime.fromtimestamp(trade_info["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                
                table_data.append([
                    signal_id,
                    symbol,
                    direction,
                    f"{entry_min:.2f} - {entry_max:.2f}",
                    f"{stop_loss:.2f}",
                    take_profits,
                    status,
                    timestamp
                ])
            
            # Afficher le tableau
            headers = ["ID", "Symbole", "Direction", "Entrée", "SL", "TP", "Statut", "Timestamp"]
            print("\n=== Trades Actifs ===")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
        except Exception as e:
            print(f"Erreur lors de la récupération des trades: {str(e)}")
            logger.error(f"Erreur lors de la récupération des trades: {str(e)}")
    
    async def _cmd_performance(self, args: List[str]) -> None:
        """Affiche les statistiques de performance."""
        try:
            # Générer le rapport de performance
            performance = await self.position_monitor.generate_performance_report()
            
            # Afficher les statistiques
            print("\n=== Statistiques de Performance ===")
            print(f"Trades totaux: {performance['total_trades']}")
            print(f"Trades gagnants: {performance['winning_trades']} ({performance['win_rate']:.2f}%)")
            print(f"Trades perdants: {performance['losing_trades']}")
            print(f"Profit total: {performance['total_profit']:.2f} USDT")
            print(f"Perte totale: {performance['total_loss']:.2f} USDT")
            print(f"Profit net: {performance['net_profit']:.2f} USDT")
            print(f"Facteur de profit: {performance['profit_factor']:.2f}")
            
            # Afficher l'historique des trades
            if performance['trades']:
                print("\n=== Historique des Trades ===")
                table_data = []
                for trade in performance['trades']:
                    table_data.append([
                        trade["symbol"],
                        trade["direction"],
                        f"{trade['entry_price']:.2f}",
                        f"{trade['exit_price']:.2f}",
                        f"{trade['pnl']:.2f} USDT",
                        trade["close_reason"],
                        datetime.fromtimestamp(trade["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    ])
                
                headers = ["Symbole", "Direction", "Entrée", "Sortie", "PnL", "Raison", "Date"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
        except Exception as e:
            print(f"Erreur lors de la génération du rapport de performance: {str(e)}")
            logger.error(f"Erreur lors de la génération du rapport de performance: {str(e)}")
    
    async def _cmd_risk(self, args: List[str]) -> None:
        """Affiche ou modifie les paramètres de risque."""
        try:
            if not args:
                # Afficher les paramètres actuels
                print("\n=== Paramètres de Risque ===")
                print(f"Risque par trade: {self.trade_executor.risk_manager.risk_per_trade}%")
                print(f"Risque total maximum: {self.trade_executor.risk_manager.max_total_risk}%")
                
                # Analyser les paramètres de risque
                analysis = await self.risk_analyzer.analyze_performance()
                
                if "recommendations" in analysis:
                    print("\n=== Recommandations ===")
                    for param, recommendation in analysis["recommendations"].items():
                        print(f"{param}: {recommendation}")
                
                return
            
            # Modifier les paramètres
            if len(args) < 2:
                print("Usage: risk <param> <value>")
                print("Paramètres disponibles: risk_per_trade, max_total_risk")
                return
            
            param = args[0]
            try:
                value = float(args[1])
            except ValueError:
                print(f"Valeur invalide: {args[1]}")
                return
            
            if param == "risk_per_trade":
                if value <= 0 or value > 20:
                    print("Le risque par trade doit être entre 0 et 20%")
                    return
                
                self.trade_executor.risk_manager.risk_per_trade = value
                print(f"Risque par trade modifié à {value}%")
                
            elif param == "max_total_risk":
                if value <= 0 or value > 50:
                    print("Le risque total maximum doit être entre 0 et 50%")
                    return
                
                self.trade_executor.risk_manager.max_total_risk = value
                print(f"Risque total maximum modifié à {value}%")
                
            else:
                print(f"Paramètre inconnu: {param}")
                print("Paramètres disponibles: risk_per_trade, max_total_risk")
            
        except Exception as e:
            print(f"Erreur lors de la gestion des paramètres de risque: {str(e)}")
            logger.error(f"Erreur lors de la gestion des paramètres de risque: {str(e)}")
    
    async def _cmd_cancel(self, args: List[str]) -> None:
        """Annule un trade actif."""
        try:
            if not args:
                print("Usage: cancel <signal_id>")
                return
            
            signal_id = args[0]
            
            # Vérifier si le trade existe
            if signal_id not in self.trade_executor.active_trades:
                print(f"Trade non trouvé: {signal_id}")
                return
            
            # Annuler le trade
            result = await self.trade_executor.cancel_trade(signal_id)
            
            if result["status"] == "success":
                print(f"Trade {signal_id} annulé avec succès")
            else:
                print(f"Erreur lors de l'annulation du trade: {result['message']}")
            
        except Exception as e:
            print(f"Erreur lors de l'annulation du trade: {str(e)}")
            logger.error(f"Erreur lors de l'annulation du trade: {str(e)}")
    
    async def _cmd_signal(self, args: List[str]) -> None:
        """Traite un signal manuellement."""
        try:
            if not args:
                print("Usage: signal <message>")
                return
            
            # Reconstruire le message
            message = " ".join(args)
            
            # Créer un parser de signaux
            from src.signal_parser import SignalParser, SignalValidator
            
            signal_parser = SignalParser()
            signal_validator = SignalValidator()
            
            # Parser le signal
            signal = signal_parser.parse(message)
            
            if not signal:
                print("Impossible de parser le signal. Format invalide.")
                return
            
            # Valider le signal
            errors = signal_validator.validate(signal)
            
            if errors:
                print("Erreurs de validation:")
                for error in errors:
                    print(f"- {error}")
                return
            
            # Exécuter le signal
            print(f"Exécution du signal: {signal.symbol} {signal.direction}")
            result = await self.trade_executor.execute_signal(signal)
            
            if result["status"] == "success":
                print(f"Signal exécuté avec succès. ID: {result['signal_id']}")
            else:
                print(f"Erreur lors de l'exécution du signal: {result['message']}")
            
        except Exception as e:
            print(f"Erreur lors du traitement du signal: {str(e)}")
            logger.error(f"Erreur lors du traitement du signal: {str(e)}")
    
    async def _cmd_config(self, args: List[str]) -> None:
        """Affiche ou modifie la configuration."""
        try:
            # Pour l'instant, simplement afficher un message
            print("\nFonctionnalité de configuration non implémentée.")
            print("Utilisez la commande 'risk' pour modifier les paramètres de risque.")
            
        except Exception as e:
            print(f"Erreur lors de la gestion de la configuration: {str(e)}")
            logger.error(f"Erreur lors de la gestion de la configuration: {str(e)}")
    
    async def _cmd_exit(self, args: List[str]) -> None:
        """Quitte l'interface."""
        print("\nArrêt de l'interface...")
        self.running = False


class DashboardUI:
    """
    Interface utilisateur de tableau de bord pour le bot de trading.
    
    Cette classe fournit une interface utilisateur de tableau de bord
    en mode texte (curses) pour surveiller les positions et les trades.
    """
    
    def __init__(self, trade_executor: TradeExecutor, position_monitor: PositionMonitor):
        """
        Initialise le tableau de bord.
        
        Args:
            trade_executor: L'exécuteur de trades à utiliser.
            position_monitor: Le moniteur de positions à utiliser.
        """
        self.trade_executor = trade_executor
        self.position_monitor = position_monitor
        self.running = False
        self.update_interval = 5  # Intervalle de mise à jour en secondes
        
        logger.info("Tableau de bord initialisé")
    
    def start(self) -> None:
        """Démarre le tableau de bord."""
        self.running = True
        
        # Lancer le tableau de bord dans un thread séparé
        dashboard_thread = threading.Thread(target=self._run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        logger.info("Tableau de bord démarré")
    
    def stop(self) -> None:
        """Arrête le tableau de bord."""
        self.running = False
        logger.info("Tableau de bord arrêté")
    
    def _run_dashboard(self) -> None:
        """Exécute le tableau de bord."""
        try:
            # Initialiser curses
            stdscr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_RED, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            curses.curs_set(0)
            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)
            stdscr.timeout(100)  # Timeout pour getch() en ms
            
            # Boucle principale
            while self.running:
                try:
                    # Gérer les entrées clavier
                    key = stdscr.getch()
                    if key == ord('q'):
                        break
                    
                    # Effacer l'écran
                    stdscr.clear()
                    
                    # Obtenir les dimensions de l'écran
                    max_y, max_x = stdscr.getmaxyx()
                    
                    # Afficher le titre
                    title = "Bot de Trading Telegram-Binance Futures"
                    stdscr.addstr(0, (max_x - len(title)) // 2, title, curses.A_BOLD)
                    
                    # Afficher l'heure
                    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    stdscr.addstr(0, max_x - len(time_str) - 1, time_str)
                    
                    # Afficher les informations du compte
                    self._display_account_info(stdscr, 2, 0, max_x)
                    
                    # Afficher les positions ouvertes
                    self._display_positions(stdscr, 6, 0, max_x, max_y - 15)
                    
                    # Afficher les trades actifs
                    self._display_trades(stdscr, max_y - 14, 0, max_x, 13)
                    
                    # Afficher les commandes
                    stdscr.addstr(max_y - 1, 0, "q: Quitter | r: Rafraîchir", curses.A_BOLD)
                    
                    # Rafraîchir l'écran
                    stdscr.refresh()
                    
                    # Attendre avant la prochaine mise à jour
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    # Gérer les erreurs dans la boucle
                    stdscr.clear()
                    stdscr.addstr(0, 0, f"Erreur: {str(e)}")
                    stdscr.refresh()
                    time.sleep(2)
            
        except Exception as e:
            logger.error(f"Erreur dans le tableau de bord: {str(e)}")
        finally:
            # Restaurer les paramètres du terminal
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()
    
    def _display_account_info(self, stdscr, y, x, max_x) -> None:
        """
        Affiche les informations du compte.
        
        Args:
            stdscr: L'écran curses.
            y: La position Y.
            x: La position X.
            max_x: La largeur maximale.
        """
        # Cette méthode devrait être appelée de manière asynchrone,
        # mais curses ne fonctionne pas bien avec asyncio.
        # Pour simplifier, nous utilisons des données fictives.
        
        stdscr.addstr(y, x, "=== Informations du Compte ===", curses.A_BOLD)
        
        # Données fictives pour l'exemple
        balance = 10000.0
        risk_per_trade = 5.0
        max_risk = 10.0
        current_risk = 7.5
        
        # Afficher les informations
        stdscr.addstr(y + 1, x, f"Solde: {balance:.2f} USDT")
        stdscr.addstr(y + 1, x + 30, f"Risque par trade: {risk_per_trade:.1f}%")
        stdscr.addstr(y + 1, x + 60, f"Risque max: {max_risk:.1f}%")
        
        # Afficher le risque actuel avec couleur
        risk_color = curses.color_pair(1)  # Vert par défaut
        if current_risk > max_risk * 0.8:
            risk_color = curses.color_pair(2)  # Rouge
        elif current_risk > max_risk * 0.5:
            risk_color = curses.color_pair(3)  # Jaune
            
        stdscr.addstr(y + 2, x, f"Risque actuel: {current_risk:.1f}%", risk_color)
    
    def _display_positions(self, stdscr, y, x, max_x, max_height) -> None:
        """
        Affiche les positions ouvertes.
        
        Args:
            stdscr: L'écran curses.
            y: La position Y.
            x: La position X.
            max_x: La largeur maximale.
            max_height: La hauteur maximale.
        """
        stdscr.addstr(y, x, "=== Positions Ouvertes ===", curses.A_BOLD)
        
        # Données fictives pour l'exemple
        positions = [
            {"symbol": "BTCUSDT", "side": "LONG", "entry_price": 50000.0, "mark_price": 51000.0,
             "position_amt": 0.1, "leverage": 10, "unrealized_pnl": 100.0, "roe": 20.0},
            {"symbol": "ETHUSDT", "side": "SHORT", "entry_price": 3000.0, "mark_price": 2900.0,
             "position_amt": 1.0, "leverage": 5, "unrealized_pnl": 100.0, "roe": 6.67}
        ]
        
        if not positions:
            stdscr.addstr(y + 1, x, "Aucune position ouverte.")
            return
        
        # Afficher l'en-tête
        headers = ["Symbole", "Direction", "Prix d'entrée", "Prix actuel", "Quantité", "Levier", "PnL", "ROE"]
        header_str = " | ".join(headers)
        stdscr.addstr(y + 1, x, header_str, curses.A_BOLD)
        
        # Afficher les positions
        for i, position in enumerate(positions):
            if i >= max_height - 3:  # Limiter le nombre de positions affichées
                stdscr.addstr(y + 2 + i, x, "...")
                break
                
            # Déterminer la couleur en fonction du PnL
            color = curses.color_pair(1) if position["unrealized_pnl"] > 0 else curses.color_pair(2)
            
            # Formater la ligne
            line = f"{position['symbol']:<8} | {position['side']:<6} | {position['entry_price']:<12.2f} | "
            line += f"{position['mark_price']:<11.2f} | {position['position_amt']:<8.6f} | {position['leverage']}x"
            line += f" | {position['unrealized_pnl']:<9.2f} | {position['roe']:<6.2f}%"
            
            stdscr.addstr(y + 2 + i, x, line, color)
    
    def _display_trades(self, stdscr, y, x, max_x, max_height) -> None:
        """
        Affiche les trades actifs.
        
        Args:
            stdscr: L'écran curses.
            y: La position Y.
            x: La position X.
            max_x: La largeur maximale.
            max_height: La hauteur maximale.
        """
        stdscr.addstr(y, x, "=== Trades Actifs ===", curses.A_BOLD)
        
        # Données fictives pour l'exemple
        trades = [
            {"id": "BTC_LONG_12345678", "symbol": "BTCUSDT", "direction": "LONG",
             "entry": "50000.00 - 51000.00", "sl": "49000.00", "tp": "52000.00, 53000.00, 54000.00",
             "status": "active", "timestamp": "2023-01-01 12:00:00"},
            {"id": "ETH_SHORT_87654321", "symbol": "ETHUSDT", "direction": "SHORT",
             "entry": "3000.00 - 3100.00", "sl": "3200.00", "tp": "2900.00, 2800.00, 2700.00",
             "status": "pending", "timestamp": "2023-01-01 13:00:00"}
        ]
        
        if not trades:
            stdscr.addstr(y + 1, x, "Aucun trade actif.")
            return
        
        # Afficher l'en-tête
        headers = ["ID", "Symbole", "Direction", "Entrée", "SL", "TP", "Statut", "Timestamp"]
        header_str = " | ".join(headers)
        stdscr.addstr(y + 1, x, header_str, curses.A_BOLD)
        
        # Afficher les trades
        for i, trade in enumerate(trades):
            if i >= max_height - 3:  # Limiter le nombre de trades affichés
                stdscr.addstr(y + 2 + i, x, "...")
                break
                
            # Déterminer la couleur en fonction du statut
            color = curses.color_pair(4)  # Cyan pour actif
            if trade["status"] == "pending":
                color = curses.color_pair(3)  # Jaune pour en attente
            
            # Formater la ligne
            id_short = trade["id"][:15] + "..." if len(trade["id"]) > 18 else trade["id"]
            tp_short = trade["tp"][:20] + "..." if len(trade["tp"]) > 23 else trade["tp"]
            
            line = f"{id_short:<18} | {trade['symbol']:<8} | {trade['direction']:<6} | "
            line += f"{trade['entry']:<16} | {trade['sl']:<8} | {tp_short:<23} | "
            line += f"{trade['status']:<8} | {trade['timestamp']}"
            
            stdscr.addstr(y + 2 + i, x, line, color)


class NotificationManager:
    """
    Gestionnaire de notifications pour le bot de trading.
    
    Cette classe gère l'envoi de notifications via différents canaux
    (Telegram, console, etc.).
    """
    
    def __init__(self, telegram_client=None):
        """
        Initialise le gestionnaire de notifications.
        
        Args:
            telegram_client: Le client Telegram à utiliser pour les notifications.
        """
        self.telegram_client = telegram_client
        self.notification_history = []
        
        logger.info("Gestionnaire de notifications initialisé")
    
    async def send_notification(self, message: str, level: str = "info") -> None:
        """
        Envoie une notification.
        
        Args:
            message: Le message à envoyer.
            level: Le niveau de la notification (info, warning, error).
        """
        # Enregistrer la notification dans l'historique
        self.notification_history.append({
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limiter la taille de l'historique
        if len(self.notification_history) > 100:
            self.notification_history = self.notification_history[-100:]
        
        # Logger la notification
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)
        
        # Envoyer via Telegram si disponible
        if self.telegram_client:
            try:
                await self.telegram_client.send_message(message)
            except Exception as e:
                logger.error(f"Erreur lors de l'envoi de la notification Telegram: {str(e)}")
        
        # Afficher dans la console
        print(f"[{level.upper()}] {message}")
    
    def get_notification_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des notifications.
        
        Args:
            limit: Le nombre maximum de notifications à récupérer.
            
        Returns:
            Une liste de dictionnaires contenant les notifications.
        """
        return self.notification_history[-limit:]


def main():
    """Fonction principale pour exécuter l'interface utilisateur."""
    import os
    import asyncio
    from src.binance_client import BinanceClient
    from src.risk_manager import RiskManager
    from src.trade_executor import TradeExecutor
    from src.position_monitor import PositionMonitor, RiskAnalyzer
    
    async def run_ui():
        # Récupérer les clés API depuis les variables d'environnement
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            print("Les clés API Binance ne sont pas définies dans les variables d'environnement.")
            print("Définissez BINANCE_API_KEY et BINANCE_API_SECRET avant de lancer le bot.")
            return
        
        # Créer les composants nécessaires
        binance_client = BinanceClient(api_key, api_secret, testnet=True)
        risk_manager = RiskManager(binance_client)
        trade_executor = TradeExecutor(binance_client, risk_manager)
        
        # Créer le gestionnaire de notifications
        notification_manager = NotificationManager()
        
        # Fonction de callback pour les notifications
        async def notification_callback(message):
            await notification_manager.send_notification(message)
        
        # Créer le moniteur de positions
        position_monitor = PositionMonitor(
            binance_client=binance_client,
            risk_manager=risk_manager,
            trade_executor=trade_executor,
            notification_callback=notification_callback
        )
        
        # Créer l'analyseur de risque
        risk_analyzer = RiskAnalyzer(risk_manager, position_monitor)
        
        # Démarrer la surveillance des positions
        await position_monitor.start_monitoring()
        
        # Créer et démarrer le tableau de bord
        dashboard = DashboardUI(trade_executor, position_monitor)
        dashboard.start()
        
        # Créer et démarrer l'interface en ligne de commande
        cli = CommandLineInterface(trade_executor, position_monitor, risk_analyzer)
        await cli.start()
        
        # Arrêter la surveillance des positions
        await position_monitor.stop_monitoring()
        
        # Arrêter le tableau de bord
        dashboard.stop()
    
    # Exécuter l'interface utilisateur
    asyncio.run(run_ui())


if __name__ == "__main__":
    main()
