"""
Module principal du bot de trading Telegram-Binance Futures.

Ce module intègre tous les composants du système et fournit le point d'entrée
principal pour l'exécution du bot.
"""

import asyncio
import logging
import os
import sys
from queue import Queue
import argparse
from typing import Dict, Any, Optional

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_parser import SignalParser, SignalValidator, TradingSignal
from src.telegram_client import TelegramClient, SignalProcessor

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBinanceBot:
    """
    Classe principale qui intègre tous les composants du bot de trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le bot avec la configuration fournie.
        
        Args:
            config: Dictionnaire contenant la configuration du bot.
        """
        self.config = config
        self.signal_queue = Queue()
        
        # Initialiser les composants
        self.signal_parser = SignalParser()
        self.signal_validator = SignalValidator()
        
        # Initialiser le client Telegram
        self.telegram_client = TelegramClient(
            token=config['telegram_token'],
            signal_queue=self.signal_queue,
            allowed_chats=config.get('allowed_chats')
        )
        
        # Initialiser le processeur de signaux
        self.signal_processor = SignalProcessor(
            signal_queue=self.signal_queue,
            parser_callback=self.parse_and_validate,
            telegram_client=self.telegram_client
        )
    
    def parse_and_validate(self, message_text: str) -> Optional[TradingSignal]:
        """
        Parse et valide un message de signal.
        
        Args:
            message_text: Le texte du message à analyser.
            
        Returns:
            Un objet TradingSignal validé ou None si le parsing ou la validation échoue.
        """
        # Parser le signal
        signal = self.signal_parser.parse(message_text)
        
        if signal:
            # Valider le signal
            errors = self.signal_validator.validate(signal)
            
            if errors:
                logger.warning(f"Erreurs de validation: {errors}")
                return None
            
            return signal
        
        return None
    
    async def start(self):
        """Démarre tous les composants du bot."""
        logger.info("Démarrage du bot de trading Telegram-Binance Futures...")
        
        # Démarrer le processeur de signaux
        self.signal_processor.start()
        
        # Démarrer le client Telegram (cela bloquera jusqu'à ce que le bot soit arrêté)
        self.telegram_client.start()


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Bot de trading Telegram-Binance Futures')
    
    parser.add_argument('--telegram-token', type=str, help='Token du bot Telegram')
    parser.add_argument('--allowed-chats', type=int, nargs='+', help='IDs des chats autorisés')
    
    return parser.parse_args()


def main():
    """Fonction principale pour exécuter le bot."""
    # Parser les arguments
    args = parse_arguments()
    
    # Créer la configuration
    config = {
        'telegram_token': args.telegram_token or os.environ.get('TELEGRAM_TOKEN'),
        'allowed_chats': args.allowed_chats
    }
    
    # Vérifier que le token Telegram est fourni
    if not config['telegram_token']:
        print("Erreur: Token Telegram non fourni. Utilisez --telegram-token ou définissez la variable d'environnement TELEGRAM_TOKEN.")
        sys.exit(1)
    
    # Créer et démarrer le bot
    bot = TelegramBinanceBot(config)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
