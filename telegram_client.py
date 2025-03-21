"""
Module de connexion à Telegram pour recevoir les signaux de trading.

Ce module contient les classes et fonctions nécessaires pour se connecter
à Telegram et recevoir les messages contenant des signaux de trading.
"""

import asyncio
import logging
from typing import List, Callable, Optional, Dict, Any
from queue import Queue

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramClient:
    """
    Classe pour gérer la connexion à Telegram et recevoir les messages.
    
    Cette classe utilise python-telegram-bot pour se connecter à l'API Telegram
    et recevoir les messages contenant des signaux de trading.
    """
    
    def __init__(self, token: str, signal_queue: Queue, allowed_chats: Optional[List[int]] = None):
        """
        Initialise le client Telegram.
        
        Args:
            token: Le token d'API du bot Telegram.
            signal_queue: Une file d'attente pour stocker les signaux reçus.
            allowed_chats: Liste des IDs de chat autorisés à envoyer des signaux.
                           Si None, tous les chats sont autorisés.
        """
        self.token = token
        self.signal_queue = signal_queue
        self.allowed_chats = allowed_chats or []
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
        
    def setup_handlers(self):
        """Configure les gestionnaires de messages et de commandes."""
        # Gestionnaire pour les commandes /start et /help
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        
        # Gestionnaire pour les messages texte (potentiels signaux)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Gestionnaire d'erreurs
        self.application.add_error_handler(self.handle_error)
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère la commande /start."""
        user = update.effective_user
        await update.message.reply_text(
            f"Bonjour {user.first_name}! Je suis votre bot de trading Binance Futures. "
            f"Envoyez-moi des signaux de trading et je les exécuterai automatiquement."
        )
    
    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère la commande /help."""
        help_text = (
            "Je peux exécuter automatiquement des trades sur Binance Futures "
            "basés sur les signaux que vous m'envoyez.\n\n"
            "Format de signal attendu:\n"
            "🟢 SYMBOLE DIRECTION\n\n"
            "🎯Entry price: MIN - MAX\n\n"
            "TP:\n"
            "NIVEAU1\n"
            "NIVEAU2\n"
            "...\n\n"
            "🛑 SL NIVEAU\n\n"
            "Exemple:\n"
            "🟢 ETH LONG\n\n"
            "🎯Entry price: 2255 - 2373\n\n"
            "TP:\n"
            "2500\n"
            "2601\n"
            "2770\n\n"
            "🛑 SL 2150"
        )
        await update.message.reply_text(help_text)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Gère les messages reçus et identifie les potentiels signaux de trading.
        
        Args:
            update: L'objet Update contenant les informations du message.
            context: Le contexte de l'application.
        """
        # Vérifier si le message provient d'une source autorisée
        if self.allowed_chats and update.effective_chat.id not in self.allowed_chats:
            logger.info(f"Message ignoré de chat non autorisé: {update.effective_chat.id}")
            return
        
        message_text = update.message.text
        
        # Vérifier si c'est un signal potentiel (filtrage basique)
        if "LONG" in message_text.upper() or "SHORT" in message_text.upper():
            logger.info(f"Signal potentiel détecté de {update.effective_chat.id}")
            
            # Ajouter à la file d'attente pour traitement
            signal_data = {
                'text': message_text,
                'chat_id': update.effective_chat.id,
                'message_id': update.message.message_id,
                'timestamp': update.message.date.timestamp(),
                'update': update  # Stocker l'objet Update pour pouvoir répondre plus tard
            }
            
            self.signal_queue.put(signal_data)
            
            # Accuser réception du signal
            await update.message.reply_text(
                "Signal reçu et en cours de traitement. Je vous tiendrai informé de son exécution."
            )
        else:
            logger.debug(f"Message non reconnu comme signal: {message_text[:20]}...")
    
    async def handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gère les erreurs survenues pendant le traitement des mises à jour."""
        logger.error(f"L'erreur {context.error} est survenue", exc_info=context.error)
        
        # Envoyer un message à l'utilisateur si possible
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "Désolé, une erreur s'est produite lors du traitement de votre message."
            )
    
    async def send_notification(self, chat_id: int, message: str) -> None:
        """
        Envoie une notification à un chat spécifique.
        
        Args:
            chat_id: L'ID du chat auquel envoyer la notification.
            message: Le message à envoyer.
        """
        try:
            await self.application.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Notification envoyée à {chat_id}")
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la notification à {chat_id}: {str(e)}")
    
    def start(self) -> None:
        """Démarre le bot en mode polling."""
        logger.info("Démarrage du bot Telegram...")
        self.application.run_polling()
    
    def start_async(self) -> None:
        """Démarre le bot de manière asynchrone."""
        logger.info("Démarrage asynchrone du bot Telegram...")
        asyncio.create_task(self.application.run_polling())


class SignalProcessor:
    """
    Classe pour traiter les signaux reçus de Telegram.
    
    Cette classe prend les signaux de la file d'attente, les parse,
    et les transmet au gestionnaire de trading.
    """
    
    def __init__(self, signal_queue: Queue, parser_callback: Callable, telegram_client: Optional[TelegramClient] = None):
        """
        Initialise le processeur de signaux.
        
        Args:
            signal_queue: La file d'attente contenant les signaux à traiter.
            parser_callback: La fonction de callback pour parser les signaux.
            telegram_client: Le client Telegram pour envoyer des notifications.
        """
        self.signal_queue = signal_queue
        self.parser_callback = parser_callback
        self.telegram_client = telegram_client
        self.running = False
        self.processing_task = None
    
    async def process_signals(self) -> None:
        """Traite les signaux de la file d'attente en continu."""
        self.running = True
        
        while self.running:
            try:
                # Vérifier si la file d'attente contient des signaux
                if not self.signal_queue.empty():
                    # Récupérer le prochain signal
                    signal_data = self.signal_queue.get()
                    
                    # Extraire les informations du signal
                    message_text = signal_data['text']
                    chat_id = signal_data['chat_id']
                    
                    logger.info(f"Traitement du signal de {chat_id}: {message_text[:20]}...")
                    
                    # Parser le signal
                    parsed_signal = self.parser_callback(message_text)
                    
                    if parsed_signal:
                        logger.info(f"Signal parsé avec succès: {parsed_signal}")
                        
                        # Envoyer une notification si le client Telegram est disponible
                        if self.telegram_client:
                            await self.telegram_client.send_notification(
                                chat_id,
                                f"Signal parsé avec succès:\n{parsed_signal}"
                            )
                    else:
                        logger.warning(f"Échec du parsing du signal: {message_text[:50]}...")
                        
                        # Envoyer une notification d'échec
                        if self.telegram_client:
                            await self.telegram_client.send_notification(
                                chat_id,
                                "Échec du parsing du signal. Veuillez vérifier le format."
                            )
                    
                    # Marquer le signal comme traité
                    self.signal_queue.task_done()
                
                # Attendre un court instant avant de vérifier à nouveau
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement du signal: {str(e)}")
                await asyncio.sleep(1)  # Attendre un peu plus longtemps en cas d'erreur
    
    def start(self) -> None:
        """Démarre le traitement des signaux dans une tâche asynchrone."""
        if not self.processing_task or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.process_signals())
            logger.info("Processeur de signaux démarré")
    
    def stop(self) -> None:
        """Arrête le traitement des signaux."""
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            logger.info("Processeur de signaux arrêté")


def main():
    """Fonction principale pour tester le module."""
    # Créer une file d'attente pour les signaux
    signal_queue = Queue()
    
    # Créer un client Telegram (remplacer 'YOUR_TOKEN' par un vrai token)
    # Pour les tests, vous pouvez utiliser un token de test obtenu via @BotFather
    telegram_client = TelegramClient("YOUR_TOKEN", signal_queue)
    
    # Démarrer le bot
    telegram_client.start()


if __name__ == "__main__":
    main()
