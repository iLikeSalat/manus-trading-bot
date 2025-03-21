import os
import logging
import asyncio
import re
import math
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("telegram_binance_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
USE_TESTNET = os.getenv('USE_TESTNET', 'true').lower() == 'true'
TRADE_AMOUNT = float(os.getenv('TRADE_AMOUNT', '100.0'))  # Amount in USDT
FIXED_LEVERAGE = 20  # Fixed leverage of 20x as requested
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '0.04'))  # 4% risk per trade
MIN_NOTIONAL_VALUE = 100.0  # Minimum notional value required by Binance

# Signal pattern for advanced extraction
SIGNAL_PATTERN = r'(🟢|🔴)\s+([A-Z]+)\s+(LONG|SHORT).*?(?:Entry price|Prix d\'entrée):\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?).*?TP:(?:\s*\n*)((?:\d+(?:\.\d+)?(?:\s*\n*))+).*?(?:SL|🛑 SL)\s*(\d+(?:\.\d+)?)'

# Progressive TP distribution
TP_DISTRIBUTION = [0.3, 0.4, 0.3]  # 30%, 40%, 30% for first 3 TPs

class TelegramBinanceBot:
    def __init__(self):
        """Initialize the bot with Telegram and Binance clients."""
        # Initialize Binance client
        self.binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=USE_TESTNET)
        
        # Initialize Telegram application
        self.telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        self.telegram_app.add_handler(CommandHandler("start", self.start_command))
        self.telegram_app.add_handler(CommandHandler("help", self.help_command))
        self.telegram_app.add_handler(CommandHandler("status", self.status_command))
        self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Store active monitoring tasks
        self.monitoring_tasks = {}
        
        logger.info("Bot initialized successfully")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        await update.message.reply_text(
            "👋 Bienvenue sur le Bot de Trading Telegram-Binance Futures!\n\n"
            "Je suis conçu pour exécuter automatiquement des trades sur Binance Futures "
            "lorsque je reçois des signaux de trading.\n\n"
            "Utilisez /help pour voir les commandes disponibles."
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        await update.message.reply_text(
            "📚 Commandes disponibles:\n\n"
            "/start - Démarrer le bot\n"
            "/help - Afficher ce message d'aide\n"
            "/status - Vérifier le statut du bot et du compte Binance\n\n"
            "Pour exécuter un trade, envoyez un message au format suivant:\n"
            "🟢 BTC LONG\n"
            "🎯Entry price: 87000 - 88000\n"
            "TP:\n"
            "90000\n"
            "92000\n"
            "95000\n"
            "🛑 SL 85000"
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send status information when the command /status is issued."""
        try:
            # Get account information
            account_info = self.binance_client.futures_account_balance()
            usdt_balance = next((item['balance'] for item in account_info if item['asset'] == 'USDT'), '0')
            
            # Get open positions
            positions = self.binance_client.futures_position_information()
            active_positions = [p for p in positions if float(p["positionAmt"]) != 0]
            
            # Format positions info
            positions_info = ""
            if active_positions:
                for pos in active_positions:
                    symbol = pos['symbol']
                    side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
                    amount = abs(float(pos['positionAmt']))
                    entry_price = float(pos['entryPrice'])
                    pnl = float(pos['unRealizedProfit'])
                    leverage = int(pos['leverage'])
                    positions_info += f"\n- {symbol} {side}: {amount} @ {entry_price} (PnL: {pnl:.2f} USDT, Levier: {leverage}x)"
            else:
                positions_info = "\nAucune position ouverte."
            
            # Get open orders
            orders = self.binance_client.futures_get_open_orders()
            orders_info = ""
            if orders:
                for order in orders:
                    symbol = order['symbol']
                    side = order['side']
                    type = order['type']
                    price = order.get('price', 'Market')
                    stop_price = order.get('stopPrice', 'N/A')
                    quantity = order['origQty']
                    orders_info += f"\n- {symbol} {side} {type}: {quantity} @ {price} (Stop: {stop_price})"
            else:
                orders_info = "\nAucun ordre en attente."
            
            # Send status message
            status_message = (
                f"📊 Statut du Bot:\n\n"
                f"Mode: {'Testnet' if USE_TESTNET else 'Production'}\n"
                f"Solde USDT: {usdt_balance}\n"
                f"Montant par trade: {TRADE_AMOUNT} USDT\n"
                f"Marge utilisée par trade: {RISK_PERCENTAGE*100}% du capital\n"
                f"Levier fixe: {FIXED_LEVERAGE}x\n\n"
                f"Positions actives:{positions_info}\n\n"
                f"Ordres en attente:{orders_info}"
            )
            
            await update.message.reply_text(status_message)
            
        except Exception as e:
            logger.error(f"Error in status command: {str(e)}")
            await update.message.reply_text(f"❌ Erreur lors de la récupération du statut: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages and detect trading signals."""
        message_text = update.message.text
        
        # Try to match the advanced signal pattern
        match = re.search(SIGNAL_PATTERN, message_text, re.DOTALL)
        if match:
            emoji, symbol, direction, entry_min, entry_max, tp_text, sl = match.groups()
            
            # Convert to proper types
            entry_min = float(entry_min)
            entry_max = float(entry_max)
            sl = float(sl)
            
            # Extract TP levels
            tp_levels = []
            for line in tp_text.strip().split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    tp_levels.append(float(line))
            
            # Log the detected signal
            logger.info(f"Signal detected: {emoji} {symbol} {direction}")
            logger.info(f"Entry: {entry_min}-{entry_max}, SL: {sl}, TP: {tp_levels}")
            
            # Acknowledge receipt of the signal
            await update.message.reply_text(
                f"Signal reçu: {emoji} {symbol} {direction}\n"
                f"Entrée: {entry_min}-{entry_max}\n"
                f"SL: {sl}\n"
                f"TP: {', '.join(str(tp) for tp in tp_levels)}\n\n"
                f"Traitement en cours..."
            )
            
            # Execute the trade with new progressive strategy
            try:
                result = await self.execute_progressive_trade(symbol, direction, entry_min, entry_max, sl, tp_levels, update)
                await update.message.reply_text(result)
            except Exception as e:
                error_msg = f"❌ Erreur lors de l'exécution du trade: {str(e)}"
                logger.error(f"Error executing trade: {str(e)}")
                await update.message.reply_text(error_msg)
        else:
            # Try the simple pattern as fallback
            simple_match = re.search(r'(🟢|🔴)\s+([A-Z]+)\s+(LONG|SHORT)', message_text)
            if simple_match:
                emoji, symbol, direction = simple_match.groups()
                
                # Log the detected signal
                logger.info(f"Simple signal detected: {emoji} {symbol} {direction}")
                
                # Acknowledge receipt of the signal
                await update.message.reply_text(f"Signal simple reçu: {emoji} {symbol} {direction}. Traitement en cours...")
                
                # Execute the trade
                try:
                    result = await self.execute_simple_trade(symbol, direction, update)
                    await update.message.reply_text(result)
                except Exception as e:
                    error_msg = f"❌ Erreur lors de l'exécution du trade: {str(e)}"
                    logger.error(f"Error executing simple trade: {str(e)}")
                    await update.message.reply_text(error_msg)
            else:
                # Not a trading signal
                logger.debug(f"Received message: {message_text}")
    
    async def execute_simple_trade(self, symbol: str, direction: str, update: Update) -> str:
        """Execute a simple trade on Binance Futures based on the signal."""
        try:
            # Format the symbol
            futures_symbol = f"{symbol}USDT"
            
            # Set the side
            side = "BUY" if direction == "LONG" else "SELL"
            
            # Get current price
            ticker = self.binance_client.futures_symbol_ticker(symbol=futures_symbol)
            current_price = float(ticker['price'])
            
            # Get exchange info for precision
            exchange_info = self.binance_client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == futures_symbol), None)
            
            if not symbol_info:
                raise ValueError(f"Symbol {futures_symbol} not found in exchange info")
            
            # Find the quantity filter
            quantity_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if not quantity_filter:
                raise ValueError(f"LOT_SIZE filter not found for {futures_symbol}")
            
            # Get minimum quantity and step size
            min_qty = float(quantity_filter['minQty'])
            step_size = float(quantity_filter['stepSize'])
            
            # Set leverage to fixed 20x
            try:
                leverage_result = self.binance_client.futures_change_leverage(
                    symbol=futures_symbol, 
                    leverage=FIXED_LEVERAGE
                )
                actual_leverage = int(leverage_result['leverage'])
                logger.info(f"Leverage set to {actual_leverage}x")
                
                if actual_leverage != FIXED_LEVERAGE:
                    logger.warning(f"Requested leverage {FIXED_LEVERAGE}x but Binance set {actual_leverage}x")
                    await update.message.reply_text(f"⚠️ Avertissement: Levier demandé {FIXED_LEVERAGE}x mais Binance a défini {actual_leverage}x")
            except Exception as e:
                logger.error(f"Error setting leverage: {str(e)}")
                await update.message.reply_text(f"⚠️ Erreur lors de la définition du levier: {str(e)}")
                actual_leverage = FIXED_LEVERAGE  # Use fixed leverage if setting fails
            
            # Get account balance
            account_info = self.binance_client.futures_account_balance()
            usdt_balance = float(next((item['balance'] for item in account_info if item['asset'] == 'USDT'), 0))
            
            # Calculate margin to use (4% of balance)
            margin_amount = usdt_balance * RISK_PERCENTAGE
            
            # Calculate position size
            position_value = margin_amount * actual_leverage
            quantity = position_value / current_price
            
            # Round to step size
            quantity = self.round_step_size(quantity, min_qty, step_size)
            
            # Check minimum notional value (100 USDT)
            notional_value = quantity * current_price
            if notional_value < MIN_NOTIONAL_VALUE:
                # Calculate minimum quantity needed
                min_notional_quantity = MIN_NOTIONAL_VALUE / current_price
                quantity = max(quantity, self.round_step_size(min_notional_quantity, min_qty, step_size))
                logger.info(f"Increased quantity to {quantity} to meet minimum notional value of {MIN_NOTIONAL_VALUE} USDT")
                await update.message.reply_text(f"ℹ️ Quantité augmentée à {quantity} pour atteindre la valeur notionnelle minimale de {MIN_NOTIONAL_VALUE} USDT")
                
                # Recalculate notional value
                notional_value = quantity * current_price
                
                # Double-check minimum notional value
                if notional_value < MIN_NOTIONAL_VALUE:
                    error_msg = f"Cannot meet minimum notional value of {MIN_NOTIONAL_VALUE} USDT. Calculated value: {notional_value} USDT"
                    logger.error(error_msg)
                    await update.message.reply_text(f"❌ Erreur: {error_msg}")
                    raise ValueError(error_msg)
            
            # Place market order
            order = self.binance_client.futures_create_order(
                symbol=futures_symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"Market order placed: {json.dumps(order)}")
            
            # Return success message
            return (
                f"✅ Trade simple exécuté avec succès!\n\n"
                f"Symbol: {futures_symbol}\n"
                f"Direction: {direction}\n"
                f"Prix: {current_price}\n"
                f"Quantité: {quantity}\n"
                f"Valeur: {current_price * quantity:.2f} USDT\n"
                f"Marge utilisée: {margin_amount:.2f} USDT ({RISK_PERCENTAGE*100}% du capital)\n"
                f"Levier: {actual_leverage}x"
            )
            
        except Exception as e:
            logger.error(f"Error executing simple trade: {str(e)}")
            raise
    
    async def execute_progressive_trade(self, symbol: str, direction: str, entry_min: float, entry_max: float, stop_loss: float, take_profit_levels: list, update: Update) -> str:
        """Execute a trade with progressive take profits and position reentry."""
        try:
            # Format the symbol
            futures_symbol = f"{symbol}USDT"
            
            # Set the side
            side = "BUY" if direction == "LONG" else "SELL"
            
            # Set the TP side (opposite of entry side)
            tp_side = "SELL" if side == "BUY" else "BUY"
            
            # Get current price
            ticker = self.binance_client.futures_symbol_ticker(symbol=futures_symbol)
            current_price = float(ticker['price'])
            
            # Get exchange info for precision
            exchange_info = self.binance_client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == futures_symbol), None)
            
            if not symbol_info:
                raise ValueError(f"Symbol {futures_symbol} not found in exchange info")
            
            # Find the quantity filter
            quantity_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if not quantity_filter:
                raise ValueError(f"LOT_SIZE filter not found for {futures_symbol}")
            
            # Get minimum quantity and step size
            min_qty = float(quantity_filter['minQty'])
            step_size = float(quantity_filter['stepSize'])
            
            # Set leverage to fixed 20x
            try:
                leverage_result = self.binance_client.futures_change_leverage(
                    symbol=futures_symbol, 
                    leverage=FIXED_LEVERAGE
                )
                actual_leverage = int(leverage_result['leverage'])
                logger.info(f"Leverage set to {actual_leverage}x")
                
                if actual_leverage != FIXED_LEVERAGE:
                    logger.warning(f"Requested leverage {FIXED_LEVERAGE}x but Binance set {actual_leverage}x")
                    await update.message.reply_text(f"⚠️ Avertissement: Levier demandé {FIXED_LEVERAGE}x mais Binance a défini {actual_leverage}x")
            except Exception as e:
                logger.error(f"Error setting leverage: {str(e)}")
                await update.message.reply_text(f"⚠️ Erreur lors de la définition du levier: {str(e)}")
                actual_leverage = FIXED_LEVERAGE  # Use fixed leverage if setting fails
            
            # Get account balance
            account_info = self.binance_client.futures_account_balance()
            usdt_balance = float(next((item['balance'] for item in account_info if item['asset'] == 'USDT'), 0))
            
            # Calculate margin to use (4% of balance)
            margin_amount = usdt_balance * RISK_PERCENTAGE
            
            # Calculate position size
            position_value = margin_amount * actual_leverage
            quantity = position_value / current_price
            
            # Round to step size
            quantity = self.round_step_size(quantity, min_qty, step_size)
            
            # Check minimum notional value (100 USDT)
            notional_value = quantity * current_price
            if notional_value < MIN_NOTIONAL_VALUE:
                # Calculate minimum quantity needed
                min_notional_quantity = MIN_NOTIONAL_VALUE / current_price
                quantity = max(quantity, self.round_step_size(min_notional_quantity, min_qty, step_size))
                logger.info(f"Increased quantity to {quantity} to meet minimum notional value of {MIN_NOTIONAL_VALUE} USDT")
                await update.message.reply_text(f"ℹ️ Quantité augmentée à {quantity} pour atteindre la valeur notionnelle minimale de {MIN_NOTIONAL_VALUE} USDT")
                
                # Recalculate notional value
                notional_value = quantity * current_price
                
                # Double-check minimum notional value
                if notional_value < MIN_NOTIONAL_VALUE:
                    error_msg = f"Cannot meet minimum notional value of {MIN_NOTIONAL_VALUE} USDT. Calculated value: {notional_value} USDT"
                    logger.error(error_msg)
                    await update.message.reply_text(f"❌ Erreur: {error_msg}")
                    raise ValueError(error_msg)
            
            # Check if current price is within entry range
            in_entry_range = entry_min <= current_price <= entry_max
            
            # Place entry order
            if in_entry_range:
                # Price is in range, use market order
                entry_order = self.binance_client.futures_create_order(
                    symbol=futures_symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                logger.info(f"Market entry order placed: {json.dumps(entry_order)}")
                
                # Wait for order to be processed
                await asyncio.sleep(1)
                
                # Check if position is open
                positions = self.binance_client.futures_position_information(symbol=futures_symbol)
                position = next((p for p in positions if p['symbol'] == futures_symbol and float(p['positionAmt']) != 0), None)
                
                if not position:
                    error_msg = "Position non ouverte après ordre d'entrée"
                    logger.error(error_msg)
                    await update.message.reply_text(f"❌ Erreur: {error_msg}")
                    return f"❌ Erreur: {error_msg}"
                
                # Get position details
                position_quantity = abs(float(position['positionAmt']))
                entry_price = float(position['entryPrice'])
                
                # Place stop loss order
                sl_order = await self.place_stop_loss(futures_symbol, side, position_quantity, stop_loss, update)
                
                # Ensure we have at least 3 TP levels
                if len(take_profit_levels) < 3:
                    error_msg = "Au moins 3 niveaux de TP sont nécessaires pour la stratégie progressive"
                    logger.error(error_msg)
                    await update.message.reply_text(f"❌ Erreur: {error_msg}")
                    return f"❌ Erreur: {error_msg}"
                
                # Try to place individual TP orders first
                individual_tp_success = False
                tp_orders = []
                tp_errors = []
                
                # Calculate TP quantities
                tp_quantities = []
                for dist_pct in TP_DISTRIBUTION:
                    tp_qty = position_quantity * dist_pct
                    tp_qty = self.round_step_size(tp_qty, min_qty, step_size)
                    tp_quantities.append(tp_qty)
                
                # Check if individual TPs would meet minimum notional
                individual_tp_viable = True
                for i, (tp_level, tp_qty) in enumerate(zip(take_profit_levels[:3], tp_quantities)):
                    tp_notional = tp_qty * tp_level
                    if tp_notional < MIN_NOTIONAL_VALUE:
                        individual_tp_viable = False
                        logger.warning(f"Individual TP{i+1} would have notional value {tp_notional:.2f} USDT, below minimum {MIN_NOTIONAL_VALUE} USDT")
                
                # If individual TPs look viable, try to place them
                if individual_tp_viable:
                    try:
                        # Place individual TP orders
                        for i, (tp_level, tp_qty) in enumerate(zip(take_profit_levels[:3], tp_quantities)):
                            try:
                                tp_order = self.binance_client.futures_create_order(
                                    symbol=futures_symbol,
                                    side=tp_side,
                                    type='TAKE_PROFIT_MARKET',
                                    quantity=tp_qty,
                                    stopPrice=tp_level,
                                    timeInForce='GTC'
                                )
                                tp_orders.append(tp_order)
                                logger.info(f"TP{i+1} order placed: {json.dumps(tp_order)}")
                                await update.message.reply_text(f"✅ TP{i+1} placé: {tp_qty} @ {tp_level}")
                                
                                # Add delay between orders
                                await asyncio.sleep(0.5)
                                
                            except Exception as e:
                                error_msg = f"Erreur lors du placement de TP{i+1}: {str(e)}"
                                logger.error(error_msg)
                                tp_errors.append(error_msg)
                                await update.message.reply_text(f"⚠️ {error_msg}")
                        
                        if len(tp_orders) == 3:
                            individual_tp_success = True
                            logger.info("All individual TP orders placed successfully")
                            await update.message.reply_text("✅ Tous les ordres TP individuels placés avec succès")
                        
                    except Exception as e:
                        logger.error(f"Error placing individual TP orders: {str(e)}")
                        await update.message.reply_text(f"⚠️ Erreur lors du placement des TP individuels: {str(e)}")
                
                # If individual TPs failed or weren't viable, try combined approach
                if not individual_tp_success:
                    logger.info("Trying combined TP approach")
                    await update.message.reply_text("ℹ️ Tentative d'approche TP combinée...")
                    
                    # Group TPs into 2 groups
                    tp_groups = [
                        {
                            "name": "TP1-3",
                            "price": sum(take_profit_levels[:3]) / 3,  # Average of first 3 TPs
                            "quantity": sum(tp_quantities)  # Sum of quantities for first 3 TPs
                        }
                    ]
                    
                    # Try to place combined TP orders
                    combined_tp_orders = []
                    combined_tp_errors = []
                    
                    for group in tp_groups:
                        try:
                            # Ensure quantity meets minimum notional
                            group_notional = group["price"] * group["quantity"]
                            if group_notional < MIN_NOTIONAL_VALUE:
                                min_group_qty = MIN_NOTIONAL_VALUE / group["price"]
                                group["quantity"] = max(group["quantity"], self.round_step_size(min_group_qty, min_qty, step_size))
                                logger.info(f"Increased {group['name']} quantity to {group['quantity']} to meet minimum notional")
                                await update.message.reply_text(f"ℹ️ Quantité de {group['name']} augmentée à {group['quantity']} pour atteindre la valeur notionnelle minimale")
                            
                            # Place combined TP order
                            tp_order = self.binance_client.futures_create_order(
                                symbol=futures_symbol,
                                side=tp_side,
                                type='TAKE_PROFIT_MARKET',
                                quantity=group["quantity"],
                                stopPrice=group["price"],
                                timeInForce='GTC'
                            )
                            combined_tp_orders.append(tp_order)
                            logger.info(f"{group['name']} order placed: {json.dumps(tp_order)}")
                            await update.message.reply_text(f"✅ {group['name']} placé: {group['quantity']} @ {group['price']}")
                            
                        except Exception as e:
                            error_msg = f"Erreur lors du placement de {group['name']}: {str(e)}"
                            logger.error(error_msg)
                            combined_tp_errors.append(error_msg)
                            await update.message.reply_text(f"⚠️ {error_msg}")
                    
                    if combined_tp_orders:
                        logger.info(f"Combined TP approach: {len(combined_tp_orders)} orders placed successfully")
                        await update.message.reply_text(f"✅ Approche TP combinée: {len(combined_tp_orders)} ordres placés avec succès")
                        tp_orders = combined_tp_orders
                    else:
                        logger.error("Failed to place any TP orders")
                        await update.message.reply_text("❌ Échec du placement des ordres TP")
                
                # Start monitoring task
                self.monitoring_tasks[futures_symbol] = asyncio.create_task(
                    self.monitor_progressive_trade(
                        symbol, 
                        direction, 
                        entry_price, 
                        stop_loss, 
                        take_profit_levels, 
                        position_quantity,
                        update
                    )
                )
                
                # Prepare TP status message
                tp_status = ""
                if individual_tp_success:
                    tp_status = f"TP individuels placés: 3/3"
                elif tp_orders:
                    tp_status = f"TP combinés placés: {len(tp_orders)}/1"
                else:
                    tp_status = "Aucun TP placé ❌"
                
                # Add TP errors if any
                if tp_errors:
                    tp_status += f"\nErreurs TP: {len(tp_errors)}"
                    for i, err in enumerate(tp_errors[:3]):  # Show first 3 errors
                        tp_status += f"\n- {err}"
                    if len(tp_errors) > 3:
                        tp_status += f"\n- ... et {len(tp_errors) - 3} autres erreurs"
                
                return (
                    f"✅ Trade progressif exécuté avec succès!\n\n"
                    f"Symbol: {futures_symbol}\n"
                    f"Direction: {direction}\n"
                    f"Prix d'entrée: {entry_price}\n"
                    f"Quantité: {position_quantity}\n"
                    f"Valeur: {entry_price * position_quantity:.2f} USDT\n"
                    f"Marge utilisée: {margin_amount:.2f} USDT ({RISK_PERCENTAGE*100}% du capital)\n"
                    f"Levier: {actual_leverage}x\n"
                    f"Stop Loss: {stop_loss}\n\n"
                    f"Statut des TP:\n{tp_status}\n\n"
                    f"Stratégie: Progression par étapes avec réentrée à TP3\n"
                    f"- TP1 (30%): {take_profit_levels[0]}\n"
                    f"- TP2 (40%): {take_profit_levels[1]} → SL déplacé à l'entrée\n"
                    f"- TP3 (30%): {take_profit_levels[2]} → Réentrée pour TP4-TP6"
                )
            else:
                # Price is outside range, use limit order at nearest boundary
                limit_price = entry_min if current_price < entry_min else entry_max
                
                limit_order = self.binance_client.futures_create_order(
                    symbol=futures_symbol,
                    side=side,
                    type='LIMIT',
                    quantity=quantity,
                    price=limit_price,
                    timeInForce='GTC'
                )
                logger.info(f"Limit entry order placed: {json.dumps(limit_order)}")
                
                # Start monitoring task for limit order
                self.monitoring_tasks[futures_symbol] = asyncio.create_task(
                    self.monitor_limit_order(
                        symbol, 
                        direction, 
                        limit_price, 
                        stop_loss, 
                        take_profit_levels, 
                        quantity,
                        update
                    )
                )
                
                return (
                    f"✅ Ordre d'entrée limite placé avec succès!\n\n"
                    f"Symbol: {futures_symbol}\n"
                    f"Direction: {direction}\n"
                    f"Prix d'entrée limite: {limit_price}\n"
                    f"Quantité: {quantity}\n"
                    f"Valeur estimée: {limit_price * quantity:.2f} USDT\n"
                    f"Marge utilisée: {margin_amount:.2f} USDT ({RISK_PERCENTAGE*100}% du capital)\n"
                    f"Levier: {actual_leverage}x\n"
                    f"Stop Loss: {stop_loss}\n\n"
                    f"Stratégie: Progression par étapes avec réentrée à TP3\n"
                    f"- TP1 (30%): {take_profit_levels[0]}\n"
                    f"- TP2 (40%): {take_profit_levels[1]} → SL déplacé à l'entrée\n"
                    f"- TP3 (30%): {take_profit_levels[2]} → Réentrée pour TP4-TP6\n\n"
                    f"Note: Les ordres TP et SL seront placés une fois la position ouverte"
                )
            
        except Exception as e:
            logger.error(f"Error executing progressive trade: {str(e)}")
            raise
    
    async def place_stop_loss(self, futures_symbol, side, quantity, stop_price, update=None):
        """Place a stop loss order."""
        try:
            # Cancel any existing SL orders
            open_orders = self.binance_client.futures_get_open_orders(symbol=futures_symbol)
            for order in open_orders:
                if order['type'] == 'STOP_MARKET':
                    self.binance_client.futures_cancel_order(
                        symbol=futures_symbol,
                        orderId=order['orderId']
                    )
                    logger.info(f"Cancelled existing SL order: {order['orderId']}")
                    if update:
                        await update.message.reply_text(f"ℹ️ Ordre SL existant annulé: {order['orderId']}")
            
            # Place new SL order
            sl_side = "SELL" if side == "BUY" else "BUY"
            sl_order = self.binance_client.futures_create_order(
                symbol=futures_symbol,
                side=sl_side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                closePosition=True,
                timeInForce='GTC'
            )
            logger.info(f"Stop loss order placed: {json.dumps(sl_order)}")
            if update:
                await update.message.reply_text(f"✅ Ordre SL placé à {stop_price}")
            return sl_order
        except Exception as e:
            logger.error(f"Error placing stop loss: {str(e)}")
            if update:
                await update.message.reply_text(f"❌ Erreur lors du placement du SL: {str(e)}")
            return None
    
    async def monitor_progressive_trade(self, symbol, direction, entry_price, stop_loss, take_profit_levels, initial_quantity, update=None):
        """
        Monitor the trade with progressive strategy:
        1. Move SL to entry price when TP2 is hit
        2. Open new position at TP3 when it's hit
        3. Continue with remaining TPs
        """
        futures_symbol = f"{symbol}USDT"
        is_long = direction == "LONG"
        side = "BUY" if is_long else "SELL"
        tp_side = "SELL" if side == "BUY" else "BUY"  # Define tp_side here for the entire function
        
        # Track state
        moved_sl_to_entry = False
        opened_second_position = False
        
        logger.info(f"Starting progressive trade monitoring for {futures_symbol}")
        if update:
            await update.message.reply_text(f"🔍 Démarrage du monitoring pour {futures_symbol}")
        
        try:
            while True:
                try:
                    # Check if position exists
                    positions = self.binance_client.futures_position_information(symbol=futures_symbol)
                    position = next((p for p in positions if p['symbol'] == futures_symbol and float(p['positionAmt']) != 0), None)
                    
                    if not position:
                        # No position, check if we need to open second position at TP3
                        if not opened_second_position and moved_sl_to_entry:
                            # TP2 was hit (SL moved to entry) but position is now closed
                            # This means either TP3 was hit or SL was hit
                            
                            # Get current price
                            ticker = self.binance_client.futures_symbol_ticker(symbol=futures_symbol)
                            current_price = float(ticker['price'])
                            
                            # Check if price is near TP3 (within 0.5%)
                            tp3_price = take_profit_levels[2]
                            price_diff_pct = abs(current_price - tp3_price) / tp3_price
                            
                            if price_diff_pct < 0.005:  # Within 0.5% of TP3
                                logger.info(f"TP3 likely hit, opening second position at {current_price}")
                                if update:
                                    await update.message.reply_text(f"🔄 TP3 probablement atteint, ouverture de la seconde position à {current_price}")
                                
                                # Open new position at current price
                                try:
                                    # Calculate new quantity (same as initial)
                                    new_quantity = initial_quantity
                                    
                                    # Get exchange info for precision
                                    exchange_info = self.binance_client.futures_exchange_info()
                                    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == futures_symbol), None)
                                    quantity_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                                    min_qty = float(quantity_filter['minQty'])
                                    step_size = float(quantity_filter['stepSize'])
                                    
                                    # Round to step size
                                    new_quantity = self.round_step_size(new_quantity, min_qty, step_size)
                                    
                                    # Check minimum notional value
                                    new_notional = new_quantity * current_price
                                    if new_notional < MIN_NOTIONAL_VALUE:
                                        min_notional_quantity = MIN_NOTIONAL_VALUE / current_price
                                        new_quantity = max(new_quantity, self.round_step_size(min_notional_quantity, min_qty, step_size))
                                        logger.info(f"Increased second position quantity to {new_quantity} to meet minimum notional")
                                        if update:
                                            await update.message.reply_text(f"ℹ️ Quantité de la seconde position augmentée à {new_quantity}")
                                    
                                    # Place market order for second position
                                    second_entry = self.binance_client.futures_create_order(
                                        symbol=futures_symbol,
                                        side=side,
                                        type='MARKET',
                                        quantity=new_quantity
                                    )
                                    logger.info(f"Second position opened: {json.dumps(second_entry)}")
                                    if update:
                                        await update.message.reply_text(f"✅ Seconde position ouverte: {new_quantity} @ {current_price}")
                                    
                                    # Wait for order to be processed
                                    await asyncio.sleep(1)
                                    
                                    # Place SL at TP1
                                    await self.place_stop_loss(futures_symbol, side, new_quantity, take_profit_levels[0], update)
                                    
                                    # Place TP orders for remaining levels
                                    if len(take_profit_levels) > 3:
                                        remaining_tps = take_profit_levels[3:]
                                        
                                        # Equal distribution for remaining TPs
                                        remaining_distribution = [1/len(remaining_tps)] * len(remaining_tps)
                                        
                                        # Try to place combined TP for remaining levels
                                        try:
                                            avg_price = sum(remaining_tps) / len(remaining_tps)
                                            
                                            tp_order = self.binance_client.futures_create_order(
                                                symbol=futures_symbol,
                                                side=tp_side,
                                                type='TAKE_PROFIT_MARKET',
                                                quantity=new_quantity,
                                                stopPrice=avg_price,
                                                timeInForce='GTC'
                                            )
                                            logger.info(f"Combined TP order for remaining levels placed: {json.dumps(tp_order)}")
                                            if update:
                                                await update.message.reply_text(f"✅ TP combiné pour les niveaux restants placé: {new_quantity} @ {avg_price}")
                                        except Exception as e:
                                            logger.error(f"Error placing combined TP for remaining levels: {str(e)}")
                                            if update:
                                                await update.message.reply_text(f"❌ Erreur lors du placement du TP combiné: {str(e)}")
                                    
                                    opened_second_position = True
                                    logger.info(f"Second position setup complete with SL at TP1 and remaining TPs")
                                    if update:
                                        await update.message.reply_text(f"✅ Configuration de la seconde position terminée avec SL à TP1")
                                    
                                except Exception as e:
                                    logger.error(f"Error opening second position: {str(e)}")
                                    if update:
                                        await update.message.reply_text(f"❌ Erreur lors de l'ouverture de la seconde position: {str(e)}")
                            else:
                                # Position closed but not at TP3, likely SL hit
                                logger.info(f"Position closed, stopping monitoring")
                                if update:
                                    await update.message.reply_text(f"ℹ️ Position fermée, arrêt du monitoring")
                                break
                        else:
                            # Position closed and no need for second position
                            logger.info(f"Position closed, stopping monitoring")
                            if update:
                                await update.message.reply_text(f"ℹ️ Position fermée, arrêt du monitoring")
                            break
                    
                    # Position exists, check if we need to move SL to entry when TP2 is hit
                    if not moved_sl_to_entry and not opened_second_position:
                        # Get current price
                        ticker = self.binance_client.futures_symbol_ticker(symbol=futures_symbol)
                        current_price = float(ticker['price'])
                        
                        # Check if price has reached TP2
                        tp2_reached = (is_long and current_price >= take_profit_levels[1]) or \
                                     (not is_long and current_price <= take_profit_levels[1])
                        
                        if tp2_reached:
                            logger.info(f"TP2 reached, moving SL to entry price")
                            if update:
                                await update.message.reply_text(f"🔄 TP2 atteint, déplacement du SL au prix d'entrée")
                            
                            # Move SL to entry price
                            await self.place_stop_loss(futures_symbol, side, float(position['positionAmt']), entry_price, update)
                            moved_sl_to_entry = True
                            logger.info(f"SL moved to entry price at {entry_price}")
                            if update:
                                await update.message.reply_text(f"✅ SL déplacé au prix d'entrée: {entry_price}")
                    
                    # Wait before checking again
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in progressive trade monitoring: {str(e)}")
                    if update:
                        await update.message.reply_text(f"⚠️ Erreur dans le monitoring: {str(e)}")
                    await asyncio.sleep(60)  # Wait before retrying
        except asyncio.CancelledError:
            logger.info(f"Monitoring task for {futures_symbol} was cancelled")
            if update:
                await update.message.reply_text(f"ℹ️ Tâche de monitoring pour {futures_symbol} annulée")
        finally:
            # Remove task from monitoring tasks
            if futures_symbol in self.monitoring_tasks:
                del self.monitoring_tasks[futures_symbol]
    
    async def monitor_limit_order(self, symbol, direction, entry_price, stop_loss, take_profit_levels, quantity, update=None):
        """Monitor a limit entry order until filled, then setup the progressive strategy."""
        futures_symbol = f"{symbol}USDT"
        
        logger.info(f"Starting limit order monitoring for {futures_symbol}")
        if update:
            await update.message.reply_text(f"🔍 Démarrage du monitoring de l'ordre limite pour {futures_symbol}")
        
        try:
            while True:
                try:
                    # Check if position exists
                    positions = self.binance_client.futures_position_information(symbol=futures_symbol)
                    position = next((p for p in positions if p['symbol'] == futures_symbol and float(p['positionAmt']) != 0), None)
                    
                    if position:
                        # Position opened, setup progressive strategy
                        logger.info(f"Limit order filled, position opened at {position['entryPrice']}")
                        if update:
                            await update.message.reply_text(f"✅ Ordre limite exécuté, position ouverte à {position['entryPrice']}")
                        
                        # Place stop loss
                        await self.place_stop_loss(futures_symbol, "BUY" if direction == "LONG" else "SELL", 
                                                 abs(float(position['positionAmt'])), stop_loss, update)
                        
                        # Try to place individual TP orders
                        individual_tp_success = False
                        tp_orders = []
                        tp_errors = []
                        
                        # Calculate TP quantities
                        position_quantity = abs(float(position['positionAmt']))
                        tp_quantities = []
                        for dist_pct in TP_DISTRIBUTION:
                            tp_qty = position_quantity * dist_pct
                            
                            # Get exchange info for precision
                            exchange_info = self.binance_client.futures_exchange_info()
                            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == futures_symbol), None)
                            quantity_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                            min_qty = float(quantity_filter['minQty'])
                            step_size = float(quantity_filter['stepSize'])
                            
                            tp_qty = self.round_step_size(tp_qty, min_qty, step_size)
                            tp_quantities.append(tp_qty)
                        
                        # Set the TP side (opposite of entry side)
                        tp_side = "SELL" if direction == "LONG" else "BUY"
                        
                        # Check if individual TPs would meet minimum notional
                        individual_tp_viable = True
                        for i, (tp_level, tp_qty) in enumerate(zip(take_profit_levels[:3], tp_quantities)):
                            tp_notional = tp_qty * tp_level
                            if tp_notional < MIN_NOTIONAL_VALUE:
                                individual_tp_viable = False
                                logger.warning(f"Individual TP{i+1} would have notional value {tp_notional:.2f} USDT, below minimum {MIN_NOTIONAL_VALUE} USDT")
                                if update:
                                    await update.message.reply_text(f"⚠️ TP{i+1} individuel aurait une valeur notionnelle de {tp_notional:.2f} USDT, inférieure au minimum de {MIN_NOTIONAL_VALUE} USDT")
                        
                        # If individual TPs look viable, try to place them
                        if individual_tp_viable:
                            try:
                                # Place individual TP orders
                                for i, (tp_level, tp_qty) in enumerate(zip(take_profit_levels[:3], tp_quantities)):
                                    try:
                                        tp_order = self.binance_client.futures_create_order(
                                            symbol=futures_symbol,
                                            side=tp_side,
                                            type='TAKE_PROFIT_MARKET',
                                            quantity=tp_qty,
                                            stopPrice=tp_level,
                                            timeInForce='GTC'
                                        )
                                        tp_orders.append(tp_order)
                                        logger.info(f"TP{i+1} order placed: {json.dumps(tp_order)}")
                                        if update:
                                            await update.message.reply_text(f"✅ TP{i+1} placé: {tp_qty} @ {tp_level}")
                                        
                                        # Add delay between orders
                                        await asyncio.sleep(0.5)
                                        
                                    except Exception as e:
                                        error_msg = f"Erreur lors du placement de TP{i+1}: {str(e)}"
                                        logger.error(error_msg)
                                        tp_errors.append(error_msg)
                                        if update:
                                            await update.message.reply_text(f"⚠️ {error_msg}")
                                
                                if len(tp_orders) == 3:
                                    individual_tp_success = True
                                    logger.info("All individual TP orders placed successfully")
                                    if update:
                                        await update.message.reply_text("✅ Tous les ordres TP individuels placés avec succès")
                                
                            except Exception as e:
                                logger.error(f"Error placing individual TP orders: {str(e)}")
                                if update:
                                    await update.message.reply_text(f"⚠️ Erreur lors du placement des TP individuels: {str(e)}")
                        
                        # If individual TPs failed or weren't viable, try combined approach
                        if not individual_tp_success:
                            logger.info("Trying combined TP approach")
                            if update:
                                await update.message.reply_text("ℹ️ Tentative d'approche TP combinée...")
                            
                            # Group TPs into 2 groups
                            tp_groups = [
                                {
                                    "name": "TP1-3",
                                    "price": sum(take_profit_levels[:3]) / 3,  # Average of first 3 TPs
                                    "quantity": sum(tp_quantities)  # Sum of quantities for first 3 TPs
                                }
                            ]
                            
                            # Try to place combined TP orders
                            combined_tp_orders = []
                            combined_tp_errors = []
                            
                            for group in tp_groups:
                                try:
                                    # Ensure quantity meets minimum notional
                                    group_notional = group["price"] * group["quantity"]
                                    if group_notional < MIN_NOTIONAL_VALUE:
                                        min_group_qty = MIN_NOTIONAL_VALUE / group["price"]
                                        group["quantity"] = max(group["quantity"], self.round_step_size(min_group_qty, min_qty, step_size))
                                        logger.info(f"Increased {group['name']} quantity to {group['quantity']} to meet minimum notional")
                                        if update:
                                            await update.message.reply_text(f"ℹ️ Quantité de {group['name']} augmentée à {group['quantity']} pour atteindre la valeur notionnelle minimale")
                                    
                                    # Place combined TP order
                                    tp_order = self.binance_client.futures_create_order(
                                        symbol=futures_symbol,
                                        side=tp_side,
                                        type='TAKE_PROFIT_MARKET',
                                        quantity=group["quantity"],
                                        stopPrice=group["price"],
                                        timeInForce='GTC'
                                    )
                                    combined_tp_orders.append(tp_order)
                                    logger.info(f"{group['name']} order placed: {json.dumps(tp_order)}")
                                    if update:
                                        await update.message.reply_text(f"✅ {group['name']} placé: {group['quantity']} @ {group['price']}")
                                    
                                except Exception as e:
                                    error_msg = f"Erreur lors du placement de {group['name']}: {str(e)}"
                                    logger.error(error_msg)
                                    combined_tp_errors.append(error_msg)
                                    if update:
                                        await update.message.reply_text(f"⚠️ {error_msg}")
                            
                            if combined_tp_orders:
                                logger.info(f"Combined TP approach: {len(combined_tp_orders)} orders placed successfully")
                                if update:
                                    await update.message.reply_text(f"✅ Approche TP combinée: {len(combined_tp_orders)} ordres placés avec succès")
                                tp_orders = combined_tp_orders
                            else:
                                logger.error("Failed to place any TP orders")
                                if update:
                                    await update.message.reply_text("❌ Échec du placement des ordres TP")
                        
                        # Start progressive monitoring
                        self.monitoring_tasks[futures_symbol] = asyncio.create_task(
                            self.monitor_progressive_trade(
                                symbol,
                                direction,
                                float(position['entryPrice']),
                                stop_loss,
                                take_profit_levels,
                                abs(float(position['positionAmt'])),
                                update
                            )
                        )
                        
                        # End this monitoring task
                        break
                    
                    # Check if limit order still exists
                    open_orders = self.binance_client.futures_get_open_orders(symbol=futures_symbol)
                    limit_orders = [o for o in open_orders if o['type'] == 'LIMIT']
                    
                    if not limit_orders:
                        # Limit order cancelled or expired
                        logger.info(f"Limit order no longer exists, stopping monitoring")
                        if update:
                            await update.message.reply_text(f"ℹ️ L'ordre limite n'existe plus, arrêt du monitoring")
                        break
                    
                    # Wait before checking again
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in limit order monitoring: {str(e)}")
                    if update:
                        await update.message.reply_text(f"⚠️ Erreur dans le monitoring de l'ordre limite: {str(e)}")
                    await asyncio.sleep(60)  # Wait before retrying
        except asyncio.CancelledError:
            logger.info(f"Limit order monitoring task for {futures_symbol} was cancelled")
            if update:
                await update.message.reply_text(f"ℹ️ Tâche de monitoring de l'ordre limite pour {futures_symbol} annulée")
        finally:
            # Remove task from monitoring tasks if it's this one
            if futures_symbol in self.monitoring_tasks and self.monitoring_tasks[futures_symbol].get_name() == asyncio.current_task().get_name():
                del self.monitoring_tasks[futures_symbol]
    
    def round_step_size(self, quantity, min_qty, step_size):
        """Round quantity to valid step size."""
        if step_size == 0:
            return quantity
        
        quantity = max(min_qty, quantity)
        decimal_places = int(round(-math.log10(step_size), 0))
        factor = 10 ** decimal_places
        return math.floor(quantity * factor) / factor
    
    def run(self):
        """Run the bot."""
        logger.info("Starting bot...")
        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main function to run the bot."""
    # Check if environment variables are set
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("Binance API credentials not set. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        return
    
    if not TELEGRAM_TOKEN:
        logger.error("Telegram token not set. Please set TELEGRAM_TOKEN environment variable.")
        return
    
    # Create and run the bot
    bot = TelegramBinanceBot()
    bot.run()

if __name__ == "__main__":
    main()
