"""
Data Pipeline Module for the Autonomous Trading Bot

This module handles data collection, processing, and storage for the trading bot.
It interfaces with the Binance API to fetch market data and provides methods for
feature engineering and data preprocessing.
"""

import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from binance import Client, BinanceSocketManager
from binance.exceptions import BinanceAPIException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Data Pipeline for collecting and processing market data.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        """
        Initialize the data pipeline.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet
        """
        # Initialize Binance client
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Initialize data storage
        self.klines_data = {}
        self.orderbook_data = {}
        self.trades_data = {}
        
        # Initialize feature data
        self.features = {}
        
        # Initialize websocket connections
        self.socket_manager = None
        self.socket_connections = {}
        self.socket_running = False
        
        # Initialize callbacks
        self.data_callbacks = []
        
        logger.info("DataPipeline initialized")
    
    async def start(self, symbols: List[str], intervals: List[str]):
        """
        Start the data pipeline.
        
        Args:
            symbols: List of symbols to collect data for
            intervals: List of intervals to collect data for
        """
        logger.info(f"Starting data pipeline for symbols: {symbols}, intervals: {intervals}")
        
        # Initialize data structures
        for symbol in symbols:
            self.klines_data[symbol] = {}
            for interval in intervals:
                self.klines_data[symbol][interval] = []
            
            self.orderbook_data[symbol] = {}
            self.trades_data[symbol] = []
        
        # Start websocket connections
        self.socket_running = True
        for symbol in symbols:
            await self._start_websocket_connections(symbol)
    
    async def stop(self):
        """
        Stop the data pipeline.
        """
        logger.info("Stopping data pipeline")
        self.socket_running = False
        
        # Close websocket connections
        if self.socket_manager:
            try:
                # For newer versions of python-binance
                if hasattr(self.socket_manager, 'stop_socket'):
                    for conn_key in self.socket_connections:
                        self.socket_manager.stop_socket(self.socket_connections[conn_key])
                
                # Close the socket manager
                if hasattr(self.socket_manager, 'close'):
                    self.socket_manager.close()
                elif hasattr(self.socket_manager, 'stop'):
                    self.socket_manager.stop()
                
            except Exception as e:
                logger.error(f"Error stopping socket manager: {str(e)}")
    
    async def _start_websocket_connections(self, symbol: str):
        """
        Start websocket connections for a symbol.
        
        Args:
            symbol: Symbol to start connections for
        """
        try:
            # Initialize socket manager if not already initialized
            if not self.socket_manager:
                self.socket_manager = BinanceSocketManager(self.client)
            
            # Start kline socket
            # Updated for compatibility with newer python-binance versions
            if hasattr(self.socket_manager, 'start_kline_socket'):
                # For older versions
                kline_socket = self.socket_manager.start_kline_socket(
                    symbol=symbol.lower(),
                    callback=self._handle_kline_message,
                    interval='1m'
                )
                self.socket_connections[f"{symbol}_kline"] = kline_socket
            elif hasattr(self.socket_manager, 'start_kline_futures_socket'):
                # For newer versions with futures support
                kline_socket = self.socket_manager.start_kline_futures_socket(
                    symbol=symbol.lower(),
                    callback=self._handle_kline_message,
                    interval='1m'
                )
                self.socket_connections[f"{symbol}_kline"] = kline_socket
            else:
                # For newest versions using multiplex streams
                streams = [f"{symbol.lower()}@kline_1m"]
                multiplex_socket = self.socket_manager.start_multiplex_socket(
                    streams=streams,
                    callback=self._handle_multiplex_message
                )
                self.socket_connections[f"{symbol}_multiplex"] = multiplex_socket
            
            # Start depth socket
            if hasattr(self.socket_manager, 'start_depth_socket'):
                # For older versions
                depth_socket = self.socket_manager.start_depth_socket(
                    symbol=symbol.lower(),
                    callback=self._handle_depth_message,
                    depth=BinanceSocketManager.WEBSOCKET_DEPTH_5
                )
                self.socket_connections[f"{symbol}_depth"] = depth_socket
            elif hasattr(self.socket_manager, 'start_symbol_book_ticker_socket'):
                # For newer versions
                depth_socket = self.socket_manager.start_symbol_book_ticker_socket(
                    symbol=symbol.lower(),
                    callback=self._handle_book_ticker_message
                )
                self.socket_connections[f"{symbol}_depth"] = depth_socket
            
            # Start trade socket
            if hasattr(self.socket_manager, 'start_trade_socket'):
                # For older versions
                trade_socket = self.socket_manager.start_trade_socket(
                    symbol=symbol.lower(),
                    callback=self._handle_trade_message
                )
                self.socket_connections[f"{symbol}_trade"] = trade_socket
            elif hasattr(self.socket_manager, 'start_aggtrade_socket'):
                # For newer versions
                trade_socket = self.socket_manager.start_aggtrade_socket(
                    symbol=symbol.lower(),
                    callback=self._handle_aggtrade_message
                )
                self.socket_connections[f"{symbol}_trade"] = trade_socket
            
            # Start the socket manager
            if hasattr(self.socket_manager, 'start'):
                self.socket_manager.start()
            
            logger.info(f"Started websocket connections for {symbol}")
            
        except Exception as e:
            logger.error(f"Error starting websocket connections: {str(e)}")
    
    def _handle_multiplex_message(self, msg):
        """
        Handle multiplex message from websocket.
        
        Args:
            msg: Message from websocket
        """
        try:
            # Extract stream name and data
            stream = msg.get('stream', '')
            data = msg.get('data', {})
            
            # Handle different stream types
            if 'kline' in stream:
                self._handle_kline_message(data)
            elif 'depth' in stream or 'bookTicker' in stream:
                self._handle_depth_message(data)
            elif 'trade' in stream or 'aggTrade' in stream:
                self._handle_trade_message(data)
            
        except Exception as e:
            logger.error(f"Error handling multiplex message: {str(e)}")
    
    def _handle_kline_message(self, msg):
        """
        Handle kline message from websocket.
        
        Args:
            msg: Message from websocket
        """
        try:
            # Extract data from message
            symbol = msg['s']
            interval = msg['k']['i']
            is_closed = msg['k']['x']
            
            # Only process closed candles
            if is_closed:
                kline = {
                    'timestamp': msg['k']['t'],
                    'open': float(msg['k']['o']),
                    'high': float(msg['k']['h']),
                    'low': float(msg['k']['l']),
                    'close': float(msg['k']['c']),
                    'volume': float(msg['k']['v']),
                    'close_time': msg['k']['T'],
                    'quote_asset_volume': float(msg['k']['q']),
                    'number_of_trades': msg['k']['n'],
                    'taker_buy_base_asset_volume': float(msg['k']['V']),
                    'taker_buy_quote_asset_volume': float(msg['k']['Q'])
                }
                
                # Store kline data
                if symbol in self.klines_data and interval in self.klines_data[symbol]:
                    self.klines_data[symbol][interval].append(kline)
                    
                    # Keep only the last 1000 klines
                    if len(self.klines_data[symbol][interval]) > 1000:
                        self.klines_data[symbol][interval] = self.klines_data[symbol][interval][-1000:]
                
                # Update features
                self._update_features(symbol)
                
                # Notify callbacks
                self._notify_callbacks()
        
        except Exception as e:
            logger.error(f"Error handling kline message: {str(e)}")
    
    def _handle_depth_message(self, msg):
        """
        Handle depth message from websocket.
        
        Args:
            msg: Message from websocket
        """
        try:
            # Extract data from message
            symbol = msg['s'] if 's' in msg else msg.get('symbol', '')
            
            # Store orderbook data
            if symbol in self.orderbook_data:
                self.orderbook_data[symbol] = {
                    'bids': [[float(price), float(qty)] for price, qty in msg.get('bids', [])],
                    'asks': [[float(price), float(qty)] for price, qty in msg.get('asks', [])],
                    'timestamp': msg.get('E', datetime.now().timestamp() * 1000)
                }
        
        except Exception as e:
            logger.error(f"Error handling depth message: {str(e)}")
    
    def _handle_book_ticker_message(self, msg):
        """
        Handle book ticker message from websocket.
        
        Args:
            msg: Message from websocket
        """
        try:
            # Extract data from message
            symbol = msg['s']
            
            # Store orderbook data
            if symbol in self.orderbook_data:
                self.orderbook_data[symbol] = {
                    'bids': [[float(msg['b']), float(msg['B'])]],
                    'asks': [[float(msg['a']), float(msg['A'])]],
                    'timestamp': msg.get('E', datetime.now().timestamp() * 1000)
                }
        
        except Exception as e:
            logger.error(f"Error handling book ticker message: {str(e)}")
    
    def _handle_trade_message(self, msg):
        """
        Handle trade message from websocket.
        
        Args:
            msg: Message from websocket
        """
        try:
            # Extract data from message
            symbol = msg['s']
            
            # Store trade data
            if symbol in self.trades_data:
                trade = {
                    'id': msg['t'],
                    'price': float(msg['p']),
                    'qty': float(msg['q']),
                    'time': msg['T'],
                    'is_buyer_maker': msg['m'],
                    'is_best_match': msg.get('M', True)
                }
                
                self.trades_data[symbol].append(trade)
                
                # Keep only the last 1000 trades
                if len(self.trades_data[symbol]) > 1000:
                    self.trades_data[symbol] = self.trades_data[symbol][-1000:]
        
        except Exception as e:
            logger.error(f"Error handling trade message: {str(e)}")
    
    def _handle_aggtrade_message(self, msg):
        """
        Handle aggregate trade message from websocket.
        
        Args:
            msg: Message from websocket
        """
        try:
            # Extract data from message
            symbol = msg['s']
            
            # Store trade data
            if symbol in self.trades_data:
                trade = {
                    'id': msg['a'],
                    'price': float(msg['p']),
                    'qty': float(msg['q']),
                    'time': msg['T'],
                    'is_buyer_maker': msg['m'],
                    'is_best_match': msg.get('M', True)
                }
                
                self.trades_data[symbol].append(trade)
                
                # Keep only the last 1000 trades
                if len(self.trades_data[symbol]) > 1000:
                    self.trades_data[symbol] = self.trades_data[symbol][-1000:]
        
        except Exception as e:
            logger.error(f"Error handling aggregate trade message: {str(e)}")
    
    def _update_features(self, symbol: str):
        """
        Update features for a symbol.
        
        Args:
            symbol: Symbol to update features for
        """
        try:
            # Get latest kline data
            if symbol in self.klines_data and '1m' in self.klines_data[symbol]:
                klines = self.klines_data[symbol]['1m']
                
                if len(klines) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(klines)
                    
                    # Calculate technical indicators
                    features = {}
                    
                    # Price features
                    features['close'] = df['close'].iloc[-1]
                    features['open'] = df['open'].iloc[-1]
                    features['high'] = df['high'].iloc[-1]
                    features['low'] = df['low'].iloc[-1]
                    features['volume'] = df['volume'].iloc[-1]
                    
                    # Moving averages
                    if len(df) >= 20:
                        features['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
                    else:
                        features['sma_20'] = df['close'].mean()
                    
                    if len(df) >= 50:
                        features['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
                    else:
                        features['sma_50'] = df['close'].mean()
                    
                    # Relative strength index (RSI)
                    if len(df) >= 14:
                        delta = df['close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(14).mean()
                        loss = -delta.where(delta < 0, 0).rolling(14).mean()
                        rs = gain / loss
                        features['rsi_14'] = 100 - (100 / (1 + rs.iloc[-1]))
                    else:
                        features['rsi_14'] = 50
                    
                    # Bollinger Bands
                    if len(df) >= 20:
                        sma_20 = df['close'].rolling(20).mean()
                        std_20 = df['close'].rolling(20).std()
                        features['bb_upper'] = sma_20.iloc[-1] + 2 * std_20.iloc[-1]
                        features['bb_lower'] = sma_20.iloc[-1] - 2 * std_20.iloc[-1]
                        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20.iloc[-1]
                    else:
                        features['bb_upper'] = df['close'].mean() + 2 * df['close'].std()
                        features['bb_lower'] = df['close'].mean() - 2 * df['close'].std()
                        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / df['close'].mean()
                    
                    # MACD
                    if len(df) >= 26:
                        ema_12 = df['close'].ewm(span=12).mean()
                        ema_26 = df['close'].ewm(span=26).mean()
                        features['macd'] = ema_12.iloc[-1] - ema_26.iloc[-1]
                        features['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
                        features['macd_hist'] = features['macd'] - features['macd_signal']
                    else:
                        features['macd'] = 0
                        features['macd_signal'] = 0
                        features['macd_hist'] = 0
                    
                    # Store features
                    self.features[symbol] = features
        
        except Exception as e:
            logger.error(f"Error updating features: {str(e)}")
    
    def _notify_callbacks(self):
        """
        Notify callbacks of new data.
        """
        for callback in self.data_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in callback: {str(e)}")
    
    def register_callback(self, callback: Callable):
        """
        Register a callback for new data.
        
        Args:
            callback: Callback function
        """
        self.data_callbacks.append(callback)
    
    def get_latest_kline(self, symbol: str, interval: str = '1m') -> Optional[Dict[str, Any]]:
        """
        Get the latest kline for a symbol and interval.
        
        Args:
            symbol: Symbol to get kline for
            interval: Interval to get kline for
            
        Returns:
            Latest kline or None if not available
        """
        if symbol in self.klines_data and interval in self.klines_data[symbol]:
            klines = self.klines_data[symbol][interval]
            if len(klines) > 0:
                return klines[-1]
        
        return None
    
    def get_latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Alias for get_features to maintain compatibility with RL model.
        
        Args:
            symbol: Symbol to get features for
            
        Returns:
            Features or None if not available
        """
        return self.get_features(symbol)


    def get_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest orderbook for a symbol.
        
        Args:
            symbol: Symbol to get orderbook for
            
        Returns:
            Latest orderbook or None if not available
        """
        if symbol in self.orderbook_data:
            return self.orderbook_data[symbol]
        
        return None
    
    def get_latest_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the latest trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            limit: Maximum number of trades to return
            
        Returns:
            List of latest trades
        """
        if symbol in self.trades_data:
            trades = self.trades_data[symbol]
            return trades[-limit:] if len(trades) > 0 else []
        
        return []
    
    def get_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get features for a symbol.
        
        Args:
            symbol: Symbol to get features for
            
        Returns:
            Features or None if not available
        """
        if symbol in self.features:
            return self.features[symbol]
        
        return None
    
    def get_historical_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        """
        Get historical klines from Binance.
        
        Args:
            symbol: Symbol to get klines for
            interval: Interval to get klines for
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of klines
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time
            )
            
            return klines
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {str(e)}")
            return []
        
        except Exception as e:
            logger.error(f"Error getting historical klines: {str(e)}")
            return []
    
    def save_data(self, symbol: str, interval: str, filename: str):
        """
        Save kline data to a file.
        
        Args:
            symbol: Symbol to save data for
            interval: Interval to save data for
            filename: Filename to save data to
        """
        try:
            if symbol in self.klines_data and interval in self.klines_data[symbol]:
                klines = self.klines_data[symbol][interval]
                
                if len(klines) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(klines)
                    
                    # Save to CSV
                    df.to_csv(filename, index=False)
                    
                    logger.info(f"Saved {len(klines)} klines to {filename}")
                else:
                    logger.warning(f"No klines to save for {symbol} {interval}")
            else:
                logger.warning(f"No data for {symbol} {interval}")
        
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def load_data(self, symbol: str, interval: str, filename: str) -> bool:
        """
        Load kline data from a file.
        
        Args:
            symbol: Symbol to load data for
            interval: Interval to load data for
            filename: Filename to load data from
            
        Returns:
            True if data was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(filename):
                # Load from CSV
                df = pd.read_csv(filename)
                
                # Convert to list of dictionaries
                klines = df.to_dict('records')
                
                # Store kline data
                if symbol not in self.klines_data:
                    self.klines_data[symbol] = {}
                
                self.klines_data[symbol][interval] = klines
                
                logger.info(f"Loaded {len(klines)} klines from {filename}")
                
                # Update features
                self._update_features(symbol)
                
                return True
            else:
                logger.warning(f"File not found: {filename}")
                return False
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
