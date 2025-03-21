"""
Data Pipeline Module for the Autonomous Trading Bot

This module is responsible for collecting, processing, and storing market data
from Binance Futures API. It provides both real-time and historical data access
for the RL model.
"""

import os
import time
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.websockets import BinanceSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Data Pipeline for collecting and processing market data from Binance.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize the DataPipeline with Binance API credentials.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use Binance testnet (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize Binance client
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Data storage
        self.ohlcv_data = {}  # Symbol -> DataFrame mapping for OHLCV data
        self.orderbook_data = {}  # Symbol -> DataFrame mapping for orderbook data
        self.trade_data = {}  # Symbol -> DataFrame mapping for trade data
        
        # WebSocket connections
        self.socket_manager = None
        self.active_connections = {}
        
        # Data processing parameters
        self.feature_window = 30  # Number of time periods to use for feature calculation
        
        logger.info("DataPipeline initialized")
    
    async def start(self, symbols: List[str], intervals: List[str] = ['1m']):
        """
        Start the data pipeline for the specified symbols and intervals.
        
        Args:
            symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            intervals: List of time intervals (e.g., ['1m', '5m', '15m'])
        """
        logger.info(f"Starting data pipeline for symbols: {symbols}, intervals: {intervals}")
        
        # Initialize WebSocket manager
        self.socket_manager = BinanceSocketManager(self.client)
        
        # Start data collection for each symbol and interval
        for symbol in symbols:
            # Initialize data storage for this symbol
            self.ohlcv_data[symbol] = {}
            self.orderbook_data[symbol] = pd.DataFrame()
            self.trade_data[symbol] = pd.DataFrame()
            
            # Start WebSocket connections
            await self._start_websocket_connections(symbol)
            
            # Fetch initial historical data
            for interval in intervals:
                await self._fetch_historical_ohlcv(symbol, interval)
        
        logger.info("Data pipeline started successfully")
    
    async def stop(self):
        """
        Stop the data pipeline and close all connections.
        """
        logger.info("Stopping data pipeline")
        
        # Close WebSocket connections
        if self.socket_manager:
            for conn_key in self.active_connections:
                self.socket_manager.stop_socket(self.active_connections[conn_key])
            self.socket_manager.close()
            self.socket_manager = None
        
        logger.info("Data pipeline stopped successfully")
    
    async def _start_websocket_connections(self, symbol: str):
        """
        Start WebSocket connections for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
        """
        # Kline/Candlestick WebSocket
        kline_socket = self.socket_manager.start_kline_socket(
            symbol.lower(),
            self._handle_kline_message,
            interval='1m'
        )
        self.active_connections[f"{symbol}_kline"] = kline_socket
        
        # Depth (Order Book) WebSocket
        depth_socket = self.socket_manager.start_depth_socket(
            symbol.lower(),
            self._handle_depth_message,
            depth=20  # Top 20 bids and asks
        )
        self.active_connections[f"{symbol}_depth"] = depth_socket
        
        # Trade WebSocket
        trade_socket = self.socket_manager.start_trade_socket(
            symbol.lower(),
            self._handle_trade_message
        )
        self.active_connections[f"{symbol}_trade"] = trade_socket
        
        # Start the WebSocket manager
        self.socket_manager.start()
        
        logger.info(f"WebSocket connections started for {symbol}")
    
    def _handle_kline_message(self, msg):
        """
        Handle incoming kline/candlestick WebSocket messages.
        
        Args:
            msg: WebSocket message
        """
        try:
            # Extract data from message
            symbol = msg['s']
            interval = msg['k']['i']
            is_closed = msg['k']['x']
            
            # Only process closed candles
            if is_closed:
                kline_data = {
                    'timestamp': msg['k']['t'],
                    'open': float(msg['k']['o']),
                    'high': float(msg['k']['h']),
                    'low': float(msg['k']['l']),
                    'close': float(msg['k']['c']),
                    'volume': float(msg['k']['v']),
                    'close_time': msg['k']['T'],
                    'quote_asset_volume': float(msg['k']['q']),
                    'number_of_trades': int(msg['k']['n']),
                    'taker_buy_base_asset_volume': float(msg['k']['V']),
                    'taker_buy_quote_asset_volume': float(msg['k']['Q'])
                }
                
                # Create DataFrame from kline data
                df = pd.DataFrame([kline_data])
                df.set_index('timestamp', inplace=True)
                
                # Initialize interval data if not exists
                if interval not in self.ohlcv_data[symbol]:
                    self.ohlcv_data[symbol][interval] = df
                else:
                    # Append new data
                    self.ohlcv_data[symbol][interval] = pd.concat([self.ohlcv_data[symbol][interval], df])
                    # Remove duplicates
                    self.ohlcv_data[symbol][interval] = self.ohlcv_data[symbol][interval].loc[~self.ohlcv_data[symbol][interval].index.duplicated(keep='last')]
                    # Sort by timestamp
                    self.ohlcv_data[symbol][interval].sort_index(inplace=True)
                
                logger.debug(f"Processed kline data for {symbol} {interval}")
        
        except Exception as e:
            logger.error(f"Error processing kline message: {str(e)}")
    
    def _handle_depth_message(self, msg):
        """
        Handle incoming depth (order book) WebSocket messages.
        
        Args:
            msg: WebSocket message
        """
        try:
            # Extract data from message
            symbol = msg['s'] if 's' in msg else None
            
            if symbol:
                timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                
                # Process bids and asks
                bids = np.array(msg['b'], dtype=float)  # [[price, quantity], ...]
                asks = np.array(msg['a'], dtype=float)  # [[price, quantity], ...]
                
                # Calculate order book features
                mid_price = (bids[0][0] + asks[0][0]) / 2
                spread = asks[0][0] - bids[0][0]
                bid_volume = np.sum(bids[:, 1])
                ask_volume = np.sum(asks[:, 1])
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                
                # Create DataFrame from order book features
                df = pd.DataFrame([{
                    'timestamp': timestamp,
                    'mid_price': mid_price,
                    'spread': spread,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'imbalance': imbalance,
                    'best_bid': bids[0][0],
                    'best_ask': asks[0][0],
                    'best_bid_volume': bids[0][1],
                    'best_ask_volume': asks[0][1]
                }])
                df.set_index('timestamp', inplace=True)
                
                # Append new data
                self.orderbook_data[symbol] = pd.concat([self.orderbook_data[symbol], df])
                # Keep only recent data (last 1000 updates)
                self.orderbook_data[symbol] = self.orderbook_data[symbol].iloc[-1000:]
                # Sort by timestamp
                self.orderbook_data[symbol].sort_index(inplace=True)
                
                logger.debug(f"Processed depth data for {symbol}")
        
        except Exception as e:
            logger.error(f"Error processing depth message: {str(e)}")
    
    def _handle_trade_message(self, msg):
        """
        Handle incoming trade WebSocket messages.
        
        Args:
            msg: WebSocket message
        """
        try:
            # Extract data from message
            symbol = msg['s']
            
            # Create trade data
            trade_data = {
                'timestamp': msg['T'],
                'price': float(msg['p']),
                'quantity': float(msg['q']),
                'is_buyer_maker': msg['m'],
                'trade_id': msg['t']
            }
            
            # Create DataFrame from trade data
            df = pd.DataFrame([trade_data])
            df.set_index('timestamp', inplace=True)
            
            # Append new data
            self.trade_data[symbol] = pd.concat([self.trade_data[symbol], df])
            # Keep only recent trades (last 1000 trades)
            self.trade_data[symbol] = self.trade_data[symbol].iloc[-1000:]
            # Sort by timestamp
            self.trade_data[symbol].sort_index(inplace=True)
            
            logger.debug(f"Processed trade data for {symbol}")
        
        except Exception as e:
            logger.error(f"Error processing trade message: {str(e)}")
    
    async def _fetch_historical_ohlcv(self, symbol: str, interval: str, limit: int = 1000):
        """
        Fetch historical OHLCV data from Binance API.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1m', '5m', '15m')
            limit: Number of candles to fetch (default: 1000, max: 1000)
        """
        try:
            logger.info(f"Fetching historical OHLCV data for {symbol} {interval}")
            
            # Fetch klines from Binance API
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = df['timestamp'].astype(int)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Store in ohlcv_data
            self.ohlcv_data[symbol][interval] = df
            
            logger.info(f"Fetched {len(df)} historical candles for {symbol} {interval}")
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching historical data: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
    
    def get_latest_features(self, symbol: str, interval: str = '1m') -> pd.DataFrame:
        """
        Get the latest features for the RL model.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval for OHLCV data (default: '1m')
            
        Returns:
            DataFrame containing the latest features
        """
        try:
            # Check if we have data for this symbol
            if symbol not in self.ohlcv_data or interval not in self.ohlcv_data[symbol]:
                logger.warning(f"No data available for {symbol} {interval}")
                return None
            
            # Get the latest OHLCV data
            ohlcv = self.ohlcv_data[symbol][interval].iloc[-self.feature_window:]
            
            # Get the latest order book data
            orderbook = self.orderbook_data[symbol].iloc[-1:] if not self.orderbook_data[symbol].empty else None
            
            # Get the latest trade data
            trades = self.trade_data[symbol].iloc[-100:] if not self.trade_data[symbol].empty else None
            
            # Calculate OHLCV features
            ohlcv_features = self._calculate_ohlcv_features(ohlcv)
            
            # Calculate order book features
            ob_features = self._calculate_orderbook_features(orderbook) if orderbook is not None else {}
            
            # Calculate trade features
            trade_features = self._calculate_trade_features(trades) if trades is not None else {}
            
            # Combine all features
            all_features = {**ohlcv_features, **ob_features, **trade_features}
            
            # Convert to DataFrame
            features_df = pd.DataFrame([all_features])
            
            return features_df
        
        except Exception as e:
            logger.error(f"Error getting latest features: {str(e)}")
            return None
    
    def _calculate_ohlcv_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate features from OHLCV data.
        
        Args:
            ohlcv: DataFrame containing OHLCV data
            
        Returns:
            Dictionary of OHLCV features
        """
        features = {}
        
        # Basic price features
        features['close'] = ohlcv['close'].iloc[-1]
        features['open'] = ohlcv['open'].iloc[-1]
        features['high'] = ohlcv['high'].iloc[-1]
        features['low'] = ohlcv['low'].iloc[-1]
        features['volume'] = ohlcv['volume'].iloc[-1]
        
        # Calculate VWAP
        vwap = np.sum(ohlcv['volume'] * ohlcv['close']) / np.sum(ohlcv['volume'])
        features['vwap'] = vwap
        
        # Calculate price changes
        features['price_change_1m'] = ohlcv['close'].pct_change(1).iloc[-1]
        features['price_change_5m'] = ohlcv['close'].pct_change(5).iloc[-1] if len(ohlcv) >= 5 else 0
        features['price_change_15m'] = ohlcv['close'].pct_change(15).iloc[-1] if len(ohlcv) >= 15 else 0
        
        # Calculate volatility
        features['volatility'] = ohlcv['close'].pct_change().std()
        
        # Calculate RSI (14 periods)
        delta = ohlcv['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 50
        
        # Calculate Bollinger Bands (20 periods, 2 standard deviations)
        sma = ohlcv['close'].rolling(window=20).mean().iloc[-1]
        std = ohlcv['close'].rolling(window=20).std().iloc[-1]
        features['bollinger_upper'] = sma + 2 * std
        features['bollinger_lower'] = sma - 2 * std
        features['bollinger_pct'] = (features['close'] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'])
        
        # Calculate MACD (12, 26, 9)
        ema12 = ohlcv['close'].ewm(span=12).mean().iloc[-1]
        ema26 = ohlcv['close'].ewm(span=26).mean().iloc[-1]
        features['macd'] = ema12 - ema26
        features['macd_signal'] = ohlcv['close'].ewm(span=12).mean().ewm(span=9).mean().iloc[-1] - ohlcv['close'].ewm(span=26).mean().ewm(span=9).mean().iloc[-1]
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        return features
    
    def _calculate_orderbook_features(self, orderbook: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate features from order book data.
        
        Args:
            orderbook: DataFrame containing order book data
            
        Returns:
            Dictionary of order book features
        """
        features = {}
        
        # Extract the latest order book snapshot
        ob = orderbook.iloc[0]
        
        # Basic order book features
        features['spread'] = ob['spread']
        features['mid_price'] = ob['mid_price']
        features['bid_volume'] = ob['bid_volume']
        features['ask_volume'] = ob['ask_volume']
        features['imbalance'] = ob['imbalance']
        features['best_bid'] = ob['best_bid']
        features['best_ask'] = ob['best_ask']
        features['best_bid_volume'] = ob['best_bid_volume']
        features['best_ask_volume'] = ob['best_ask_volume']
        
        # Calculate additional features
        features['spread_pct'] = features['spread'] / features['mid_price']
        features['bid_ask_volume_ratio'] = features['bid_volume'] / features['ask_volume'] if features['ask_volume'] > 0 else 1
        
        return features
    
    def _calculate_trade_features(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate features from trade data.
        
        Args:
            trades: DataFrame containing trade data
            
        Returns:
            Dictionary of trade features
        """
        features = {}
        
        # Calculate trade imbalance (buy vs sell volume)
        buy_volume = trades[~trades['is_buyer_maker']]['quantity'].sum()
        sell_volume = trades[trades['is_buyer_maker']]['quantity'].sum()
        total_volume = buy_volume + sell_volume
        
        features['buy_volume'] = buy_volume
        features['sell_volume'] = sell_volume
        features['trade_imbalance'] = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Calculate trade frequency
        features['trade_count'] = len(trades)
        
        # Calculate average trade size
        features['avg_trade_size'] = trades['quantity'].mean()
        
        # Calculate large trade detection (trades > 2 standard deviations from mean)
        mean_size = trades['quantity'].mean()
        std_size = trades['quantity'].std()
        large_trades = trades[trades['quantity'] > mean_size + 2 * std_size]
        features['large_trade_count'] = len(large_trades)
        features['large_trade_volume'] = large_trades['quantity'].sum() if not large_trades.empty else 0
        
        # Calculate price impact (correlation between trade size and price change)
        if len(trades) > 1:
            trades['price_change'] = trades['price'].diff()
            trades['size_times_direction'] = trades['quantity'] * (~trades['is_buyer_maker']).astype(int) * 2 - 1
            features['price_impact'] = trades['price_change'].corr(trades['size_times_direction'])
        else:
            features['price_impact'] = 0
        
        return features
    
    def save_data(self, symbol: str, data_dir: str = 'data'):
        """
        Save collected data to disk.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            data_dir: Directory to save data (default: 'data')
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save OHLCV data
            for interval, df in self.ohlcv_data[symbol].items():
                filename = f"{data_dir}/{symbol}_{interval}_ohlcv.csv"
                df.to_csv(filename)
                logger.info(f"Saved OHLCV data to {filename}")
            
            # Save order book data
            if not self.orderbook_data[symbol].empty:
                filename = f"{data_dir}/{symbol}_orderbook.csv"
                self.orderbook_data[symbol].to_csv(filename)
                logger.info(f"Saved order book data to {filename}")
            
            # Save trade data
            if not self.trade_data[symbol].empty:
                filename = f"{data_dir}/{symbol}_trades.csv"
                self.trade_data[symbol].to_csv(filename)
                logger.info(f"Saved trade data to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def load_data(self, symbol: str, data_dir: str = 'data'):
        """
        Load data from disk.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            data_dir: Directory to load data from (default: 'data')
        """
        try:
            # Initialize data storage for this symbol
            self.ohlcv_data[symbol] = {}
            self.orderbook_data[symbol] = pd.DataFrame()
            self.trade_data[symbol] = pd.DataFrame()
            
            # Load OHLCV data
            for interval in ['1m', '5m', '15m']:
                filename = f"{data_dir}/{symbol}_{interval}_ohlcv.csv"
                if os.path.exists(filename):
                    df = pd.read_csv(filename, index_col='timestamp')
                    self.ohlcv_data[symbol][interval] = df
                    logger.info(f"Loaded OHLCV data from {filename}")
            
            # Load order book data
            filename = f"{data_dir}/{symbol}_orderbook.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col='timestamp')
                self.orderbook_data[symbol] = df
                logger.info(f"Loaded order book data from {filename}")
            
            # Load trade data
            filename = f"{data_dir}/{symbol}_trades.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col='timestamp')
                self.trade_data[symbol] = df
                logger.info(f"Loaded trade data from {filename}")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")

# Example usage
async def main():
    # Get API credentials from environment variables
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("API credentials not found in environment variables")
        return
    
    # Create data pipeline
    pipeline = DataPipeline(api_key, api_secret, testnet=True)
    
    # Start data collection
    await pipeline.start(['BTCUSDT'], ['1m'])
    
    # Wait for some data to be collected
    logger.info("Collecting data for 60 seconds...")
    await asyncio.sleep(60)
    
    # Get latest features
    features = pipeline.get_latest_features('BTCUSDT')
    logger.info(f"Latest features:\n{features}")
    
    # Save data
    pipeline.save_data('BTCUSDT')
    
    # Stop data collection
    await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
