"""
Modified Training Module for the Autonomous Trading Bot

This module provides a modified training approach that uses historical data
instead of real-time data collection for faster and more reliable training.
"""

import os
import logging
import argparse
import asyncio
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from rl_model import PPOAgent, TradingEnvironment
from risk_management import RiskManager, PositionManager
from performance_tracking import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataTrainer:
    """
    Trainer class that uses historical data for training the RL agent.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Default configuration
        self.config = {
            # Training parameters
            "symbols": ["BTCUSDT"],
            "initial_balance": 10000.0,
            "max_position_size": 0.1,
            "transaction_fee": 0.0004,
            
            # RL model parameters
            "state_dim": 30,
            "action_dim": 3,
            "hidden_dims": [128, 64, 32],
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            
            # Risk management parameters
            "max_risk_per_trade": 0.02,
            "max_daily_risk": 0.05,
            "max_drawdown": 0.15,
            
            # System parameters
            "data_dir": "data",
            "models_dir": "models",
            "performance_dir": "performance_data"
        }
        
        # Create directories
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["models_dir"], exist_ok=True)
        os.makedirs(self.config["performance_dir"], exist_ok=True)
        
        # Initialize components
        self.rl_agent = None
        self.risk_manager = None
        self.performance_tracker = None
        
        logger.info("Historical Data Trainer initialized")
    
    def initialize(self):
        """
        Initialize all components.
        """
        try:
            # Initialize risk manager
            self.risk_manager = RiskManager(
                max_position_size=self.config["max_position_size"],
                max_risk_per_trade=self.config["max_risk_per_trade"],
                max_daily_risk=self.config["max_daily_risk"],
                max_drawdown=self.config["max_drawdown"]
            )
            
            # Initialize position manager
            self.position_manager = PositionManager(self.risk_manager)
            
            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker(
                initial_balance=self.config["initial_balance"],
                data_dir=self.config["performance_dir"]
            )
            
            # Initialize RL agent
            self.rl_agent = PPOAgent(
                state_dim=self.config["state_dim"],
                action_dim=self.config["action_dim"],
                hidden_dims=self.config["hidden_dims"],
                lr_policy=self.config["learning_rate"],
                gamma=self.config["gamma"],
                gae_lambda=self.config["gae_lambda"],
                clip_ratio=self.config["clip_ratio"],
                value_coef=self.config["value_coef"],
                entropy_coef=self.config["entropy_coef"]
            )
            
            # Load model if exists
            model_path = os.path.join(self.config["models_dir"], "ppo_agent.pt")
            if os.path.exists(model_path):
                self.rl_agent.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def load_historical_data(self, data_file):
        """
        Load historical data from a file.
        
        Args:
            data_file: Path to historical data file
            
        Returns:
            DataFrame with historical data
        """
        try:
            if not os.path.exists(data_file):
                logger.error(f"Data file not found: {data_file}")
                return None
            
            # Load data from CSV
            df = pd.read_csv(data_file)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.debug(f"Converted timestamp column to datetime: {df['timestamp'].dtype}")
            
            logger.info(f"Loaded historical data from {data_file}")
            logger.debug(f"Historical data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess historical data for training.
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            DataFrame with preprocessed data
        """
        try:
            # Ensure all required columns are present
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column missing: {col}")
                    return None
            
            # Convert numeric columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            # Calculate technical indicators
            # SMA
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # MACD
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Drop NaN values
            df = df.dropna()
            
            logger.info(f"Preprocessed data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None
    
    def create_features(self, df, idx):
        """
        Create feature vector for a specific index in the DataFrame.
        
        Args:
            df: DataFrame with preprocessed data
            idx: Index to create features for
            
        Returns:
            Feature vector
        """
        try:
            # Get current row
            row = df.iloc[idx]
            
            # Create feature vector
            features = np.zeros(self.config["state_dim"])
            
            # Price features
            features[0] = row['open']
            features[1] = row['high']
            features[2] = row['low']
            features[3] = row['close']
            features[4] = row['volume']
            
            # Technical indicators
            features[5] = row['sma_20']
            features[6] = row['sma_50']
            features[7] = row['rsi_14']
            features[8] = row['bb_upper']
            features[9] = row['bb_lower']
            features[10] = row['bb_width']
            features[11] = row['macd']
            features[12] = row['macd_signal']
            features[13] = row['macd_hist']
            
            # Price ratios
            features[14] = row['close'] / row['open'] - 1  # Current candle change
            features[15] = row['close'] / df.iloc[idx-1]['close'] - 1 if idx > 0 else 0  # Previous candle change
            features[16] = row['close'] / row['sma_20'] - 1  # Deviation from SMA20
            features[17] = row['close'] / row['sma_50'] - 1  # Deviation from SMA50
            
            # Volatility
            if idx >= 20:
                features[18] = df.iloc[idx-19:idx+1]['close'].std() / row['close']  # 20-day volatility
            else:
                features[18] = df.iloc[:idx+1]['close'].std() / row['close']
            
            # The remaining features (19-29) will be filled with portfolio state in the training loop
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return np.zeros(self.config["state_dim"])
    
    def train(self, data_file, num_episodes=100, save_interval=10):
        """
        Train the RL agent using historical data.
        
        Args:
            data_file: Path to historical data file
            num_episodes: Number of training episodes
            save_interval: Interval for saving the model
        """
        try:
            # Initialize components
            if not self.initialize():
                logger.error("Failed to initialize components")
                return
            
            # Load and preprocess historical data
            df = self.load_historical_data(data_file)
            if df is None:
                logger.error("Failed to load historical data")
                return
            
            df = self.preprocess_data(df)
            if df is None:
                logger.error("Failed to preprocess data")
                return
            
            logger.info(f"Starting training for {num_episodes} episodes")
            
            # Training loop
            for episode in range(num_episodes):
                logger.info(f"Starting episode {episode+1}/{num_episodes}")
                
                # Reset environment state
                current_balance = self.config["initial_balance"]
                position = 0.0
                entry_price = 0.0
                done = False
                episode_reward = 0.0
                
                # Reset performance tracker
                self.performance_tracker = PerformanceTracker(
                    initial_balance=self.config["initial_balance"],
                    data_dir=self.config["performance_dir"]
                )
                
                # Episode loop - iterate through historical data
                for i in range(len(df) - 1):
                    # Get current and next candle
                    current_candle = df.iloc[i]
                    next_candle = df.iloc[i + 1]
                    
                    # Create state representation
                    state = self.create_features(df, i)
                    
                    # Add portfolio state
                    state[19] = current_balance / self.config["initial_balance"]  # Normalized balance
                    state[20] = position / self.config["max_position_size"]  # Normalized position
                    state[21] = 1 if position > 0 else 0  # Long flag
                    state[22] = 1 if position < 0 else 0  # Short flag
                    state[23] = (current_candle['close'] / entry_price - 1) if position != 0 and entry_price != 0 else 0  # Unrealized PnL
                    
                    # Select action
                    action, action_prob, value = self.rl_agent.select_action(state)
                    
                    # Execute action
                    current_price = current_candle['close']
                    next_price = next_candle['close']
                    
                    # Calculate portfolio value before action
                    portfolio_value_before = current_balance + position * current_price
                    
                    # Initialize reward
                    reward = 0.0
                    
                    if action == 0:  # Hold
                        # No action, just calculate unrealized PnL
                        unrealized_pnl = position * (next_price - current_price)
                        reward = unrealized_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
                        
                    elif action == 1:  # Buy
                        if position < 0:  # Close short position
                            # Calculate realized PnL
                            realized_pnl = -position * (current_price - entry_price)
                            # Apply transaction fee
                            fee = abs(position * current_price * self.config["transaction_fee"])
                            # Update balance
                            current_balance += realized_pnl - fee
                            
                            # Record trade
                            self.performance_tracker.record_trade({
                                'trade_id': f"train_ep{episode+1}_trade_{i}_close_short",
                                'symbol': self.config["symbols"][0],
                                'type': 'short',
                                'entry_time': current_candle['timestamp'] - pd.Timedelta(minutes=10),
                                'exit_time': current_candle['timestamp'],
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'size': abs(position),
                                'pnl': realized_pnl,
                                'fee': fee,
                                'exit_reason': 'signal'
                            })
                            
                            # Reset position
                            position = 0.0
                            entry_price = 0.0
                            
                            # Calculate reward
                            reward = realized_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
                        
                        # Open long position
                        position_size = self.config["max_position_size"] * self.config["initial_balance"] / current_price
                        # Check if we have enough balance
                        if current_balance >= position_size * current_price:
                            # Apply transaction fee
                            fee = position_size * current_price * self.config["transaction_fee"]
                            # Update balance and position
                            current_balance -= position_size * current_price + fee
                            position = position_size
                            entry_price = current_price
                            
                            # No immediate reward for opening position
                            reward = 0.0
                        else:
                            # Not enough balance
                            reward = -0.01  # Small penalty for invalid action
                    
                    elif action == 2:  # Sell
                        if position > 0:  # Close long position
                            # Calculate realized PnL
                            realized_pnl = position * (current_price - entry_price)
                            # Apply transaction fee
                            fee = position * current_price * self.config["transaction_fee"]
                            # Update balance
                            current_balance += position * current_price - fee
                            
                            # Record trade
                            self.performance_tracker.record_trade({
                                'trade_id': f"train_ep{episode+1}_trade_{i}_close_long",
                                'symbol': self.config["symbols"][0],
                                'type': 'long',
                                'entry_time': current_candle['timestamp'] - pd.Timedelta(minutes=10),
                                'exit_time': current_candle['timestamp'],
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'size': position,
                                'pnl': realized_pnl,
                                'fee': fee,
                                'exit_reason': 'signal'
                            })
                            
                            # Reset position
                            position = 0.0
                            entry_price = 0.0
                            
                            # Calculate reward
                            reward = realized_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
                        
                        # Open short position
                        position_size = self.config["max_position_size"] * self.config["initial_balance"] / current_price
                        # Check if we have enough balance
                        if current_balance >= position_size * current_price:
                            # Apply transaction fee
                            fee = position_size * current_price * self.config["transaction_fee"]
                            # Update balance and position
                            current_balance -= fee
                            position = -position_size
                            entry_price = current_price
                            
                            # No immediate reward for opening position
                            reward = 0.0
                        else:
                            # Not enough balance
                            reward = -0.01  # Small penalty for invalid action
                    
                    # Calculate portfolio value after action
                    portfolio_value_after = current_balance + position * next_price
                    
                    # Update performance tracker
                    self.performance_tracker.update_balance(portfolio_value_after, current_candle['timestamp'])
                    
                    # Check if episode is done
                    done = i == len(df) - 2  # Last step
                    
                    # Create next state
                    next_state = self.create_features(df, i + 1)
                    next_state[19] = current_balance / self.config["initial_balance"]
                    next_state[20] = position / self.config["max_position_size"]
                    next_state[21] = 1 if position > 0 else 0
                    next_state[22] = 1 if position < 0 else 0
                    next_state[23] = (next_candle['close'] / entry_price - 1) if position != 0 and entry_price != 0 else 0
                    
                    # Store transition
                    self.rl_agent.store_transition(state, action, action_prob, reward, value, done)
                    
                    # Update episode reward
                    episode_reward += reward
                    
                    # Log progress
                    if i % 100 == 0:
                        logger.debug(f"Episode {episode+1}, Step {i}/{len(df)-1}, Action: {action}, Reward: {reward:.4f}, Portfolio: ${portfolio_value_after:.2f}")
                
                # Close any open position at the end
                if position != 0:
                    final_price = df.iloc[-1]['close']
                    
                    if position > 0:  # Close long position
                        # Calculate realized PnL
                        realized_pnl = position * (final_price - entry_price)
                        # Apply transaction fee
                        fee = position * final_price * self.config["transaction_fee"]
                        # Update balance
                        current_balance += position * final_price - fee
                        
                        # Record trade
                        self.performance_tracker.record_trade({
                            'trade_id': f"train_ep{episode+1}_trade_final_close_long",
                            'symbol': self.config["symbols"][0],
                            'type': 'long',
                            'entry_time': df.iloc[-2]['timestamp'],
                            'exit_time': df.iloc[-1]['timestamp'],
                            'entry_price': entry_price,
                            'exit_price': final_price,
                            'size': position,
                            'pnl': realized_pnl,
                            'fee': fee,
                            'exit_reason': 'end_of_episode'
                        })
                    else:  # Close short position
                        # Calculate realized PnL
                        realized_pnl = -position * (final_price - entry_price)
                        # Apply transaction fee
                        fee = abs(position * final_price * self.config["transaction_fee"])
                        # Update balance
                        current_balance += realized_pnl - fee
                        
                        # Record trade
                        self.performance_tracker.record_trade({
                            'trade_id': f"train_ep{episode+1}_trade_final_close_short",
                            'symbol': self.config["symbols"][0],
                            'type': 'short',
                            'entry_time': df.iloc[-2]['timestamp'],
                            'exit_time': df.iloc[-1]['timestamp'],
                            'entry_price': entry_price,
                            'exit_price': final_price,
                            'size': abs(position),
                            'pnl': realized_pnl,
                            'fee': fee,
                            'exit_reason': 'end_of_episode'
                        })
                
                # Update final balance
                self.performance_tracker.update_balance(current_balance, df.iloc[-1]['timestamp'])
                
                # Update agent
                next_value = 0  # Terminal state value
                metrics = self.rl_agent.update(next_value)
                
                # Calculate performance metrics
                final_balance = current_balance + position * df.iloc[-1]['close']
                roi = (final_balance / self.config["initial_balance"] - 1) * 100
                
                logger.info(f"Episode {episode+1} finished")
                logger.info(f"Episode reward: {episode_reward:.4f}")
                logger.info(f"Final balance: ${final_balance:.2f} (ROI: {roi:.2f}%)")
                logger.info(f"Training metrics: {metrics}")
                
                # Save model at intervals
                if (episode + 1) % save_interval == 0:
                    model_path = os.path.join(self.config["models_dir"], f"ppo_agent_ep{episode+1}.pt")
                    self.rl_agent.save_model(model_path)
                    logger.info(f"Model saved to {model_path}")
            
            # Save final model
            model_path = os.path.join(self.config["models_dir"], "ppo_agent.pt")
            self.rl_agent.save_model(model_path)
            logger.info(f"Final model saved to {model_path}")
            
            logger.info("Training completed")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

def main():
    """
    Main function to run the historical data trainer.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Historical Data Trainer for Autonomous Trading Bot')
    parser.add_argument('--data-file', type=str, required=True, help='Path to historical data file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=10, help='Interval for saving the model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = HistoricalDataTrainer()
    
    # Train the RL agent
    trainer.train(args.data_file, args.episodes, args.save_interval)

if __name__ == "__main__":
    main()
