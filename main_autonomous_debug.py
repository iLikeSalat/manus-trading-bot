"""
Main Module for the Autonomous Trading Bot with Enhanced Logging

This module integrates all components of the autonomous trading system and
provides the main entry point for running the bot.
"""

import os
import logging
import asyncio
import argparse
import json
import numpy as np
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import torch

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot_detailed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add a separate debug logger for detailed diagnostics
debug_logger = logging.getLogger('debug')
debug_handler = logging.FileHandler("debug_trace.log")
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
debug_handler.setFormatter(debug_formatter)
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.DEBUG)

class AutonomousTradingBot:
    """
    Main class for the Autonomous Trading Bot.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Autonomous Trading Bot.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Autonomous Trading Bot")
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_pipeline = None
        self.rl_agent = None
        self.trading_env = None
        self.risk_manager = None
        self.position_manager = None
        self.performance_tracker = None
        self.performance_dashboard = None
        
        # Bot state
        self.is_running = False
        self.is_training = False
        self.is_backtesting = False
        
        logger.info("Autonomous Trading Bot initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        logger.info(f"Loading configuration from {config_path if config_path else 'defaults'}")
        # Default configuration
        default_config = {
            # API credentials
            "binance_api_key": os.environ.get("BINANCE_API_KEY", ""),
            "binance_api_secret": os.environ.get("BINANCE_API_SECRET", ""),
            "testnet": True,
            
            # Trading parameters
            "symbols": ["BTCUSDT"],
            "intervals": ["1m"],
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
            "enable_dynamic_sizing": True,
            "enable_trailing_stops": True,
            
            # Performance tracking parameters
            "log_trades": True,
            "log_metrics": True,
            "log_interval": 60,
            "visualization_enabled": True,
            
            # System parameters
            "data_dir": "data",
            "models_dir": "models",
            "performance_dir": "performance_data",
            "dashboard_dir": "dashboard"
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            try:
                logger.info(f"Reading configuration file: {config_path}")
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Update default config with file config
                default_config.update(file_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
                debug_logger.error(f"Config loading error details: {traceback.format_exc()}")
        
        return default_config
    
    # Synchronous initialization method for use in backtesting
    def initialize_sync(self):
        """
        Initialize all components of the trading bot synchronously.
        """
        logger.info("Starting synchronous initialization for backtesting")
        try:
            # Create directories
            logger.debug("Creating necessary directories")
            os.makedirs(self.config["data_dir"], exist_ok=True)
            os.makedirs(self.config["models_dir"], exist_ok=True)
            os.makedirs(self.config["performance_dir"], exist_ok=True)
            os.makedirs(self.config["dashboard_dir"], exist_ok=True)
            
            # Initialize data pipeline
            logger.debug("Initializing data pipeline")
            try:
                from data_pipeline import DataPipeline
                self.data_pipeline = DataPipeline(
                    api_key=self.config["binance_api_key"],
                    api_secret=self.config["binance_api_secret"],
                    testnet=self.config["testnet"]
                )
            except Exception as e:
                logger.error(f"Error initializing data pipeline: {str(e)}")
                debug_logger.error(f"Data pipeline initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize risk manager
            logger.debug("Initializing risk manager")
            try:
                from risk_management import RiskManager, PositionManager
                self.risk_manager = RiskManager(
                    max_position_size=self.config["max_position_size"],
                    max_risk_per_trade=self.config["max_risk_per_trade"],
                    max_daily_risk=self.config["max_daily_risk"],
                    max_drawdown=self.config["max_drawdown"],
                    enable_dynamic_sizing=self.config["enable_dynamic_sizing"],
                    enable_trailing_stops=self.config["enable_trailing_stops"]
                )
            except Exception as e:
                logger.error(f"Error initializing risk manager: {str(e)}")
                debug_logger.error(f"Risk manager initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize position manager
            logger.debug("Initializing position manager")
            try:
                self.position_manager = PositionManager(self.risk_manager)
            except Exception as e:
                logger.error(f"Error initializing position manager: {str(e)}")
                debug_logger.error(f"Position manager initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize performance tracker
            logger.debug("Initializing performance tracker")
            try:
                from performance_tracking import PerformanceTracker, PerformanceDashboard
                self.performance_tracker = PerformanceTracker(
                    initial_balance=self.config["initial_balance"],
                    data_dir=self.config["performance_dir"],
                    log_trades=self.config["log_trades"],
                    log_metrics=self.config["log_metrics"],
                    log_interval=self.config["log_interval"],
                    visualization_enabled=self.config["visualization_enabled"]
                )
            except Exception as e:
                logger.error(f"Error initializing performance tracker: {str(e)}")
                debug_logger.error(f"Performance tracker initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize performance dashboard
            logger.debug("Initializing performance dashboard")
            try:
                self.performance_dashboard = PerformanceDashboard(
                    performance_tracker=self.performance_tracker,
                    dashboard_dir=self.config["dashboard_dir"]
                )
            except Exception as e:
                logger.error(f"Error initializing performance dashboard: {str(e)}")
                debug_logger.error(f"Performance dashboard initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize RL agent
            logger.debug("Initializing RL agent")
            try:
                from rl_model import PPOAgent, TradingEnvironment
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
            except Exception as e:
                logger.error(f"Error initializing RL agent: {str(e)}")
                debug_logger.error(f"RL agent initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize trading environment
            logger.debug("Initializing trading environment")
            try:
                self.trading_env = TradingEnvironment(
                    data_pipeline=self.data_pipeline,
                    initial_balance=self.config["initial_balance"],
                    transaction_fee=self.config["transaction_fee"],
                    max_position=self.config["max_position_size"]
                )
            except Exception as e:
                logger.error(f"Error initializing trading environment: {str(e)}")
                debug_logger.error(f"Trading environment initialization error: {traceback.format_exc()}")
                return False
            
            # Load model if exists
            model_path = os.path.join(self.config["models_dir"], "ppo_agent.pt")
            if os.path.exists(model_path):
                try:
                    logger.debug(f"Loading model from {model_path}")
                    self.rl_agent.load_model(model_path)
                    logger.info(f"Model loaded from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    debug_logger.error(f"Model loading error: {traceback.format_exc()}")
                    # Continue even if model loading fails
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            debug_logger.error(f"Component initialization error: {traceback.format_exc()}")
            return False
    
    async def initialize(self):
        """
        Initialize all components of the trading bot asynchronously.
        """
        logger.info("Starting asynchronous initialization")
        try:
            # Create directories
            logger.debug("Creating necessary directories")
            os.makedirs(self.config["data_dir"], exist_ok=True)
            os.makedirs(self.config["models_dir"], exist_ok=True)
            os.makedirs(self.config["performance_dir"], exist_ok=True)
            os.makedirs(self.config["dashboard_dir"], exist_ok=True)
            
            # Initialize data pipeline
            logger.debug("Initializing data pipeline")
            try:
                from data_pipeline import DataPipeline
                self.data_pipeline = DataPipeline(
                    api_key=self.config["binance_api_key"],
                    api_secret=self.config["binance_api_secret"],
                    testnet=self.config["testnet"]
                )
            except Exception as e:
                logger.error(f"Error initializing data pipeline: {str(e)}")
                debug_logger.error(f"Data pipeline initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize risk manager
            logger.debug("Initializing risk manager")
            try:
                from risk_management import RiskManager, PositionManager
                self.risk_manager = RiskManager(
                    max_position_size=self.config["max_position_size"],
                    max_risk_per_trade=self.config["max_risk_per_trade"],
                    max_daily_risk=self.config["max_daily_risk"],
                    max_drawdown=self.config["max_drawdown"],
                    enable_dynamic_sizing=self.config["enable_dynamic_sizing"],
                    enable_trailing_stops=self.config["enable_trailing_stops"]
                )
            except Exception as e:
                logger.error(f"Error initializing risk manager: {str(e)}")
                debug_logger.error(f"Risk manager initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize position manager
            logger.debug("Initializing position manager")
            try:
                self.position_manager = PositionManager(self.risk_manager)
            except Exception as e:
                logger.error(f"Error initializing position manager: {str(e)}")
                debug_logger.error(f"Position manager initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize performance tracker
            logger.debug("Initializing performance tracker")
            try:
                from performance_tracking import PerformanceTracker, PerformanceDashboard
                self.performance_tracker = PerformanceTracker(
                    initial_balance=self.config["initial_balance"],
                    data_dir=self.config["performance_dir"],
                    log_trades=self.config["log_trades"],
                    log_metrics=self.config["log_metrics"],
                    log_interval=self.config["log_interval"],
                    visualization_enabled=self.config["visualization_enabled"]
                )
            except Exception as e:
                logger.error(f"Error initializing performance tracker: {str(e)}")
                debug_logger.error(f"Performance tracker initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize performance dashboard
            logger.debug("Initializing performance dashboard")
            try:
                self.performance_dashboard = PerformanceDashboard(
                    performance_tracker=self.performance_tracker,
                    dashboard_dir=self.config["dashboard_dir"]
                )
            except Exception as e:
                logger.error(f"Error initializing performance dashboard: {str(e)}")
                debug_logger.error(f"Performance dashboard initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize RL agent
            logger.debug("Initializing RL agent")
            try:
                from rl_model import PPOAgent, TradingEnvironment
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
            except Exception as e:
                logger.error(f"Error initializing RL agent: {str(e)}")
                debug_logger.error(f"RL agent initialization error: {traceback.format_exc()}")
                return False
            
            # Initialize trading environment
            logger.debug("Initializing trading environment")
            try:
                self.trading_env = TradingEnvironment(
                    data_pipeline=self.data_pipeline,
                    initial_balance=self.config["initial_balance"],
                    transaction_fee=self.config["transaction_fee"],
                    max_position=self.config["max_position_size"]
                )
            except Exception as e:
                logger.error(f"Error initializing trading environment: {str(e)}")
                debug_logger.error(f"Trading environment initialization error: {traceback.format_exc()}")
                return False
            
            # Load model if exists
            model_path = os.path.join(self.config["models_dir"], "ppo_agent.pt")
            if os.path.exists(model_path):
                try:
                    logger.debug(f"Loading model from {model_path}")
                    self.rl_agent.load_model(model_path)
                    logger.info(f"Model loaded from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    debug_logger.error(f"Model loading error: {traceback.format_exc()}")
                    # Continue even if model loading fails
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            debug_logger.error(f"Component initialization error: {traceback.format_exc()}")
            return False
    
    async def start(self):
        """
        Start the trading bot.
        """
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        # Initialize components
        if not await self.initialize():
            logger.error("Failed to initialize components")
            return
        
        # Start data pipeline
        await self.data_pipeline.start(self.config["symbols"], self.config["intervals"])
        
        # Start performance dashboard
        self.performance_dashboard.start()
        
        # Set running state
        self.is_running = True
        
        logger.info("Trading bot started")
        
        # Start main loop
        await self._main_loop()
    
    async def stop(self):
        """
        Stop the trading bot.
        """
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return
        
        # Stop data pipeline
        await self.data_pipeline.stop()
        
        # Stop performance dashboard
        self.performance_dashboard.stop()
        
        # Save model
        model_path = os.path.join(self.config["models_dir"], "ppo_agent.pt")
        self.rl_agent.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Generate final performance report
        report = self.performance_tracker.generate_performance_report()
        logger.info(f"Final performance report generated: {report['files'].get('html_report', '')}")
        
        # Set running state
        self.is_running = False
        
        logger.info("Trading bot stopped")
    
    async def _main_loop(self):
        """
        Main trading loop.
        """
        try:
            # Wait for some data to be collected
            logger.info("Collecting initial data...")
            await asyncio.sleep(60)
            
            # Main loop
            while self.is_running:
                # Get current state
                state = self.trading_env.reset()
                
                # Select action
                action, action_prob, value = self.rl_agent.select_action(state, training=False)
                
                # Take action
                next_state, reward, done, info = self.trading_env.step(action)
                
                # Update performance tracker
                self.performance_tracker.update_balance(info["portfolio_value"])
                
                # Update performance dashboard
                self.performance_dashboard.update()
                
                # Log trading activity
                logger.info(f"Action: {action}, Reward: {reward:.4f}, Portfolio: ${info['portfolio_value']:.2f}")
                
                # Wait for next step
                await asyncio.sleep(60)  # 1-minute interval
                
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            debug_logger.error(f"Main loop error: {traceback.format_exc()}")
        finally:
            # Ensure bot is stopped
            if self.is_running:
                await self.stop()
    
    async def train(self, num_episodes: int = 100, save_interval: int = 10):
        """
        Train the RL agent.
        
        Args:
            num_episodes: Number of training episodes
            save_interval: Interval for saving the model
        """
        if self.is_training:
            logger.warning("Training is already in progress")
            return
        
        # Set training state
        self.is_training = True
        
        try:
            # Initialize components if not already initialized
            if not self.data_pipeline:
                if not await self.initialize():
                    logger.error("Failed to initialize components")
                    self.is_training = False
                    return
            
            # Start data pipeline if not already started
            if not self.is_running:
                await self.data_pipeline.start(self.config["symbols"], self.config["intervals"])
                logger.info("Data pipeline started for training")
                
                # Wait for some data to be collected
                logger.info("Collecting initial data...")
                await asyncio.sleep(60)
            
            logger.info(f"Starting training for {num_episodes} episodes")
            
            # Training loop
            for episode in range(num_episodes):
                logger.info(f"Starting episode {episode+1}/{num_episodes}")
                
                # Reset environment
                state = self.trading_env.reset()
                episode_reward = 0
                
                # Episode loop
                while not self.trading_env.done:
                    # Select action
                    action, action_prob, value = self.rl_agent.select_action(state)
                    
                    # Take action
                    next_state, reward, done, info = self.trading_env.step(action)
                    
                    # Store transition
                    self.rl_agent.store_transition(state, action, action_prob, reward, value, done)
                    
                    # Update state and reward
                    state = next_state
                    episode_reward += reward
                    
                    # Wait for next step
                    await asyncio.sleep(1)
                
                # Update agent
                next_value = 0  # Terminal state value
                metrics = self.rl_agent.update(next_value)
                
                # Calculate performance metrics
                performance = self.trading_env.get_performance_metrics()
                
                logger.info(f"Episode {episode+1} finished")
                logger.info(f"Episode reward: {episode_reward:.4f}")
                logger.info(f"Training metrics: {metrics}")
                logger.info(f"Performance metrics: {performance}")
                
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
            debug_logger.error(f"Training error: {traceback.format_exc()}")
        finally:
            # Set training state
            self.is_training = False
            
            # Stop data pipeline if it was started for training
            if not self.is_running and self.data_pipeline:
                await self.data_pipeline.stop()
                logger.info("Data pipeline stopped after training")
    
    def backtest(self, start_date: str, end_date: str, data_file: str = None):
        """
        Run backtesting on historical data.
        
        Args:
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            data_file: Path to historical data file (optional)
        """
        if self.is_backtesting:
            logger.warning("Backtesting is already in progress")
            return
        
        # Set backtesting state
        self.is_backtesting = True
        
        try:
            # Initialize components if not already initialized
            if not self.rl_agent or not self.risk_manager or not self.performance_tracker:
                # Use synchronous initialization instead of async
                logger.info("Initializing components for backtesting")
                if not self.initialize_sync():
                    logger.error("Failed to initialize components")
                    self.is_backtesting = False
                    return
            
            logger.info(f"Starting backtesting from {start_date} to {end_date}")
            
            # Load historical data
            if data_file and os.path.exists(data_file):
                # Load from file
                logger.info(f"Loading historical data from file: {data_file}")
                try:
                    historical_data = pd.read_csv(data_file)
                    
                    # Convert timestamp column to datetime if it's a string
                    if 'timestamp' in historical_data.columns:
                        logger.debug(f"Timestamp column type before conversion: {historical_data['timestamp'].dtype}")
                        if historical_data['timestamp'].dtype == 'object':
                            try:
                                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                                logger.debug(f"Converted timestamp column to datetime: {historical_data['timestamp'].dtype}")
                            except Exception as e:
                                logger.error(f"Error converting timestamp to datetime: {str(e)}")
                                debug_logger.error(f"Timestamp conversion error: {traceback.format_exc()}")
                                self.is_backtesting = False
                                return None
                    
                    # Log data info
                    logger.debug(f"Historical data columns: {historical_data.columns.tolist()}")
                    logger.debug(f"Historical data shape: {historical_data.shape}")
                    logger.debug(f"Historical data types: {historical_data.dtypes}")
                    logger.debug(f"First few rows: {historical_data.head(3).to_dict('records')}")
                    
                    logger.info(f"Loaded historical data from {data_file}")
                except Exception as e:
                    logger.error(f"Error loading historical data from file: {str(e)}")
                    debug_logger.error(f"Data loading error: {traceback.format_exc()}")
                    self.is_backtesting = False
                    return None
            else:
                # Fetch from Binance
                symbol = self.config["symbols"][0]
                interval = self.config["intervals"][0]
                
                # Convert dates to timestamps
                logger.debug(f"Converting dates to timestamps: {start_date} to {end_date}")
                try:
                    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
                    logger.debug(f"Converted timestamps: {start_ts} to {end_ts}")
                except Exception as e:
                    logger.error(f"Error converting dates to timestamps: {str(e)}")
                    debug_logger.error(f"Date conversion error: {traceback.format_exc()}")
                    self.is_backtesting = False
                    return None
                
                # Fetch historical data (synchronous version)
                logger.info(f"Fetching historical data from Binance for {symbol} with interval {interval}")
                klines = []
                try:
                    # Handle potential exceptions during data fetching
                    logger.debug("Calling get_historical_klines")
                    klines = self.data_pipeline.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_ts,
                        end_str=end_ts
                    )
                    logger.debug(f"Received {len(klines)} klines from Binance")
                except Exception as e:
                    logger.error(f"Error fetching historical data: {str(e)}")
                    debug_logger.error(f"Data fetching error: {traceback.format_exc()}")
                    self.is_backtesting = False
                    return None
                
                # Check if we got any data
                if not klines or len(klines) == 0:
                    logger.error("No historical data retrieved from Binance")
                    self.is_backtesting = False
                    return None
                
                # Convert to DataFrame
                try:
                    logger.debug("Converting klines to DataFrame")
                    historical_data = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert types
                    logger.debug("Converting data types")
                    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms')
                    historical_data['open'] = historical_data['open'].astype(float)
                    historical_data['high'] = historical_data['high'].astype(float)
                    historical_data['low'] = historical_data['low'].astype(float)
                    historical_data['close'] = historical_data['close'].astype(float)
                    historical_data['volume'] = historical_data['volume'].astype(float)
                    
                    logger.info(f"Fetched {len(historical_data)} historical candles from Binance")
                except Exception as e:
                    logger.error(f"Error processing historical data: {str(e)}")
                    debug_logger.error(f"Data processing error: {traceback.format_exc()}")
                    self.is_backtesting = False
                    return None
            
            # Check if we have enough data
            if len(historical_data) < 2:
                logger.error("Not enough historical data for backtesting")
                self.is_backtesting = False
                return None
            
            # Create backtesting environment
            logger.debug("Creating backtesting environment")
            try:
                from rl_model import TradingEnvironment
                backtest_env = TradingEnvironment(
                    data_pipeline=None,  # Will use historical data directly
                    initial_balance=self.config["initial_balance"],
                    transaction_fee=self.config["transaction_fee"],
                    max_position=self.config["max_position_size"]
                )
                logger.debug("Backtesting environment created")
            except Exception as e:
                logger.error(f"Error creating backtesting environment: {str(e)}")
                debug_logger.error(f"Backtesting environment error: {traceback.format_exc()}")
                self.is_backtesting = False
                return None
            
            # Reset performance tracker
            logger.debug("Resetting performance tracker for backtesting")
            try:
                from performance_tracking import PerformanceTracker
                backtest_dir = os.path.join(self.config["performance_dir"], "backtest")
                os.makedirs(backtest_dir, exist_ok=True)
                self.performance_tracker = PerformanceTracker(
                    initial_balance=self.config["initial_balance"],
                    data_dir=backtest_dir,
                    log_trades=self.config["log_trades"],
                    log_metrics=self.config["log_metrics"],
                    visualization_enabled=self.config["visualization_enabled"]
                )
                logger.debug("Performance tracker reset")
            except Exception as e:
                logger.error(f"Error resetting performance tracker: {str(e)}")
                debug_logger.error(f"Performance tracker reset error: {traceback.format_exc()}")
                self.is_backtesting = False
                return None
            
            # Backtesting loop
            logger.info("Starting backtesting loop")
            current_balance = self.config["initial_balance"]
            position = 0.0
            entry_price = 0.0
            
            for i in range(len(historical_data) - 1):
                # Get current candle
                try:
                    current_candle = historical_data.iloc[i]
                    next_candle = historical_data.iloc[i + 1]
                    
                    # Log candle data for debugging
                    if i % 100 == 0 or i < 5:  # Log first few and then every 100th
                        logger.debug(f"Processing candle {i}: {current_candle.to_dict()}")
                except Exception as e:
                    logger.error(f"Error accessing candle data at index {i}: {str(e)}")
                    debug_logger.error(f"Candle access error: {traceback.format_exc()}")
                    continue
                
                # Create state representation
                try:
                    # In a real implementation, this would use the same feature engineering as live trading
                    state = np.zeros(self.config["state_dim"])
                    
                    # Fill in basic price features
                    state[0] = float(current_candle['open'])
                    state[1] = float(current_candle['high'])
                    state[2] = float(current_candle['low'])
                    state[3] = float(current_candle['close'])
                    state[4] = float(current_candle['volume'])
                    
                    # Add position information
                    state[-5] = current_balance / self.config["initial_balance"]
                    state[-4] = position / self.config["max_position_size"]
                    state[-3] = 1 if position > 0 else 0
                    state[-2] = 1 if position < 0 else 0
                    state[-1] = (float(current_candle['close']) / entry_price - 1) if position != 0 and entry_price != 0 else 0
                except Exception as e:
                    logger.error(f"Error creating state representation at index {i}: {str(e)}")
                    debug_logger.error(f"State creation error: {traceback.format_exc()}")
                    continue
                
                # Select action
                try:
                    action, _, _ = self.rl_agent.select_action(state, training=False)
                except Exception as e:
                    logger.error(f"Error selecting action at index {i}: {str(e)}")
                    debug_logger.error(f"Action selection error: {traceback.format_exc()}")
                    continue
                
                # Execute action
                try:
                    current_price = float(current_candle['close'])
                    next_price = float(next_candle['close'])
                    
                    # Calculate portfolio value before action
                    portfolio_value_before = current_balance + position * current_price
                    
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
                                'trade_id': f"backtest_trade_{i}_close_short",
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
                            reward = 0.0
                    
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
                                'trade_id': f"backtest_trade_{i}_close_long",
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
                            reward = 0.0
                    
                    # Calculate portfolio value after action
                    portfolio_value_after = current_balance + position * next_price
                    
                    # Update performance tracker
                    self.performance_tracker.update_balance(portfolio_value_after, current_candle['timestamp'])
                except Exception as e:
                    logger.error(f"Error executing action at index {i}: {str(e)}")
                    debug_logger.error(f"Action execution error: {traceback.format_exc()}")
                    continue
                
                # Log progress
                if i % 100 == 0:
                    logger.info(f"Backtesting progress: {i}/{len(historical_data)} ({i/len(historical_data)*100:.1f}%)")
                    logger.info(f"Current balance: ${current_balance:.2f}, Position: {position}, Portfolio: ${portfolio_value_after:.2f}")
            
            # Close any open position at the end
            logger.info("Closing any open positions at the end of backtesting")
            try:
                if position != 0:
                    final_price = float(historical_data.iloc[-1]['close'])
                    
                    if position > 0:  # Close long position
                        # Calculate realized PnL
                        realized_pnl = position * (final_price - entry_price)
                        # Apply transaction fee
                        fee = position * final_price * self.config["transaction_fee"]
                        # Update balance
                        current_balance += position * final_price - fee
                        
                        # Record trade
                        self.performance_tracker.record_trade({
                            'trade_id': "backtest_trade_final_close_long",
                            'symbol': self.config["symbols"][0],
                            'type': 'long',
                            'entry_time': historical_data.iloc[-2]['timestamp'],
                            'exit_time': historical_data.iloc[-1]['timestamp'],
                            'entry_price': entry_price,
                            'exit_price': final_price,
                            'size': position,
                            'pnl': realized_pnl,
                            'fee': fee,
                            'exit_reason': 'end_of_backtest'
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
                            'trade_id': "backtest_trade_final_close_short",
                            'symbol': self.config["symbols"][0],
                            'type': 'short',
                            'entry_time': historical_data.iloc[-2]['timestamp'],
                            'exit_time': historical_data.iloc[-1]['timestamp'],
                            'entry_price': entry_price,
                            'exit_price': final_price,
                            'size': abs(position),
                            'pnl': realized_pnl,
                            'fee': fee,
                            'exit_reason': 'end_of_backtest'
                        })
            except Exception as e:
                logger.error(f"Error closing final positions: {str(e)}")
                debug_logger.error(f"Final position closing error: {traceback.format_exc()}")
            
            # Update final balance
            logger.info("Updating final balance")
            try:
                self.performance_tracker.update_balance(current_balance, historical_data.iloc[-1]['timestamp'])
            except Exception as e:
                logger.error(f"Error updating final balance: {str(e)}")
                debug_logger.error(f"Final balance update error: {traceback.format_exc()}")
            
            # Generate performance report
            logger.info("Generating performance report")
            try:
                report_dir = os.path.join(self.config["performance_dir"], "backtest")
                os.makedirs(report_dir, exist_ok=True)
                report = self.performance_tracker.generate_performance_report(report_dir)
                
                logger.info("Backtesting completed")
                logger.info(f"Final balance: ${current_balance:.2f}")
                logger.info(f"Performance report: {report['files'].get('html_report', '')}")
                
                return report
            except Exception as e:
                logger.error(f"Error generating performance report: {str(e)}")
                debug_logger.error(f"Performance report generation error: {traceback.format_exc()}")
                return None
            
        except Exception as e:
            logger.error(f"Error during backtesting: {str(e)}")
            debug_logger.error(f"Backtesting error: {traceback.format_exc()}")
            return None
        finally:
            # Set backtesting state
            self.is_backtesting = False

def main():
    """
    Main function to run the trading bot.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autonomous Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--train', action='store_true', help='Train the RL agent')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--start-date', type=str, help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--data-file', type=str, help='Path to historical data file for backtesting')
    
    args = parser.parse_args()
    
    # Create trading bot
    bot = AutonomousTradingBot(args.config)
    
    if args.train:
        # Train the RL agent (async)
        asyncio.run(bot.train(num_episodes=args.episodes))
    elif args.backtest:
        # Run backtesting (sync)
        if not args.start_date or not args.end_date:
            logger.error("Start date and end date are required for backtesting")
            return
        
        # Call the non-async backtest method directly
        bot.backtest(args.start_date, args.end_date, args.data_file)
    else:
        # Run the trading bot (async)
        asyncio.run(bot.start())

if __name__ == "__main__":
    main()
