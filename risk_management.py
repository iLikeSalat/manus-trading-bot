"""
Risk Management Module for the Autonomous Trading Bot

This module implements advanced risk management strategies including
dynamic position sizing, risk limits, and stop-loss/take-profit mechanisms.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk Manager for controlling trading risk.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # Maximum position size as fraction of account
        max_risk_per_trade: float = 0.02,  # Maximum risk per trade as fraction of account
        max_daily_risk: float = 0.05,  # Maximum daily risk as fraction of account
        max_drawdown: float = 0.15,  # Maximum allowed drawdown before reducing position sizes
        volatility_lookback: int = 20,  # Lookback period for volatility calculation
        atr_multiplier: float = 2.0,  # Multiplier for ATR-based stop loss
        enable_dynamic_sizing: bool = True,  # Enable dynamic position sizing
        enable_adaptive_stops: bool = True,  # Enable adaptive stop losses
        enable_trailing_stops: bool = True,  # Enable trailing stops
        enable_take_profit: bool = True,  # Enable take profit targets
        take_profit_risk_ratio: float = 2.0,  # Risk-reward ratio for take profit
    ):
        """
        Initialize the Risk Manager.
        
        Args:
            max_position_size: Maximum position size as fraction of account
            max_risk_per_trade: Maximum risk per trade as fraction of account
            max_daily_risk: Maximum daily risk as fraction of account
            max_drawdown: Maximum allowed drawdown before reducing position sizes
            volatility_lookback: Lookback period for volatility calculation
            atr_multiplier: Multiplier for ATR-based stop loss
            enable_dynamic_sizing: Enable dynamic position sizing
            enable_adaptive_stops: Enable adaptive stop losses
            enable_trailing_stops: Enable trailing stops
            enable_take_profit: Enable take profit targets
            take_profit_risk_ratio: Risk-reward ratio for take profit
        """
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown = max_drawdown
        self.volatility_lookback = volatility_lookback
        self.atr_multiplier = atr_multiplier
        self.enable_dynamic_sizing = enable_dynamic_sizing
        self.enable_adaptive_stops = enable_adaptive_stops
        self.enable_trailing_stops = enable_trailing_stops
        self.enable_take_profit = enable_take_profit
        self.take_profit_risk_ratio = take_profit_risk_ratio
        
        # Performance tracking
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.daily_pnl = 0.0
        self.trades_today = []
        self.all_trades = []
        self.drawdown_history = []
        
        # Risk state
        self.current_risk_level = 1.0  # Risk scaling factor (1.0 = normal)
        self.last_reset_time = datetime.now()
        
        logger.info("Risk Manager initialized")
    
    def update_account_balance(self, balance: float):
        """
        Update the account balance.
        
        Args:
            balance: Current account balance
        """
        if self.initial_balance == 0.0:
            self.initial_balance = balance
        
        self.current_balance = balance
        
        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Calculate current drawdown
        current_drawdown = 0.0
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Record drawdown
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'balance': balance,
            'peak_balance': self.peak_balance,
            'drawdown': current_drawdown
        })
        
        # Adjust risk level based on drawdown
        self._adjust_risk_level(current_drawdown)
        
        # Reset daily metrics if needed
        self._check_daily_reset()
        
        logger.info(f"Account balance updated: ${balance:.2f}, Drawdown: {current_drawdown:.2%}")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        account_balance: float,
        market_data: pd.DataFrame = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            market_data: Recent market data for volatility calculation
            
        Returns:
            Tuple of (position size, metadata)
        """
        # Update account balance
        self.update_account_balance(account_balance)
        
        # Calculate risk amount
        risk_amount = account_balance * self.max_risk_per_trade * self.current_risk_level
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero, using default stop distance")
            # Use a default stop distance of 1% if not provided
            risk_per_unit = entry_price * 0.01
        
        # Calculate position size based on risk
        position_size = risk_amount / risk_per_unit
        
        # Apply dynamic sizing if enabled
        if self.enable_dynamic_sizing and market_data is not None:
            volatility_factor = self._calculate_volatility_factor(market_data)
            position_size *= volatility_factor
            logger.info(f"Applied volatility factor: {volatility_factor:.2f}")
        
        # Limit position size
        max_allowed_position = account_balance * self.max_position_size
        position_size = min(position_size, max_allowed_position)
        
        # Check daily risk limit
        if self._check_daily_risk_limit(position_size, risk_per_unit):
            logger.warning("Daily risk limit reached, reducing position size")
            remaining_risk = (account_balance * self.max_daily_risk) - self.daily_pnl
            if remaining_risk <= 0:
                position_size = 0
            else:
                position_size = min(position_size, remaining_risk / risk_per_unit)
        
        # Calculate dollar risk
        dollar_risk = position_size * risk_per_unit
        
        # Calculate risk percentage
        risk_percentage = (dollar_risk / account_balance) * 100
        
        # Prepare metadata
        metadata = {
            'position_size': position_size,
            'dollar_risk': dollar_risk,
            'risk_percentage': risk_percentage,
            'risk_per_unit': risk_per_unit,
            'max_position_size': max_allowed_position,
            'risk_level': self.current_risk_level
        }
        
        logger.info(f"Calculated position size: {position_size:.6f} (${dollar_risk:.2f}, {risk_percentage:.2f}%)")
        
        return position_size, metadata
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_type: str,
        market_data: pd.DataFrame = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the appropriate stop loss price.
        
        Args:
            entry_price: Entry price for the position
            position_type: Type of position ('long' or 'short')
            market_data: Recent market data for ATR calculation
            
        Returns:
            Tuple of (stop loss price, metadata)
        """
        # Default stop loss (1% from entry)
        default_stop_distance = entry_price * 0.01
        stop_loss = entry_price - default_stop_distance if position_type.lower() == 'long' else entry_price + default_stop_distance
        
        # Use ATR-based stop loss if market data is provided
        if market_data is not None and len(market_data) >= self.volatility_lookback:
            # Calculate ATR
            high_low = market_data['high'] - market_data['low']
            high_close = abs(market_data['high'] - market_data['close'].shift(1))
            low_close = abs(market_data['low'] - market_data['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=self.volatility_lookback).mean().iloc[-1]
            
            # Calculate stop distance
            stop_distance = atr * self.atr_multiplier
            
            # Calculate stop loss price
            if position_type.lower() == 'long':
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
            
            logger.info(f"ATR-based stop loss: {stop_loss:.2f} (ATR: {atr:.2f})")
        
        # Prepare metadata
        metadata = {
            'stop_loss': stop_loss,
            'stop_distance': abs(entry_price - stop_loss),
            'stop_percentage': abs(entry_price - stop_loss) / entry_price * 100
        }
        
        return stop_loss, metadata
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        position_type: str
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Calculate take profit levels.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            position_type: Type of position ('long' or 'short')
            
        Returns:
            Tuple of (take profit levels, metadata)
        """
        if not self.enable_take_profit:
            return [], {'enabled': False}
        
        # Calculate risk (distance to stop)
        risk = abs(entry_price - stop_loss)
        
        # Calculate take profit levels at different risk-reward ratios
        take_profit_levels = []
        
        # Multiple take profit levels (1R, 2R, 3R)
        for ratio in [1.0, self.take_profit_risk_ratio, self.take_profit_risk_ratio * 1.5]:
            if position_type.lower() == 'long':
                tp_price = entry_price + (risk * ratio)
            else:
                tp_price = entry_price - (risk * ratio)
            
            take_profit_levels.append(tp_price)
        
        # Prepare metadata
        metadata = {
            'enabled': True,
            'risk': risk,
            'risk_reward_ratios': [1.0, self.take_profit_risk_ratio, self.take_profit_risk_ratio * 1.5]
        }
        
        logger.info(f"Take profit levels: {take_profit_levels}")
        
        return take_profit_levels, metadata
    
    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        position_type: str,
        unrealized_pnl: float
    ) -> Tuple[float, bool]:
        """
        Update trailing stop loss.
        
        Args:
            current_price: Current market price
            current_stop: Current stop loss price
            position_type: Type of position ('long' or 'short')
            unrealized_pnl: Unrealized profit/loss
            
        Returns:
            Tuple of (new stop loss price, whether stop was updated)
        """
        if not self.enable_trailing_stops:
            return current_stop, False
        
        # Only trail in profit
        if unrealized_pnl <= 0:
            return current_stop, False
        
        # Calculate new stop based on position type
        if position_type.lower() == 'long':
            # For long positions, trail below price
            trail_distance = current_price * 0.01  # 1% trail distance
            new_stop = current_price - trail_distance
            
            # Only update if new stop is higher than current stop
            if new_stop > current_stop:
                logger.info(f"Updated trailing stop: {current_stop:.2f} -> {new_stop:.2f}")
                return new_stop, True
        else:
            # For short positions, trail above price
            trail_distance = current_price * 0.01  # 1% trail distance
            new_stop = current_price + trail_distance
            
            # Only update if new stop is lower than current stop
            if new_stop < current_stop:
                logger.info(f"Updated trailing stop: {current_stop:.2f} -> {new_stop:.2f}")
                return new_stop, True
        
        return current_stop, False
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """
        Record a completed trade.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
        
        # Add to trades list
        self.all_trades.append(trade_data)
        
        # Add to today's trades
        if self._is_same_day(trade_data['timestamp'], datetime.now()):
            self.trades_today.append(trade_data)
            
            # Update daily PnL
            if 'pnl' in trade_data:
                self.daily_pnl += trade_data['pnl']
        
        logger.info(f"Trade recorded: {trade_data}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        # Calculate current drawdown
        current_drawdown = 0.0
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Calculate win rate
        total_trades = len(self.all_trades)
        winning_trades = len([t for t in self.all_trades if t.get('pnl', 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate average profit/loss
        total_pnl = sum([t.get('pnl', 0) for t in self.all_trades])
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Calculate Sharpe ratio (if we have enough trades)
        sharpe_ratio = 0.0
        if total_trades >= 30:
            returns = [t.get('pnl', 0) / t.get('risk', 1) for t in self.all_trades]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': current_drawdown,
            'max_drawdown': max([d['drawdown'] for d in self.drawdown_history]) if self.drawdown_history else 0.0,
            'daily_pnl': self.daily_pnl,
            'trades_today': len(self.trades_today),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe_ratio,
            'current_risk_level': self.current_risk_level
        }
    
    def _calculate_volatility_factor(self, market_data: pd.DataFrame) -> float:
        """
        Calculate volatility factor for dynamic position sizing.
        
        Args:
            market_data: Recent market data
            
        Returns:
            Volatility factor (1.0 = normal volatility)
        """
        # Check if we have enough data
        if len(market_data) < self.volatility_lookback:
            return 1.0
        
        # Calculate historical volatility
        returns = market_data['close'].pct_change().dropna()
        current_vol = returns.rolling(window=self.volatility_lookback).std().iloc[-1]
        
        # Calculate long-term volatility (3x lookback period)
        long_term_vol = returns.rolling(window=self.volatility_lookback * 3).std().iloc[-1]
        
        # Calculate volatility ratio
        vol_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Inverse relationship: higher volatility = smaller position
        vol_factor = 1.0 / vol_ratio if vol_ratio > 0 else 1.0
        
        # Limit the factor to a reasonable range
        vol_factor = max(0.5, min(vol_factor, 1.5))
        
        return vol_factor
    
    def _check_daily_risk_limit(self, position_size: float, risk_per_unit: float) -> bool:
        """
        Check if the daily risk limit would be exceeded.
        
        Args:
            position_size: Calculated position size
            risk_per_unit: Risk per unit
            
        Returns:
            True if daily risk limit would be exceeded, False otherwise
        """
        # Calculate potential risk for this trade
        potential_risk = position_size * risk_per_unit
        
        # Check if adding this risk would exceed daily limit
        return (self.daily_pnl + potential_risk) > (self.current_balance * self.max_daily_risk)
    
    def _adjust_risk_level(self, current_drawdown: float):
        """
        Adjust risk level based on current drawdown.
        
        Args:
            current_drawdown: Current drawdown as a fraction
        """
        # Scale risk level based on drawdown
        if current_drawdown > self.max_drawdown:
            # Reduce risk when drawdown exceeds threshold
            self.current_risk_level = max(0.25, 1.0 - (current_drawdown / self.max_drawdown))
            logger.info(f"Reduced risk level to {self.current_risk_level:.2f} due to drawdown of {current_drawdown:.2%}")
        elif current_drawdown < self.max_drawdown / 2:
            # Gradually increase risk as drawdown decreases
            self.current_risk_level = min(1.0, 1.0 - (current_drawdown / self.max_drawdown) + 0.5)
            logger.info(f"Increased risk level to {self.current_risk_level:.2f} with drawdown of {current_drawdown:.2%}")
    
    def _check_daily_reset(self):
        """
        Check if daily metrics should be reset.
        """
        now = datetime.now()
        
        # Reset daily metrics if it's a new day
        if not self._is_same_day(self.last_reset_time, now):
            logger.info("Resetting daily risk metrics")
            self.daily_pnl = 0.0
            self.trades_today = []
            self.last_reset_time = now
    
    def _is_same_day(self, dt1: datetime, dt2: datetime) -> bool:
        """
        Check if two datetimes are on the same day.
        
        Args:
            dt1: First datetime
            dt2: Second datetime
            
        Returns:
            True if same day, False otherwise
        """
        return dt1.date() == dt2.date()

class PositionManager:
    """
    Position Manager for tracking and managing trading positions.
    """
    
    def __init__(self, risk_manager: RiskManager):
        """
        Initialize the Position Manager.
        
        Args:
            risk_manager: Risk Manager instance
        """
        self.risk_manager = risk_manager
        
        # Active positions
        self.active_positions = {}
        
        logger.info("Position Manager initialized")
    
    def open_position(
        self,
        symbol: str,
        position_type: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit_levels: List[float],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Open a new trading position.
        
        Args:
            symbol: Trading symbol
            position_type: Type of position ('long' or 'short')
            entry_price: Entry price
            position_size: Position size
            stop_loss: Stop loss price
            take_profit_levels: List of take profit prices
            metadata: Additional metadata
            
        Returns:
            Position ID
        """
        # Generate position ID
        position_id = f"{symbol}_{position_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create position object
        position = {
            'id': position_id,
            'symbol': symbol,
            'type': position_type.lower(),
            'entry_price': entry_price,
            'current_price': entry_price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit_levels': take_profit_levels,
            'take_profit_hits': [False] * len(take_profit_levels),
            'entry_time': datetime.now(),
            'last_update_time': datetime.now(),
            'metadata': metadata or {},
            'status': 'open',
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0
        }
        
        # Add to active positions
        self.active_positions[position_id] = position
        
        logger.info(f"Opened position: {position_id} ({symbol} {position_type}, {position_size} @ {entry_price})")
        
        return position_id
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        update_stops: bool = True
    ) -> Dict[str, Any]:
        """
        Update a position with current market data.
        
        Args:
            position_id: Position ID
            current_price: Current market price
            update_stops: Whether to update stop loss
            
        Returns:
            Updated position object
        """
        # Check if position exists
        if position_id not in self.active_positions:
            logger.warning(f"Position not found: {position_id}")
            return None
        
        # Get position
        position = self.active_positions[position_id]
        
        # Update current price
        position['current_price'] = current_price
        
        # Calculate unrealized PnL
        if position['type'] == 'long':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
        else:
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
        
        # Check take profit levels
        for i, tp_price in enumerate(position['take_profit_levels']):
            if not position['take_profit_hits'][i]:
                if (position['type'] == 'long' and current_price >= tp_price) or \
                   (position['type'] == 'short' and current_price <= tp_price):
                    position['take_profit_hits'][i] = True
                    logger.info(f"Take profit hit: {position_id} TP{i+1} @ {tp_price}")
                    
                    # If this is TP1 or TP2, move stop loss to breakeven
                    if i < 2 and update_stops:
                        position['stop_loss'] = position['entry_price']
                        logger.info(f"Moved stop loss to breakeven: {position_id}")
        
        # Update trailing stop if enabled
        if update_stops and self.risk_manager.enable_trailing_stops:
            new_stop, updated = self.risk_manager.update_trailing_stop(
                current_price,
                position['stop_loss'],
                position['type'],
                position['unrealized_pnl']
            )
            
            if updated:
                position['stop_loss'] = new_stop
                logger.info(f"Updated trailing stop: {position_id} -> {new_stop}")
        
        # Check stop loss
        if (position['type'] == 'long' and current_price <= position['stop_loss']) or \
           (position['type'] == 'short' and current_price >= position['stop_loss']):
            # Stop loss hit
            self.close_position(position_id, current_price, 'stop_loss')
        
        # Update last update time
        position['last_update_time'] = datetime.now()
        
        return position
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str
    ) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_reason: Reason for closing ('take_profit', 'stop_loss', 'manual')
            
        Returns:
            Closed position object
        """
        # Check if position exists
        if position_id not in self.active_positions:
            logger.warning(f"Position not found: {position_id}")
            return None
        
        # Get position
        position = self.active_positions[position_id]
        
        # Calculate realized PnL
        if position['type'] == 'long':
            position['realized_pnl'] = (exit_price - position['entry_price']) * position['size']
        else:
            position['realized_pnl'] = (position['entry_price'] - exit_price) * position['size']
        
        # Update position
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = exit_reason
        position['status'] = 'closed'
        position['duration'] = (position['exit_time'] - position['entry_time']).total_seconds() / 60  # in minutes
        
        # Record trade in risk manager
        self.risk_manager.record_trade({
            'position_id': position_id,
            'symbol': position['symbol'],
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'entry_time': position['entry_time'],
            'exit_time': position['exit_time'],
            'duration': position['duration'],
            'pnl': position['realized_pnl'],
            'exit_reason': exit_reason,
            'risk': abs(position['entry_price'] - position['stop_loss']) * position['size'],
            'reward': position['realized_pnl'],
            'risk_reward_ratio': position['realized_pnl'] / (abs(position['entry_price'] - position['stop_loss']) * position['size']) if position['stop_loss'] != position['entry_price'] else 0
        })
        
        logger.info(f"Closed position: {position_id} ({exit_reason}, PnL: ${position['realized_pnl']:.2f})")
        
        # Remove from active positions
        del self.active_positions[position_id]
        
        return position
    
    def get_position(self, position_id: str) -> Dict[str, Any]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position object
        """
        return self.active_positions.get(position_id)
    
    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active positions.
        
        Returns:
            Dictionary of active positions
        """
        return self.active_positions
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all positions.
        
        Returns:
            Dictionary with position summary
        """
        # Count positions by type
        long_positions = len([p for p in self.active_positions.values() if p['type'] == 'long'])
        short_positions = len([p for p in self.active_positions.values() if p['type'] == 'short'])
        
        # Calculate total unrealized PnL
        total_unrealized_pnl = sum([p['unrealized_pnl'] for p in self.active_positions.values()])
        
        # Calculate total position value
        total_position_value = sum([p['size'] * p['current_price'] for p in self.active_positions.values()])
        
        return {
            'total_positions': len(self.active_positions),
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_position_value': total_position_value
        }

# Example usage
def main():
    # Create risk manager
    risk_manager = RiskManager()
    
    # Create position manager
    position_manager = PositionManager(risk_manager)
    
    # Simulate account balance
    account_balance = 10000.0
    risk_manager.update_account_balance(account_balance)
    
    # Simulate market data
    market_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]
    })
    
    # Calculate position size
    entry_price = 120.0
    stop_loss, sl_metadata = risk_manager.calculate_stop_loss(entry_price, 'long', market_data)
    position_size, ps_metadata = risk_manager.calculate_position_size(entry_price, stop_loss, account_balance, market_data)
    take_profit_levels, tp_metadata = risk_manager.calculate_take_profit(entry_price, stop_loss, 'long')
    
    print(f"Entry Price: ${entry_price}")
    print(f"Stop Loss: ${stop_loss} ({sl_metadata['stop_percentage']:.2f}%)")
    print(f"Position Size: {position_size:.6f} (${position_size * entry_price:.2f})")
    print(f"Take Profit Levels: {[f'${tp:.2f}' for tp in take_profit_levels]}")
    
    # Open position
    position_id = position_manager.open_position(
        'BTCUSDT',
        'long',
        entry_price,
        position_size,
        stop_loss,
        take_profit_levels,
        {'risk_data': ps_metadata}
    )
    
    # Simulate price movement
    for i in range(10):
        # Simulate price increase
        current_price = entry_price + (i + 1) * 2
        
        # Update position
        position = position_manager.update_position(position_id, current_price)
        
        if position and position['status'] == 'open':
            print(f"Price: ${current_price}, Unrealized PnL: ${position['unrealized_pnl']:.2f}, Stop Loss: ${position['stop_loss']}")
        else:
            print(f"Position closed at ${current_price}")
            break
    
    # Get risk metrics
    risk_metrics = risk_manager.get_risk_metrics()
    print("\nRisk Metrics:")
    for key, value in risk_metrics.items():
        print(f"{key}: {value}")
    
    # Get position summary
    position_summary = position_manager.get_position_summary()
    print("\nPosition Summary:")
    for key, value in position_summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
