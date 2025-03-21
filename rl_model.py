"""
RL Model Module for the Autonomous Trading Bot

This module implements the Reinforcement Learning model using PPO algorithm
for autonomous trading decision making.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """
    Policy Network for the PPO algorithm.
    Maps state observations to action probabilities.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of the input (state) vector
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the output (action) vector
        """
        super(PolicyNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Policy Network initialized with architecture: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Action probabilities
        """
        return self.network(x)

class ValueNetwork(nn.Module):
    """
    Value Network for the PPO algorithm.
    Maps state observations to value estimates.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initialize the value network.
        
        Args:
            input_dim: Dimension of the input (state) vector
            hidden_dims: List of hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Value Network initialized with architecture: {input_dim} -> {hidden_dims} -> 1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Value estimate
        """
        return self.network(x)

class PPOAgent:
    """
    PPO Agent for autonomous trading.
    """
    
    def __init__(
        self,
        state_dim: int = 30,
        action_dim: int = 3,
        hidden_dims: List[int] = [256, 128, 64],  # Increased network capacity
        lr_policy: float = 1e-4,  # Reduced learning rate for more stable learning
        lr_value: float = 5e-4,   # Reduced learning rate for more stable learning
        gamma: float = 0.995,     # Increased discount factor for longer-term rewards
        gae_lambda: float = 0.97, # Increased lambda for better advantage estimation
        clip_ratio: float = 0.1,  # Reduced clip ratio for more conservative updates
        value_coef: float = 0.5,
        entropy_coef: float = 0.02, # Increased entropy coefficient for more exploration
        max_grad_norm: float = 0.3, # Reduced gradient norm for more stable updates
        device: str = None
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions for both networks
            lr_policy: Learning rate for the policy network
            lr_value: Learning rate for the value network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, hidden_dims, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dims).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        # Set hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize memory buffers
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        logger.info("PPO Agent initialized")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Tuple of (selected action, action probability, state value)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action probabilities and state value
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            state_value = self.value_net(state_tensor).item()
        
        # Sample action from probability distribution
        if training:
            action_dist = Categorical(action_probs)
            action = action_dist.sample().item()
            action_prob = action_probs[action].item()
        else:
            # During evaluation, take the most probable action
            action = torch.argmax(action_probs).item()
            action_prob = action_probs[action].item()
        
        return action, action_prob, state_value
    
    def store_transition(self, state: np.ndarray, action: int, action_prob: float, reward: float, value: float, done: bool):
        """
        Store a transition in memory.
        
        Args:
            state: State observation
            action: Selected action
            action_prob: Probability of the selected action
            reward: Received reward
            value: Estimated state value
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value estimate of the next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [False])
        
        # Initialize arrays
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, next_value: float, epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.
        
        Args:
            next_value: Value estimate of the next state
            epochs: Number of epochs to train on the collected data
            batch_size: Mini-batch size for training
            
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_action_probs = torch.FloatTensor(np.array(self.action_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'total_loss': 0
        }
        
        # PPO update
        for _ in range(epochs):
            # Generate random indices
            indices = np.random.permutation(len(states))
            
            # Mini-batch training
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_action_probs = old_action_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                batch_action_probs = self.policy_net(batch_states)
                batch_values = self.value_net(batch_states).squeeze()
                
                # Create action distribution
                action_dist = Categorical(batch_action_probs)
                
                # Get log probabilities of actions
                batch_new_action_probs = action_dist.probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                
                # Compute ratio
                ratio = batch_new_action_probs / (batch_old_action_probs + 1e-8)
                
                # Compute surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                # Compute policy loss
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute value loss
                value_loss = 0.5 * ((batch_values - batch_returns) ** 2).mean()
                
                # Compute entropy bonus
                entropy = action_dist.entropy().mean()
                
                # Compute total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['total_loss'] += total_loss.item()
                
                # Backward pass and optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                # Update networks
                self.policy_optimizer.step()
                self.value_optimizer.step()
        
        # Average metrics
        num_updates = epochs * (len(states) // batch_size + (1 if len(states) % batch_size != 0 else 0))
        for key in metrics:
            metrics[key] /= num_updates
        
        # Clear memory buffers
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return metrics
    
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'hyperparams': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm
            }
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        # Check if file exists
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            # Load model
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load hyperparameters
            hyperparams = checkpoint['hyperparams']
            self.state_dim = hyperparams['state_dim']
            self.action_dim = hyperparams['action_dim']
            self.gamma = hyperparams['gamma']
            self.gae_lambda = hyperparams['gae_lambda']
            self.clip_ratio = hyperparams['clip_ratio']
            self.value_coef = hyperparams['value_coef']
            self.entropy_coef = hyperparams['entropy_coef']
            self.max_grad_norm = hyperparams['max_grad_norm']
            
            # Reinitialize networks if dimensions don't match
            if self.policy_net.network[0].in_features != self.state_dim or self.policy_net.network[-2].out_features != self.action_dim:
                logger.info("Reinitializing networks due to dimension mismatch")
                hidden_dims = [self.policy_net.network[1].in_features, self.policy_net.network[3].in_features]
                self.policy_net = PolicyNetwork(self.state_dim, hidden_dims, self.action_dim).to(self.device)
                self.value_net = ValueNetwork(self.state_dim, hidden_dims).to(self.device)
                self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
                self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
            
            # Load state dicts
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

class TradingEnvironment:
    """
    Trading environment for the RL agent.
    """
    
    def __init__(
        self,
        data_pipeline,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.0004,
        max_position: float = 1.0,
        reward_scaling: float = 1.0,
        window_size: int = 30
    ):
        """
        Initialize the trading environment.
        
        Args:
            data_pipeline: Data pipeline for market data
            initial_balance: Initial account balance
            transaction_fee: Transaction fee as a fraction of trade value
            max_position: Maximum position size as a fraction of balance
            reward_scaling: Scaling factor for rewards
            window_size: Window size for feature calculation
        """
        self.data_pipeline = data_pipeline
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        
        # Trading state
        self.balance = initial_balance
        self.position = 0.0  # Current position size (negative for short)
        self.entry_price = 0.0  # Entry price of current position
        self.last_price = 0.0  # Last observed price
        
        # Performance tracking
        self.portfolio_values = []
        self.trades = []
        
        # Episode state
        self.current_step = 0
        self.done = False
        
        logger.info("Trading environment initialized")
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Reset trading state
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.last_price = 0.0
        
        # Reset performance tracking
        self.portfolio_values = []
        self.trades = []
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action: int):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            Tuple of (next state, reward, done, info)
        """
        # Increment step counter
        self.current_step += 1
        
        # Get current price
        current_price = self._get_current_price()
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.position * current_price
        
        # Execute action
        reward, info = self._execute_action(action, current_price)
        
        # Calculate portfolio value after action
        portfolio_value_after = self.balance + self.position * current_price
        
        # Update last price
        self.last_price = current_price
        
        # Update portfolio values
        self.portfolio_values.append(portfolio_value_after)
        
        # Check if episode is done
        self.done = self._is_done()
        
        # Get next state
        next_state = self._get_state()
        
        # Update info
        info.update({
            'portfolio_value': portfolio_value_after,
            'balance': self.balance,
            'position': self.position,
            'price': current_price,
            'step': self.current_step
        })
        
        return next_state, reward, self.done, info
    
    def _get_state(self):
        """
        Get the current state observation.
        
        Returns:
            State observation as numpy array
        """
        # Get market features from data pipeline
        market_features = self.data_pipeline.get_latest_features('BTCUSDT')
        
        if market_features is None or market_features.empty:
            # If no data available, return zeros
            return np.zeros(self.window_size)
        
        # Convert to numpy array
        market_features = market_features.values.flatten()
        
        # Add account features
        account_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position / self.max_position,  # Normalized position
            1 if self.position > 0 else 0,  # Long indicator
            1 if self.position < 0 else 0,  # Short indicator
            (self.last_price / self.entry_price - 1) if self.position != 0 else 0  # Unrealized PnL
        ])
        
        # Combine features
        state = np.concatenate([market_features, account_features])
        
        return state
    
    def _get_current_price(self):
        """
        Get the current market price.
        
        Returns:
            Current price
        """
        # Get latest features
        features = self.data_pipeline.get_latest_features('BTCUSDT')
        
        if features is None or features.empty:
            # If no data available, use last price
            return self.last_price if self.last_price > 0 else 100.0
        
        # Extract close price
        current_price = features['close'].values[0]
        
        return current_price
    
    def _execute_action(self, action: int, current_price: float):
        """
        Execute a trading action.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
            current_price: Current market price
            
        Returns:
            Tuple of (reward, info)
        """
        info = {'action': action}
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.position * current_price
        
        if action == 0:  # Hold
            # No action, just calculate unrealized PnL
            unrealized_pnl = self.position * (current_price - self.last_price)
            
            # Improved reward function with stability and risk management components
            base_reward = unrealized_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
            
            # Add time decay to discourage holding positions too long without profit
            time_factor = 1.0 / (1.0 + 0.01 * len(self.portfolio_values))
            
            # Calculate portfolio volatility for risk adjustment
            if len(self.portfolio_values) > 1:
                portfolio_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
                volatility_penalty = min(1.0, 0.5 * np.std(portfolio_returns) * 10) if len(portfolio_returns) > 0 else 0
            else:
                volatility_penalty = 0
                
            # Combine components for final reward
            reward = base_reward * (1 - volatility_penalty) * time_factor
            
            info['action_taken'] = 'hold'
            info['base_reward'] = base_reward
            info['volatility_penalty'] = volatility_penalty
            info['time_factor'] = time_factor
            
        elif action == 1:  # Buy
            if self.position < 0:  # Close short position
                # Calculate realized PnL
                realized_pnl = -self.position * (current_price - self.entry_price)
                # Apply transaction fee
                fee = abs(self.position * current_price * self.transaction_fee)
                # Update balance
                self.balance += realized_pnl - fee
                # Record trade
                self.trades.append({
                    'type': 'close_short',
                    'step': self.current_step,
                    'price': current_price,
                    'size': abs(self.position),
                    'pnl': realized_pnl,
                    'fee': fee
                })
                # Reset position
                self.position = 0.0
                self.entry_price = 0.0
                # Calculate improved reward with risk management components
                base_reward = realized_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
                
                # Add reward scaling based on position holding time
                holding_time = self.current_step - self.trades[-1]['step'] if len(self.trades) > 1 else 1
                time_bonus = min(1.5, 1.0 + (0.1 * min(holding_time, 10)))
                
                # Add reward scaling based on trade size relative to portfolio
                size_factor = 1.0 - (abs(self.position) * current_price / portfolio_value_before) * 0.5
                
                # Combine components for final reward
                reward = base_reward * time_bonus * size_factor
                
                info['action_taken'] = 'close_short'
                info['realized_pnl'] = realized_pnl
                info['fee'] = fee
                info['base_reward'] = base_reward
                info['time_bonus'] = time_bonus
                info['size_factor'] = size_factor
                
            # Open long position
            position_size = self.max_position * self.initial_balance / current_price
            # Check if we have enough balance
            if self.balance >= position_size * current_price:
                # Apply transaction fee
                fee = position_size * current_price * self.transaction_fee
                # Update balance and position
                self.balance -= position_size * current_price + fee
                self.position = position_size
                self.entry_price = current_price
                # Record trade
                self.trades.append({
                    'type': 'open_long',
                    'step': self.current_step,
                    'price': current_price,
                    'size': position_size,
                    'fee': fee
                })
                # No immediate reward for opening position
                reward = 0.0
                info['action_taken'] = 'open_long'
                info['fee'] = fee
            else:
                # Not enough balance
                reward = 0.0
                info['action_taken'] = 'insufficient_balance'
            
        elif action == 2:  # Sell
            if self.position > 0:  # Close long position
                # Calculate realized PnL
                realized_pnl = self.position * (current_price - self.entry_price)
                # Apply transaction fee
                fee = self.position * current_price * self.transaction_fee
                # Update balance
                self.balance += self.position * current_price - fee
                # Record trade
                self.trades.append({
                    'type': 'close_long',
                    'step': self.current_step,
                    'price': current_price,
                    'size': self.position,
                    'pnl': realized_pnl,
                    'fee': fee
                })
                # Reset position
                self.position = 0.0
                self.entry_price = 0.0
                # Calculate reward
                reward = realized_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
                info['action_taken'] = 'close_long'
                info['realized_pnl'] = realized_pnl
                info['fee'] = fee
                
            # Open short position
            position_size = self.max_position * self.initial_balance / current_price
            # Check if we have enough balance
            if self.balance >= position_size * current_price:
                # Apply transaction fee
                fee = position_size * current_price * self.transaction_fee
                # Update balance and position
                self.balance -= fee
                self.position = -position_size
                self.entry_price = current_price
                # Record trade
                self.trades.append({
                    'type': 'open_short',
                    'step': self.current_step,
                    'price': current_price,
                    'size': position_size,
                    'fee': fee
                })
                # No immediate reward for opening position
                reward = 0.0
                info['action_taken'] = 'open_short'
                info['fee'] = fee
            else:
                # Not enough balance
                reward = 0.0
                info['action_taken'] = 'insufficient_balance'
        
        # Calculate portfolio value after action
        portfolio_value_after = self.balance + self.position * current_price
        
        # Add portfolio change to info
        info['portfolio_change'] = portfolio_value_after - portfolio_value_before
        
        # Scale reward
        reward *= self.reward_scaling
        
        return reward, info
    
    def _is_done(self):
        """
        Check if the episode is done.
        
        Returns:
            Boolean indicating if the episode is done
        """
        # Check if balance is too low
        if self.balance <= 0:
            return True
        
        # Check if maximum steps reached
        if self.current_step >= 1000:
            return True
        
        return False
    
    def render(self):
        """
        Render the environment state.
        """
        portfolio_value = self.balance + self.position * self._get_current_price()
        position_type = "LONG" if self.position > 0 else "SHORT" if self.position < 0 else "NONE"
        
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {position_type} {abs(self.position):.6f}")
        print(f"Entry Price: ${self.entry_price:.2f}")
        print(f"Current Price: ${self._get_current_price():.2f}")
        print(f"Trades: {len(self.trades)}")
        print("-----------------------------------")
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Check if we have enough data
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'num_trades': 0
            }
        
        # Calculate total return
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        
        # Calculate daily returns
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        # Calculate Sharpe ratio (assuming daily returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Calculate maximum drawdown
        peak = self.portfolio_values[0]
        max_drawdown = 0.0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate trade metrics
        num_trades = len([t for t in self.trades if t['type'] in ['close_long', 'close_short']])
        winning_trades = len([t for t in self.trades if t['type'] in ['close_long', 'close_short'] and t.get('pnl', 0) > 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0.0
        
        # Calculate average profit per trade
        total_pnl = sum([t.get('pnl', 0) for t in self.trades if t['type'] in ['close_long', 'close_short']])
        avg_profit_per_trade = total_pnl / num_trades if num_trades > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade,
            'num_trades': num_trades
        }

# Example usage
def main():
    import os
    import asyncio
    from data_pipeline import DataPipeline
    
    async def test_rl_model():
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
        
        # Create trading environment
        env = TradingEnvironment(pipeline)
        
        # Create PPO agent
        agent = PPOAgent()
        
        # Training loop
        num_episodes = 5
        for episode in range(num_episodes):
            logger.info(f"Starting episode {episode+1}/{num_episodes}")
            
            # Reset environment
            state = env.reset()
            episode_reward = 0
            
            # Episode loop
            while not env.done:
                # Select action
                action, action_prob, value = agent.select_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, action_prob, reward, value, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Render environment
                env.render()
                
                # Wait for next step
                await asyncio.sleep(1)
            
            # Update agent
            next_value = 0  # Terminal state value
            metrics = agent.update(next_value)
            
            # Calculate performance metrics
            performance = env.get_performance_metrics()
            
            logger.info(f"Episode {episode+1} finished")
            logger.info(f"Episode reward: {episode_reward:.4f}")
            logger.info(f"Training metrics: {metrics}")
            logger.info(f"Performance metrics: {performance}")
        
        # Save model
        agent.save_model("models/ppo_agent.pt")
        
        # Stop data collection
        await pipeline.stop()
    
    # Run test
    asyncio.run(test_rl_model())

if __name__ == "__main__":
    main()
