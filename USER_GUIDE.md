# Autonomous Trading Bot - User Guide

## Overview

This autonomous trading bot is a fully self-contained trading system that uses Reinforcement Learning (RL) with Proximal Policy Optimization (PPO) to make trading decisions on the Binance Futures market. The system is designed to operate without relying on external signals, using market data to identify trading opportunities and manage risk.

## Features

- **Autonomous Decision Making**: Uses RL to make trading decisions based on market data
- **Binance Futures Integration**: Connects directly to Binance Futures API
- **Advanced Risk Management**: Implements dynamic position sizing and risk controls
- **Performance Tracking**: Provides detailed metrics and visualizations
- **Backtesting Capabilities**: Test strategies on historical data
- **Training Mode**: Train the RL model to improve performance

## System Requirements

- Python 3.8 or higher
- Binance Futures account with API access
- Minimum 8GB RAM recommended
- Internet connection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/manus-trading-bot.git
cd manus-trading-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for API access:
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

## Configuration

The bot can be configured using a JSON configuration file. Create a file named `config.json` with the following structure:

```json
{
  "binance_api_key": "your_api_key",
  "binance_api_secret": "your_api_secret",
  "testnet": true,
  "symbols": ["BTCUSDT"],
  "intervals": ["1m"],
  "initial_balance": 10000.0,
  "max_position_size": 0.1,
  "transaction_fee": 0.0004,
  "max_risk_per_trade": 0.02,
  "max_daily_risk": 0.05,
  "enable_dynamic_sizing": true,
  "enable_trailing_stops": true
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| binance_api_key | Binance API key | Environment variable |
| binance_api_secret | Binance API secret | Environment variable |
| testnet | Use Binance testnet | true |
| symbols | Trading symbols | ["BTCUSDT"] |
| intervals | Timeframes for data collection | ["1m"] |
| initial_balance | Initial account balance | 10000.0 |
| max_position_size | Maximum position size as fraction of balance | 0.1 |
| transaction_fee | Transaction fee as fraction | 0.0004 |
| max_risk_per_trade | Maximum risk per trade as fraction of balance | 0.02 |
| max_daily_risk | Maximum daily risk as fraction of balance | 0.05 |
| enable_dynamic_sizing | Enable dynamic position sizing | true |
| enable_trailing_stops | Enable trailing stop losses | true |

## Usage

### Running the Bot

To start the bot in live trading mode:

```bash
python main_autonomous.py --config config.json
```

### Training the RL Model

To train the RL model:

```bash
python main_autonomous.py --config config.json --train --episodes 100
```

### Backtesting

To run backtesting on historical data:

```bash
python main_autonomous.py --config config.json --backtest --start-date 2023-01-01 --end-date 2023-12-31
```

You can also provide a CSV file with historical data:

```bash
python main_autonomous.py --config config.json --backtest --start-date 2023-01-01 --end-date 2023-12-31 --data-file historical_data.csv
```

## System Architecture

The trading bot consists of several key components:

1. **Data Pipeline**: Collects and processes market data from Binance
2. **RL Model**: Makes trading decisions using PPO algorithm
3. **Risk Management**: Controls position sizing and risk exposure
4. **Position Manager**: Tracks and manages trading positions
5. **Performance Tracker**: Monitors and evaluates trading performance
6. **Main Controller**: Integrates all components and provides the main entry point

## Performance Monitoring

The bot generates detailed performance reports and visualizations, including:

- Equity curve
- Drawdown chart
- Trade distribution
- Monthly returns
- Performance metrics (win rate, Sharpe ratio, etc.)

Performance data is stored in the `performance_data` directory, and a dashboard is available in the `dashboard` directory.

## Risk Management

The bot implements several risk management features:

- **Dynamic Position Sizing**: Adjusts position size based on volatility
- **Risk Limits**: Enforces maximum risk per trade and daily risk limits
- **Drawdown Protection**: Reduces position sizes during drawdowns
- **Trailing Stops**: Automatically moves stop losses to protect profits

## Customization

### Adding New Features

To add new features to the state representation:

1. Modify the `_calculate_ohlcv_features`, `_calculate_orderbook_features`, or `_calculate_trade_features` methods in `data_pipeline.py`
2. Update the `state_dim` parameter in the configuration file

### Modifying the RL Model

To modify the RL model architecture:

1. Edit the `PolicyNetwork` and `ValueNetwork` classes in `rl_model.py`
2. Update the `hidden_dims` parameter in the configuration file

### Changing Risk Parameters

To adjust risk management parameters:

1. Modify the relevant parameters in the configuration file
2. For more advanced changes, edit the `RiskManager` class in `risk_management.py`

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Verify your API keys and network connection
2. **Insufficient Balance**: Ensure your account has sufficient funds
3. **Model Performance Issues**: Try retraining the model with more data

### Logs

Logs are stored in the `trading_bot.log` file. Check this file for detailed information about errors and system operation.

## Security Considerations

- API keys are stored locally and never shared
- Use API keys with trading permissions only (no withdrawal permissions)
- Enable two-factor authentication on your Binance account
- Run the bot on a secure, dedicated machine

## Disclaimer

Trading cryptocurrencies involves significant risk. This bot is provided for educational and experimental purposes only. Use at your own risk and never trade with funds you cannot afford to lose.

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

---

Happy trading!
