"""
Performance Tracking Module for the Autonomous Trading Bot

This module implements comprehensive performance tracking, metrics calculation,
logging, and visualization for evaluating trading performance.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Performance Tracker for monitoring and evaluating trading performance.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        data_dir: str = 'performance_data',
        log_trades: bool = True,
        log_metrics: bool = True,
        log_interval: int = 60,  # seconds
        visualization_enabled: bool = True
    ):
        """
        Initialize the Performance Tracker.
        
        Args:
            initial_balance: Initial account balance
            data_dir: Directory to store performance data
            log_trades: Whether to log individual trades
            log_metrics: Whether to log performance metrics
            log_interval: Interval for logging metrics (in seconds)
            visualization_enabled: Whether to enable visualization
        """
        self.initial_balance = initial_balance
        self.data_dir = data_dir
        self.log_trades = log_trades
        self.log_metrics = log_metrics
        self.log_interval = log_interval
        self.visualization_enabled = visualization_enabled
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Performance data
        self.balance_history = []
        self.trades = []
        self.metrics_history = []
        
        # Tracking state
        self.last_log_time = datetime.now()
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        logger.info("Performance Tracker initialized")
        
        # Initialize data files
        self._initialize_data_files()
    
    def _initialize_data_files(self):
        """
        Initialize data files for storing performance data.
        """
        # Balance history file
        balance_file = os.path.join(self.data_dir, 'balance_history.csv')
        if not os.path.exists(balance_file):
            pd.DataFrame(columns=['timestamp', 'balance']).to_csv(balance_file, index=False)
        
        # Trades file
        trades_file = os.path.join(self.data_dir, 'trades.csv')
        if not os.path.exists(trades_file):
            pd.DataFrame(columns=[
                'trade_id', 'symbol', 'type', 'entry_time', 'exit_time', 'entry_price',
                'exit_price', 'size', 'pnl', 'pnl_percent', 'duration', 'exit_reason'
            ]).to_csv(trades_file, index=False)
        
        # Metrics file
        metrics_file = os.path.join(self.data_dir, 'metrics.csv')
        if not os.path.exists(metrics_file):
            pd.DataFrame(columns=[
                'timestamp', 'balance', 'total_pnl', 'total_pnl_percent', 'win_rate',
                'profit_factor', 'sharpe_ratio', 'max_drawdown', 'avg_trade_pnl',
                'avg_win', 'avg_loss', 'num_trades', 'num_wins', 'num_losses'
            ]).to_csv(metrics_file, index=False)
    
    def update_balance(self, balance: float, timestamp: datetime = None):
        """
        Update the current account balance.
        
        Args:
            balance: Current account balance
            timestamp: Timestamp of the balance update (default: current time)
        """
        timestamp = timestamp or datetime.now()
        
        # Update current balance
        self.current_balance = balance
        
        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Add to balance history
        self.balance_history.append({
            'timestamp': timestamp,
            'balance': balance
        })
        
        # Log balance update
        if self.log_metrics and (timestamp - self.last_log_time).total_seconds() >= self.log_interval:
            self._log_metrics()
            self.last_log_time = timestamp
        
        # Append to balance history file
        if len(self.balance_history) % 10 == 0:  # Only write every 10 updates to reduce I/O
            balance_file = os.path.join(self.data_dir, 'balance_history.csv')
            pd.DataFrame([self.balance_history[-1]]).to_csv(balance_file, mode='a', header=False, index=False)
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """
        Record a completed trade.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        # Add timestamp if not present
        if 'exit_time' not in trade_data:
            trade_data['exit_time'] = datetime.now()
        
        # Calculate PnL percentage
        if 'pnl_percent' not in trade_data and 'entry_price' in trade_data and 'size' in trade_data:
            trade_value = trade_data['entry_price'] * trade_data['size']
            if trade_value > 0:
                trade_data['pnl_percent'] = (trade_data.get('pnl', 0) / trade_value) * 100
        
        # Add to trades list
        self.trades.append(trade_data)
        
        # Log trade
        if self.log_trades:
            logger.info(f"Trade recorded: {trade_data}")
        
        # Append to trades file
        trades_file = os.path.join(self.data_dir, 'trades.csv')
        
        # Convert trade data to DataFrame format
        trade_row = {
            'trade_id': trade_data.get('trade_id', f"trade_{len(self.trades)}"),
            'symbol': trade_data.get('symbol', ''),
            'type': trade_data.get('type', ''),
            'entry_time': trade_data.get('entry_time', ''),
            'exit_time': trade_data.get('exit_time', ''),
            'entry_price': trade_data.get('entry_price', 0),
            'exit_price': trade_data.get('exit_price', 0),
            'size': trade_data.get('size', 0),
            'pnl': trade_data.get('pnl', 0),
            'pnl_percent': trade_data.get('pnl_percent', 0),
            'duration': trade_data.get('duration', 0),
            'exit_reason': trade_data.get('exit_reason', '')
        }
        
        pd.DataFrame([trade_row]).to_csv(trades_file, mode='a', header=False, index=False)
        
        # Update metrics after recording trade
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics based on trade history.
        """
        # Check if we have trades
        if not self.trades:
            return
        
        # Current timestamp
        timestamp = datetime.now()
        
        # Extract PnL values
        pnls = [t.get('pnl', 0) for t in self.trades]
        
        # Calculate total PnL
        total_pnl = sum(pnls)
        total_pnl_percent = (self.current_balance / self.initial_balance - 1) * 100
        
        # Calculate win/loss metrics
        num_trades = len(pnls)
        num_wins = sum(1 for pnl in pnls if pnl > 0)
        num_losses = sum(1 for pnl in pnls if pnl <= 0)
        win_rate = num_wins / num_trades if num_trades > 0 else 0
        
        # Calculate average trade metrics
        avg_trade_pnl = total_pnl / num_trades if num_trades > 0 else 0
        avg_win = sum(pnl for pnl in pnls if pnl > 0) / num_wins if num_wins > 0 else 0
        avg_loss = sum(pnl for pnl in pnls if pnl <= 0) / num_losses if num_losses > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (assuming daily returns)
        if len(self.balance_history) >= 2:
            balances = [b['balance'] for b in self.balance_history]
            returns = np.diff(balances) / balances[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = self.initial_balance
        for balance_point in self.balance_history:
            balance = balance_point['balance']
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Create metrics dictionary
        metrics = {
            'timestamp': timestamp,
            'balance': self.current_balance,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': num_trades,
            'num_wins': num_wins,
            'num_losses': num_losses
        }
        
        # Add to metrics history
        self.metrics_history.append(metrics)
        
        # Append to metrics file
        metrics_file = os.path.join(self.data_dir, 'metrics.csv')
        pd.DataFrame([metrics]).to_csv(metrics_file, mode='a', header=False, index=False)
    
    def _log_metrics(self):
        """
        Log current performance metrics.
        """
        # Calculate metrics if we don't have any
        if not self.metrics_history:
            self._calculate_metrics()
            if not self.metrics_history:
                return
        
        # Get latest metrics
        metrics = self.metrics_history[-1]
        
        # Log metrics
        logger.info(f"Performance Metrics:")
        logger.info(f"Balance: ${metrics['balance']:.2f} (Initial: ${self.initial_balance:.2f})")
        logger.info(f"Total PnL: ${metrics['total_pnl']:.2f} ({metrics['total_pnl_percent']:.2f}%)")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%} ({metrics['num_wins']}/{metrics['num_trades']})")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Avg Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
        logger.info(f"Avg Win: ${metrics['avg_win']:.2f}, Avg Loss: ${metrics['avg_loss']:.2f}")
    
    def generate_performance_report(self, output_dir: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_dir: Directory to save report files (default: data_dir)
            
        Returns:
            Dictionary containing performance metrics and file paths
        """
        output_dir = output_dir or self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics if needed
        if not self.metrics_history:
            self._calculate_metrics()
            if not self.metrics_history:
                return {'error': 'No performance data available'}
        
        # Get latest metrics
        metrics = self.metrics_history[-1]
        
        # Create report dictionary
        report = {
            'generated_at': datetime.now().isoformat(),
            'metrics': metrics,
            'files': {}
        }
        
        # Generate visualizations if enabled
        if self.visualization_enabled:
            # Equity curve
            equity_curve_path = os.path.join(output_dir, 'equity_curve.png')
            self._plot_equity_curve(equity_curve_path)
            report['files']['equity_curve'] = equity_curve_path
            
            # Drawdown chart
            drawdown_path = os.path.join(output_dir, 'drawdown.png')
            self._plot_drawdown(drawdown_path)
            report['files']['drawdown'] = drawdown_path
            
            # Trade distribution
            trade_dist_path = os.path.join(output_dir, 'trade_distribution.png')
            self._plot_trade_distribution(trade_dist_path)
            report['files']['trade_distribution'] = trade_dist_path
            
            # Monthly returns
            monthly_returns_path = os.path.join(output_dir, 'monthly_returns.png')
            self._plot_monthly_returns(monthly_returns_path)
            report['files']['monthly_returns'] = monthly_returns_path
        
        # Generate HTML report
        html_report_path = os.path.join(output_dir, 'performance_report.html')
        self._generate_html_report(html_report_path, report)
        report['files']['html_report'] = html_report_path
        
        # Generate JSON report
        json_report_path = os.path.join(output_dir, 'performance_report.json')
        with open(json_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        report['files']['json_report'] = json_report_path
        
        logger.info(f"Performance report generated: {html_report_path}")
        
        return report
    
    def _plot_equity_curve(self, output_path: str):
        """
        Plot equity curve.
        
        Args:
            output_path: Path to save the plot
        """
        if not self.balance_history:
            return
        
        # Create DataFrame from balance history
        df = pd.DataFrame(self.balance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['balance'], linewidth=2)
        plt.axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.5)
        plt.fill_between(df.index, df['balance'], self.initial_balance, where=(df['balance'] >= self.initial_balance), color='green', alpha=0.3)
        plt.fill_between(df.index, df['balance'], self.initial_balance, where=(df['balance'] < self.initial_balance), color='red', alpha=0.3)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Account Balance ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_drawdown(self, output_path: str):
        """
        Plot drawdown chart.
        
        Args:
            output_path: Path to save the plot
        """
        if not self.balance_history:
            return
        
        # Create DataFrame from balance history
        df = pd.DataFrame(self.balance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate running maximum
        df['peak'] = df['balance'].cummax()
        
        # Calculate drawdown
        df['drawdown'] = (df['peak'] - df['balance']) / df['peak'] * 100
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['drawdown'], linewidth=2, color='red')
        plt.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        plt.title('Drawdown (%)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_trade_distribution(self, output_path: str):
        """
        Plot trade PnL distribution.
        
        Args:
            output_path: Path to save the plot
        """
        if not self.trades:
            return
        
        # Extract PnL values
        pnls = [t.get('pnl', 0) for t in self.trades]
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(pnls, kde=True, bins=30)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Trade PnL Distribution')
        plt.xlabel('PnL ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_monthly_returns(self, output_path: str):
        """
        Plot monthly returns.
        
        Args:
            output_path: Path to save the plot
        """
        if not self.balance_history:
            return
        
        # Create DataFrame from balance history
        df = pd.DataFrame(self.balance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample to monthly returns
        df.set_index('timestamp', inplace=True)
        monthly = df.resample('M').last()
        monthly['return'] = monthly['balance'].pct_change() * 100
        
        # Fill first month
        if len(monthly) > 0:
            monthly['return'].iloc[0] = (monthly['balance'].iloc[0] / self.initial_balance - 1) * 100
        
        # Plot monthly returns
        plt.figure(figsize=(12, 6))
        bars = plt.bar(monthly.index, monthly['return'], width=20)
        
        # Color bars based on return
        for i, bar in enumerate(bars):
            if monthly['return'].iloc[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.title('Monthly Returns (%)')
        plt.xlabel('Month')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _generate_html_report(self, output_path: str, report: Dict[str, Any]):
        """
        Generate HTML performance report.
        
        Args:
            output_path: Path to save the HTML report
            report: Report data dictionary
        """
        metrics = report['metrics']
        files = report.get('files', {})
        
        # Convert image paths to relative paths
        image_paths = {}
        for key, path in files.items():
            if path.endswith('.png'):
                image_paths[key] = os.path.basename(path)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 30px; }}
                .metric-card {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin-top: 10px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .charts {{ display: flex; flex-wrap: wrap; }}
                .chart {{ margin: 10px; flex: 1; min-width: 500px; }}
                .chart img {{ max-width: 100%; height: auto; }}
                .footer {{ margin-top: 30px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trading Performance Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Account Balance</h3>
                        <div class="metric-value">${metrics['balance']:.2f}</div>
                        <p>Initial: ${self.initial_balance:.2f}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Total PnL</h3>
                        <div class="metric-value {'positive' if metrics['total_pnl'] >= 0 else 'negative'}">${metrics['total_pnl']:.2f} ({metrics['total_pnl_percent']:.2f}%)</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Win Rate</h3>
                        <div class="metric-value">{metrics['win_rate']:.2%}</div>
                        <p>{metrics['num_wins']} wins, {metrics['num_losses']} losses</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Profit Factor</h3>
                        <div class="metric-value">{metrics['profit_factor']:.2f}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Sharpe Ratio</h3>
                        <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Max Drawdown</h3>
                        <div class="metric-value {'negative'}">{metrics['max_drawdown']:.2%}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Avg Trade</h3>
                        <div class="metric-value {'positive' if metrics['avg_trade_pnl'] >= 0 else 'negative'}">${metrics['avg_trade_pnl']:.2f}</div>
                        <p>Win: ${metrics['avg_win']:.2f}, Loss: ${metrics['avg_loss']:.2f}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Total Trades</h3>
                        <div class="metric-value">{metrics['num_trades']}</div>
                    </div>
                </div>
                
                <div class="charts">
                    {'<div class="chart"><h3>Equity Curve</h3><img src="' + image_paths.get('equity_curve', '') + '" alt="Equity Curve"></div>' if 'equity_curve' in image_paths else ''}
                    {'<div class="chart"><h3>Drawdown</h3><img src="' + image_paths.get('drawdown', '') + '" alt="Drawdown"></div>' if 'drawdown' in image_paths else ''}
                    {'<div class="chart"><h3>Trade Distribution</h3><img src="' + image_paths.get('trade_distribution', '') + '" alt="Trade Distribution"></div>' if 'trade_distribution' in image_paths else ''}
                    {'<div class="chart"><h3>Monthly Returns</h3><img src="' + image_paths.get('monthly_returns', '') + '" alt="Monthly Returns"></div>' if 'monthly_returns' in image_paths else ''}
                </div>
                
                <div class="footer">
                    <p>Generated by Autonomous Trading Bot Performance Tracker</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)

class PerformanceDashboard:
    """
    Performance Dashboard for real-time monitoring of trading performance.
    """
    
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        update_interval: int = 60,  # seconds
        dashboard_dir: str = 'dashboard'
    ):
        """
        Initialize the Performance Dashboard.
        
        Args:
            performance_tracker: Performance Tracker instance
            update_interval: Dashboard update interval (in seconds)
            dashboard_dir: Directory to store dashboard files
        """
        self.performance_tracker = performance_tracker
        self.update_interval = update_interval
        self.dashboard_dir = dashboard_dir
        
        # Create dashboard directory
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Dashboard state
        self.is_running = False
        self.last_update_time = datetime.now()
        
        logger.info("Performance Dashboard initialized")
    
    def start(self):
        """
        Start the dashboard.
        """
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        logger.info("Performance Dashboard started")
        
        # Generate initial dashboard
        self.update()
    
    def stop(self):
        """
        Stop the dashboard.
        """
        if not self.is_running:
            logger.warning("Dashboard is not running")
            return
        
        self.is_running = False
        logger.info("Performance Dashboard stopped")
    
    def update(self):
        """
        Update the dashboard.
        """
        if not self.is_running:
            return
        
        current_time = datetime.now()
        
        # Check if it's time to update
        if (current_time - self.last_update_time).total_seconds() < self.update_interval:
            return
        
        # Generate performance report
        report = self.performance_tracker.generate_performance_report(self.dashboard_dir)
        
        # Update dashboard index
        self._update_dashboard_index(report)
        
        # Update last update time
        self.last_update_time = current_time
        
        logger.info("Dashboard updated")
    
    def _update_dashboard_index(self, report: Dict[str, Any]):
        """
        Update the dashboard index file.
        
        Args:
            report: Performance report data
        """
        # Create index.html that redirects to the performance report
        index_path = os.path.join(self.dashboard_dir, 'index.html')
        
        # Get performance report path
        report_path = report['files'].get('html_report', '')
        if report_path:
            report_filename = os.path.basename(report_path)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Performance Dashboard</title>
                <meta http-equiv="refresh" content="0; url={report_filename}" />
            </head>
            <body>
                <p>Redirecting to <a href="{report_filename}">performance report</a>...</p>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(index_path, 'w') as f:
                f.write(html_content)

# Example usage
def main():
    # Create performance tracker
    tracker = PerformanceTracker(initial_balance=10000.0)
    
    # Create performance dashboard
    dashboard = PerformanceDashboard(tracker)
    
    # Start dashboard
    dashboard.start()
    
    # Simulate trading activity
    initial_balance = 10000.0
    current_balance = initial_balance
    
    # Simulate balance updates
    for i in range(100):
        # Simulate daily balance changes
        day = i + 1
        daily_return = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
        current_balance *= (1 + daily_return)
        
        # Update balance
        tracker.update_balance(current_balance, datetime.now() - timedelta(days=100-day))
        
        # Simulate trades
        if i % 5 == 0:  # Every 5 days
            # Simulate a trade
            is_win = np.random.random() > 0.4  # 60% win rate
            trade_pnl = np.random.normal(200, 50) if is_win else np.random.normal(-150, 30)
            
            # Record trade
            tracker.record_trade({
                'trade_id': f"trade_{i//5+1}",
                'symbol': 'BTCUSDT',
                'type': 'long' if np.random.random() > 0.5 else 'short',
                'entry_time': datetime.now() - timedelta(days=100-day, hours=4),
                'exit_time': datetime.now() - timedelta(days=100-day),
                'entry_price': 50000 + np.random.normal(0, 1000),
                'exit_price': 50000 + np.random.normal(0, 1000),
                'size': np.random.uniform(0.1, 0.5),
                'pnl': trade_pnl,
                'duration': np.random.uniform(60, 240),  # 1-4 hours
                'exit_reason': np.random.choice(['take_profit', 'stop_loss', 'manual'])
            })
    
    # Update dashboard
    dashboard.update()
    
    # Generate performance report
    report = tracker.generate_performance_report()
    
    print(f"Performance report generated:")
    for key, path in report['files'].items():
        print(f"- {key}: {path}")
    
    # Print metrics
    metrics = report['metrics']
    print("\nPerformance Metrics:")
    print(f"Balance: ${metrics['balance']:.2f} (Initial: ${initial_balance:.2f})")
    print(f"Total PnL: ${metrics['total_pnl']:.2f} ({metrics['total_pnl_percent']:.2f}%)")
    print(f"Win Rate: {metrics['win_rate']:.2%} ({metrics['num_wins']}/{metrics['num_trades']})")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Avg Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
    print(f"Avg Win: ${metrics['avg_win']:.2f}, Avg Loss: ${metrics['avg_loss']:.2f}")

if __name__ == "__main__":
    main()
