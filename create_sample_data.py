import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create synthetic data for backtesting
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
days = (end_date - start_date).days + 1

print(f"Generating data for {days} days from {start_date} to {end_date}")

# Generate timestamps
timestamps = [start_date + timedelta(days=i) for i in range(days)]

# Generate price data (simple random walk)
close_price = 20000  # Starting BTC price
prices = []
for i in range(days):
    # Random daily change between -5% and +5%
    change = np.random.uniform(-0.05, 0.05)
    close_price *= (1 + change)
    
    # Generate OHLCV data
    open_price = close_price * (1 + np.random.uniform(-0.01, 0.01))
    high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.02))
    low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.02))
    volume = np.random.uniform(500, 5000)
    
    prices.append([timestamps[i], open_price, high_price, low_price, close_price, volume])

# Create DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(prices, columns=columns)

# Add additional columns needed by the backtester
df['close_time'] = df['timestamp'] + timedelta(days=1)
df['quote_asset_volume'] = df['volume'] * df['close']
df['number_of_trades'] = np.random.randint(1000, 10000, size=len(df))
df['taker_buy_base_asset_volume'] = df['volume'] * np.random.uniform(0.4, 0.6, size=len(df))
df['taker_buy_quote_asset_volume'] = df['taker_buy_base_asset_volume'] * df['close']
df['ignore'] = 0

# Important: Store timestamps as strings in ISO format that pandas can parse back to datetime
print("Converting timestamps to string format for CSV storage")
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') if hasattr(df['timestamp'], 'dt') else [d.strftime('%Y-%m-%d %H:%M:%S') for d in df['timestamp']]
df['close_time'] = df['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S') if hasattr(df['close_time'], 'dt') else [d.strftime('%Y-%m-%d %H:%M:%S') for d in df['close_time']]

# Save to CSV
df.to_csv('btc_historical_data_2023.csv', index=False)
print("Sample data file created: btc_historical_data_2023.csv")

# Create a small test file to verify the data format
test_df = pd.read_csv('btc_historical_data_2023.csv')
print("\nVerifying data format:")
print(f"Columns: {test_df.columns.tolist()}")
print(f"Data types: {test_df.dtypes}")
print(f"First row: {test_df.iloc[0].to_dict()}")

# Convert timestamp back to datetime to verify it works
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
test_df['close_time'] = pd.to_datetime(test_df['close_time'])
print("\nAfter conversion back to datetime:")
print(f"Timestamp type: {test_df['timestamp'].dtype}")
print(f"First timestamp: {test_df['timestamp'].iloc[0]}")

# Test datetime operations
print("\nTesting datetime operations:")
test_time = test_df['timestamp'].iloc[0]
test_time_minus_10 = test_time - pd.Timedelta(minutes=10)
print(f"Original time: {test_time}")
print(f"Time minus 10 minutes: {test_time_minus_10}")
print("Datetime operations working correctly!")
