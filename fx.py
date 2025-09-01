import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# Initialize MT5 connection
def initialize_mt5():
    """
    Initialize connection to MetaTrader 5
    """
    if not mt5.initialize():
        print("MT5 initialization failed")
        print("Error code:", mt5.last_error())
        return False
    
    # Login to your account (replace with your credentials)
    account = 103964004  # Your account number
    password = "C-CMr@t8"
    server = "FBS-Demo"
    
    authorized = mt5.login(account, password=password, server=server)
    if not authorized:
        print("Login failed")
        print("Error code:", mt5.last_error())
        mt5.shutdown()
        return False
    
    print("Connected to MetaTrader 5")
    print("Account info:", mt5.account_info())
    return True

# Get historical data from MT5
def get_historical_data(symbol, timeframe, num_bars):
    """
    Retrieve historical data from MT5
    """
    # Set the symbol
    mt5.symbol_select(symbol, True)
    
    # Define the timeframe
    tf = get_timeframe_enum(timeframe)
    
    # Get rates
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_bars)
    
    if rates is None:
        print(f"Failed to get rates for {symbol}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

# Convert timeframe string to MT5 enum
def get_timeframe_enum(timeframe):
    """
    Convert timeframe string to MT5 timeframe constant
    """
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }
    return timeframes.get(timeframe, mt5.TIMEFRAME_H1)

# Calculate technical indicators
def calculate_indicators(df):
    """
    Calculate technical indicators for the strategy
    """
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_trend'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # MACD
    df['macd_line'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    return df

# Generate trading signals
def generate_signals(df):
    """
    Generate buy/sell signals based on strategy
    """
    # Get the latest data
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Strategy conditions
    bullish_trend = latest['close'] > latest['ema_trend']
    ema_bullish = latest['ema_fast'] > latest['ema_slow']
    macd_bullish = latest['macd_line'] > latest['macd_signal']
    rsi_oversold = latest['rsi'] < 35
    
    # Signal strength
    bullish_strength = sum([
        2 if bullish_trend else 0,
        1.5 if ema_bullish else 0,
        1 if macd_bullish else 0,
        0.5 if rsi_oversold else 0
    ])
    
    # Determine signal
    if bullish_strength >= 3:
        return 'BUY'
    elif bullish_strength <= 1:
        return 'SELL'
    else:
        return 'HOLD'

# Execute trade
def execute_trade(symbol, signal, lot_size=0.01, stop_loss_pips=20, take_profit_pips=40):
    """
    Execute a trade based on the signal
    """
    # Get current price
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        return None
    
    # Calculate stop loss and take profit
    point = symbol_info.point
    ask = mt5.symbol_info_tick(symbol).ask
    bid = mt5.symbol_info_tick(symbol).bid
    
    if signal == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        price = ask
        sl = price - stop_loss_pips * point
        tp = price + take_profit_pips * point
    elif signal == 'SELL':
        order_type = mt5.ORDER_TYPE_SELL
        price = bid
        sl = price + stop_loss_pips * point
        tp = price - take_profit_pips * point
    else:
        return None  # No trade for HOLD
    
    # Prepare order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "Python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    # Send order
    result = mt5.order_send(request)
    
    # Check result
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode={result.retcode}")
        return None
    
    print(f"Order executed: {signal} {lot_size} lots of {symbol}")
    return result

# Check open positions
def get_open_positions(symbol=None):
    """
    Get all open positions or for a specific symbol
    """
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None:
        return []
    return positions

# Close all positions
def close_all_positions():
    """
    Close all open positions
    """
    positions = get_open_positions()
    for position in positions:
        # Prepare close request
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send close order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position {position.ticket}, retcode={result.retcode}")
        else:
            print(f"Closed position {position.ticket}")

# Main trading function
def run_forex_bot(symbol="EURUSD", timeframe="H1", num_bars=100, lot_size=0.1, trade_interval_minutes=60):
    """
    Main function to run the forex trading bot
    """
    # Initialize MT5
    if not initialize_mt5():
        return
    
    print(f"Starting forex bot for {symbol} on {timeframe} timeframe")
    
    try:
        while True:
            # Get historical data
            df = get_historical_data(symbol, timeframe, num_bars)
            if df is None:
                print("Failed to get data, retrying in 5 minutes")
                time.sleep(300)
                continue
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Generate signal
            signal = generate_signals(df)
            print(f"{datetime.now()}: Signal for {symbol} - {signal}")
            
            # Check if we already have a position
            positions = get_open_positions(symbol)
            
            if signal == 'BUY' and not positions:
                # Execute buy order
                execute_trade(symbol, 'BUY', lot_size)
            elif signal == 'SELL' and not positions:
                # Execute sell order
                execute_trade(symbol, 'SELL', lot_size)
            elif positions and signal == 'HOLD':
                # Consider closing position if signal changes to HOLD
                print("Considering closing position due to HOLD signal")
                # You can add logic here to close based on other conditions
            
            # Wait for next interval
            print(f"Waiting {trade_interval_minutes} minutes for next check...")
            time.sleep(trade_interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Error in bot: {e}")
    finally:
        # Close connection
        mt5.shutdown()
        print("MT5 connection closed")

# Run the bot
if __name__ == "__main__":
    # Configuration
    symbol = "EURUSD"           # Forex pair to trade
    timeframe = "1H"            # 1-hour timeframe
    num_bars = 100              # Number of bars to retrieve
    lot_size = 0.1              # Lot size (standard lot = 1.0, mini lot = 0.1)
    trade_interval_minutes = 60 # Check every hour
    
    # Run the bot
    run_forex_bot(symbol, timeframe, num_bars, lot_size, trade_interval_minutes)