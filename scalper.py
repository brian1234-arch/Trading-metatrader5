import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

class UltraFastScalper:
    def __init__(self, symbol, timeframe=mt5.TIMEFRAME_M1, magic_number=12345, risk_per_trade=0.01):
        self.symbol = symbol
        self.timeframe = timeframe
        self.magic_number = magic_number
        self.risk_per_trade = risk_per_trade
        self.initialized = False
        
        # Strategy parameters
        self.rsi_period = 7
        self.stoch_k = 14
        self.stoch_d = 3
        self.stoch_slow = 3
        
    def initialize(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        
        # Check if symbol is available
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found")
            return False
            
        # If symbol is not enabled, enable it
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                print(f"Failed to select {self.symbol}")
                return False
                
        self.initialized = True
        print(f"MT5 initialized successfully for {self.symbol}")
        return True
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            print("MT5 connection closed")
    
    def get_rates(self, num_bars=100):
        """Get historical data from MT5"""
        if not self.initialized:
            print("MT5 not initialized")
            return None
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, num_bars)
        if rates is None:
            print("Failed to get rates")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    def calculate_heikin_ashi(self, df):
        """Calculate Heikin-Ashi candles"""
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)
        ha_open = pd.Series(ha_open, index=df.index)
        
        ha_high = pd.concat([ha_open, ha_close, df['high']], axis=1).max(axis=1)
        ha_low = pd.concat([ha_open, ha_close, df['low']], axis=1).min(axis=1)
        
        return ha_open, ha_high, ha_low, ha_close
    
    def calculate_rsi(self, df, period=14):
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, df, k_period=14, d_period=3, slow=3):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def heikin_ashi_signal(self, df):
        """Generate signal based on Heikin-Ashi Pullback strategy"""
        ha_open, ha_high, ha_low, ha_close = self.calculate_heikin_ashi(df)
        
        # Get the last few candles
        current_color = ha_close.iloc[-1] > ha_open.iloc[-1]
        prev_color = ha_close.iloc[-2] > ha_open.iloc[-2]
        prev_prev_color = ha_close.iloc[-3] > ha_open.iloc[-3] if len(df) > 2 else None
        
        signal = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Check for bearish trend with pullback (buy signal)
        if not prev_prev_color and not prev_color and current_color:
            signal = 1  # Buy signal: red candles followed by green pullback
        
        # Check for bullish trend with pullback (sell signal)
        elif prev_prev_color and prev_color and not current_color:
            signal = -1  # Sell signal: green candles followed by red pullback
            
        return signal
    
    def rsi_extremes_signal(self, df):
        """Generate signal based on RSI Extremes strategy"""
        rsi = self.calculate_rsi(df, self.rsi_period)
        
        # Get the last few RSI values
        rsi_current = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]
        
        signal = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Buy signal: RSI crosses above 20 from below
        if rsi_prev <= 20 and rsi_current > 20:
            signal = 1
            
        # Sell signal: RSI crosses below 80 from above
        elif rsi_prev >= 80 and rsi_current < 80:
            signal = -1
            
        return signal
    
    def stochastic_signal(self, df):
        """Generate signal based on Stochastic Oscillator Quick Signal strategy"""
        stoch_k, stoch_d = self.calculate_stochastic(df, self.stoch_k, self.stoch_d, self.stoch_slow)
        
        # Get the last few values
        k_current = stoch_k.iloc[-1]
        d_current = stoch_d.iloc[-1]
        k_prev = stoch_k.iloc[-2]
        d_prev = stoch_d.iloc[-2]
        
        signal = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Buy signal: %K crosses above %D and both are below 20
        if k_prev <= d_prev and k_current > d_current and k_current < 20 and d_current < 20:
            signal = 1
            
        # Sell signal: %K crosses below %D and both are above 80
        elif k_prev >= d_prev and k_current < d_current and k_current > 80 and d_current > 80:
            signal = -1
            
        return signal
    
    def get_signal(self):
        """Get trading signal from all strategies"""
        df = self.get_rates(50)  # Get last 50 bars
        if df is None or len(df) < 20:
            return 0, "No data"
        
        # Get signals from all strategies
        ha_signal = self.heikin_ashi_signal(df)
        rsi_signal = self.rsi_extremes_signal(df)
        stoch_signal = self.stochastic_signal(df)
        
        # Combine signals (majority vote)
        signals = [ha_signal, rsi_signal, stoch_signal]
        buy_signals = sum(1 for s in signals if s == 1)
        sell_signals = sum(1 for s in signals if s == -1)
        
        if buy_signals >= 2:
            return 1, f"BUY (HA: {ha_signal}, RSI: {rsi_signal}, Stoch: {stoch_signal})"
        elif sell_signals >= 2:
            return -1, f"SELL (HA: {ha_signal}, RSI: {rsi_signal}, Stoch: {stoch_signal})"
        else:
            return 0, f"NO SIGNAL (HA: {ha_signal}, RSI: {rsi_signal}, Stoch: {stoch_signal})"
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk management"""
        if not self.initialized:
            return 0
            
        # Get account balance
        account_info = mt5.account_info()
        if account_info is None:
            return 0
            
        balance = account_info.balance
        risk_amount = balance * self.risk_per_trade
        
        # Calculate stop loss in points
        point = mt5.symbol_info(self.symbol).point
        digits = mt5.symbol_info(self.symbol).digits
        
        if entry_price > stop_loss_price:  # Buy position
            sl_points = (entry_price - stop_loss_price) / point
        else:  # Sell position
            sl_points = (stop_loss_price - entry_price) / point
            
        # Calculate tick value
        tick_value = (mt5.symbol_info(self.symbol).trade_tick_value_profit * 
                     mt5.symbol_info(self.symbol).trade_contract_size)
        
        # Calculate position size
        position_size = risk_amount / (sl_points * tick_value)
        
        # Normalize position size to allowed lot steps
        lot_step = mt5.symbol_info(self.symbol).volume_step
        position_size = round(position_size / lot_step) * lot_step
        
        # Apply min and max limits
        min_lot = mt5.symbol_info(self.symbol).volume_min
        max_lot = mt5.symbol_info(self.symbol).volume_max
        position_size = max(min(position_size, max_lot), min_lot)
        
        return position_size
    
    def place_trade(self, signal, comment=""):
        """Place a trade based on the signal"""
        if not self.initialized:
            return False
            
        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
            
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": 0.1,  # Default volume, will be updated
            "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if signal == 1 else tick.bid,
            "sl": 0.0,
            "tp": 0.0,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Calculate stop loss and take profit
        if signal == 1:  # Buy
            # For buy orders, SL is below current price, TP is above
            sl_price = tick.bid - 0.0010  # 10 pips SL
            tp_price = tick.ask + 0.0020  # 20 pips TP
        else:  # Sell
            # For sell orders, SL is above current price, TP is below
            sl_price = tick.ask + 0.0010  # 10 pips SL
            tp_price = tick.bid - 0.0020  # 20 pips TP
            
        request["sl"] = sl_price
        request["tp"] = tp_price
        
        # Calculate position size based on risk
        request["volume"] = self.calculate_position_size(request["price"], sl_price)
        
        # Send trade request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Trade failed: {result.retcode}")
            return False
            
        print(f"Trade executed: {result}")
        return True
    
    def check_existing_positions(self):
        """Check if there are existing positions for this symbol and magic number"""
        if not self.initialized:
            return 0
            
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return 0
            
        # Filter by magic number
        our_positions = [p for p in positions if p.magic == self.magic_number]
        return len(our_positions)
    
    def run_scalper(self, max_positions=10):
        """Run the scalping strategy"""
        if not self.initialize():
            return
            
        print(f"Starting scalper for {self.symbol}...")
        
        try:
            while True:
                # Check if we already have max positions
                if self.check_existing_positions() >= max_positions:
                    print("Max positions reached, waiting...")
                    time.sleep(10)
                    continue
                
                # Get trading signal
                signal, message = self.get_signal()
                print(f"{datetime.now()}: {message}")
                
                # Place trade if we have a signal
                if signal != 0:
                    success = self.place_trade(signal, message)
                    if success:
                        print(f"Trade placed based on: {message}")
                    else:
                        print("Failed to place trade")
                
                # Wait before next check
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("Stopping scalper...")
        finally:
            self.shutdown()

# Example usage
if __name__ == "__main__":
    # Create scalper for EURUSD
    scalper = UltraFastScalper("EURUSD")
    
    # Run the scalper (will run until interrupted)
    scalper.run_scalper()