import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import MetaTrader5 as mt5

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ema_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EMATradingBot:
    def __init__(self, account_number, password, server, initial_balance=10000):
        self.account_number = account_number
        self.password = password
        self.server = server
        self.initial_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.failed_symbols = set()
        self.manual_confirmation = False
        self.available_symbols = []
        
        # EMA parameters
        self.fast_ema_period = 9
        self.slow_ema_period = 21
        self.timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe
        self.bars_to_analyze = 100
        
        # Trading parameters
        self.max_symbols_to_trade = 20  # Limit number of symbols to trade
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.fixed_take_profit = 100.00  # Fixed $1.00 take profit
        
    def initialize_mt5(self):
        """Initialize connection to MT5 terminal"""
        try:
            # Close any existing connection
            if mt5.initialize():
                mt5.shutdown()
                time.sleep(1)
            
            # Initialize with specific account details
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                logger.error(f"Error code: {mt5.last_error()}")
                return False
            
            # Attempt to login
            authorized = mt5.login(
                login=self.account_number,
                password=self.password,
                server=self.server
            )
            
            if not authorized:
                logger.error(f"Login failed. Error: {mt5.last_error()}")
                return False
            
            # Check connection status
            terminal_info = mt5.terminal_info()
            if not terminal_info.connected:
                logger.error("MT5 terminal is not connected to server")
                return False
                
            # Check if trading is allowed
            trade_allowed = getattr(terminal_info, 'trade_allowed', False)
            
            if not trade_allowed:
                logger.warning("Trading not allowed by terminal. Using manual confirmation mode.")
                self.manual_confirmation = True
                
            logger.info("MT5 initialized successfully")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Connected to: {getattr(account_info, 'server', 'Unknown')}")
                logger.info(f"Account: {account_info.login}")
                logger.info(f"Balance: ${account_info.balance}")
                logger.info(f"Equity: ${account_info.equity}")
            
            # Get only forex symbols from Market Watch
            self.available_symbols = self.get_forex_symbols()
            logger.info(f"Found {len(self.available_symbols)} forex symbols")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization exception: {e}")
            return False
    
    def get_forex_symbols(self):
        """Get only forex symbols from Market Watch"""
        forex_symbols = []
        
        try:
            # Get all symbols and filter for forex only
            all_symbols = mt5.symbols_get()
            for symbol in all_symbols:
                if (symbol.visible and symbol.select and 
                    hasattr(symbol, 'path') and 'Forex' in symbol.path):
                    forex_symbols.append(symbol.name)
            
            # If no forex symbols found, use major forex pairs
            if not forex_symbols:
                major_forex = [
                    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 
                    'AUDUSD', 'NZDUSD', 'EURGBP', 'EURJPY', 'EURCHF',
                    'GBPJPY', 'GBPCHF', 'AUDJPY', 'CADJPY', 'CHFJPY'
                ]
                for symbol in major_forex:
                    if mt5.symbol_select(symbol, True):
                        forex_symbols.append(symbol)
            
            # Limit to max symbols to trade
            return forex_symbols[:self.max_symbols_to_trade]
            
        except Exception as e:
            logger.error(f"Error getting forex symbols: {e}")
            return []
    
    def calculate_ema(self, symbol):
        """Calculate EMA values for a symbol without TA-Lib"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, self.bars_to_analyze)
            
            if rates is None or len(rates) < self.slow_ema_period + 1:
                logger.warning(f"Not enough data for {symbol}")
                return None, None, None, None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate EMAs manually
            close_prices = df['close'].values
            
            # Fast EMA calculation
            fast_ema = self.calculate_ema_manual(close_prices, self.fast_ema_period)
            
            # Slow EMA calculation
            slow_ema = self.calculate_ema_manual(close_prices, self.slow_ema_period)
            
            # Get current and previous values
            if len(fast_ema) > 1 and len(slow_ema) > 1:
                current_fast_ema = fast_ema[-1]
                current_slow_ema = slow_ema[-1]
                prev_fast_ema = fast_ema[-2]
                prev_slow_ema = slow_ema[-2]
                
                return current_fast_ema, current_slow_ema, prev_fast_ema, prev_slow_ema
            
            return None, None, None, None
            
        except Exception as e:
            logger.error(f"Error calculating EMA for {symbol}: {e}")
            return None, None, None, None
    
    def calculate_ema_manual(self, prices, period):
        """Manual EMA calculation without TA-Lib"""
        if len(prices) < period:
            return []
        
        ema = []
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        ema.append(sum(prices[:period]) / period)
        
        # Calculate EMA for remaining values
        for price in prices[period:]:
            ema_value = (price * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return ema
    
    def get_trading_signal(self, symbol):
        """Get trading signal based on EMA crossover"""
        fast_ema, slow_ema, prev_fast_ema, prev_slow_ema = self.calculate_ema(symbol)
        
        if any(x is None for x in [fast_ema, slow_ema, prev_fast_ema, prev_slow_ema]):
            return 'HOLD'
        
        # Check for bullish crossover (fast EMA crosses above slow EMA)
        if prev_fast_ema <= prev_slow_ema and fast_ema > slow_ema:
            return 'BUY'
        
        # Check for bearish crossover (fast EMA crosses below slow EMA)
        if prev_fast_ema >= prev_slow_ema and fast_ema < slow_ema:
            return 'SELL'
        
        return 'HOLD'
    
    def should_enter_trade(self, symbol, signal):
        """Determine if we should enter a trade"""
        if signal == 'HOLD' or symbol in self.failed_symbols:
            return False
        
        # Check if we already have an open position for this symbol
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is not None and len(positions) > 0:
                logger.debug(f"Already have position for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error checking existing positions for {symbol}: {e}")
        
        return True
    
    def calculate_lot_size(self, symbol, price):
        """Calculate appropriate lot size for a symbol"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return 0.01  # Default minimum
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01
            
            # Get symbol properties
            contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
            min_lot = getattr(symbol_info, 'volume_min', 0.01)
            max_lot = getattr(symbol_info, 'volume_max', 100)
            lot_step = getattr(symbol_info, 'volume_step', 0.01)
            
            # Calculate position size based on risk
            risk_amount = account_info.equity * self.risk_per_trade
            lot_size = risk_amount / (price * contract_size)
            
            # Adjust to lot step
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Ensure within min/max limits
            lot_size = max(min_lot, min(max_lot, lot_size))
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Error calculating lot size for {symbol}: {e}")
            return 0.01
    
    def calculate_take_profit_price(self, symbol, order_type, entry_price):
        """Calculate take profit price for $1.00 fixed profit"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0
                
            # Get contract size and tick value
            contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
            tick_value = getattr(symbol_info, 'trade_tick_value', 0)
            tick_size = getattr(symbol_info, 'trade_tick_size', 0.00001)
            
            if tick_value <= 0 or tick_size <= 0:
                return 0
                
            # Calculate required price movement for $1.00 profit
            required_ticks = self.fixed_take_profit / tick_value
            
            if order_type == mt5.ORDER_TYPE_BUY:
                take_profit_price = entry_price + (required_ticks * tick_size)
            else:  # SELL
                take_profit_price = entry_price - (required_ticks * tick_size)
                
            return take_profit_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit for {symbol}: {e}")
            return 0
    
    def execute_trade(self, symbol, signal):
        """Execute a trade based on EMA signal"""
        try:
            # Get current price
            symbol_info_tick = mt5.symbol_info_tick(symbol)
            if symbol_info_tick is None:
                logger.error(f"Could not get price info for {symbol}")
                return False
            
            if signal == 'BUY':
                price = symbol_info_tick.ask
                order_type = mt5.ORDER_TYPE_BUY
            else:  # SELL
                price = symbol_info_tick.bid
                order_type = mt5.ORDER_TYPE_SELL
            
            # Calculate lot size
            lot_size = self.calculate_lot_size(symbol, price)
            
            # Calculate take profit price for $1.00 fixed profit
            take_profit_price = self.calculate_take_profit_price(symbol, order_type, price)
            
            if take_profit_price == 0:
                logger.warning(f"Could not calculate take profit for {symbol}, using default 10 pip TP")
                if order_type == mt5.ORDER_TYPE_BUY:
                    take_profit_price = price + (0.0010 * price)  # Default 10 pips
                else:
                    take_profit_price = price - (0.0010 * price)  # Default 10 pips
            
            if self.manual_confirmation:
                # Manual confirmation mode
                logger.info("="*50)
                logger.info("EMA TRADE SIGNAL (Manual Confirmation Required)")
                logger.info(f"Symbol: {symbol}")
                logger.info(f"Signal: {signal}")
                logger.info(f"Recommended Lot Size: {lot_size}")
                logger.info(f"Entry Price: {price}")
                logger.info(f"Take Profit Price: {take_profit_price:.5f}")
                logger.info(f"Take Profit Amount: ${self.fixed_take_profit:.2f}")
                logger.info("Please execute this trade manually in MT5")
                logger.info("="*50)
                return True
            else:
                # Automated trading mode - use IOC filling mode instead of FOK
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": price,
                    "sl": 0.0,  # No stop loss
                    "tp": take_profit_price,
                    "deviation": 20,
                    "magic": 123456,
                    "comment": f"EMA {self.fast_ema_period}/{self.slow_ema_period} crossover",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,  # Changed from FOK to IOC
                }
                
                # Send the order
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Order failed for {symbol}: {result.retcode} - {result.comment}")
                    
                    if "disabled" in result.comment.lower():
                        logger.error(f"Trading disabled for {symbol}. Adding to failed symbols.")
                        self.failed_symbols.add(symbol)
                    
                    return False
                
                logger.info(f"Executed {signal} trade for {symbol} with size {lot_size}")
                logger.info(f"Take profit set at {take_profit_price:.5f} for ${self.fixed_take_profit:.2f} profit")
                return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def manage_open_positions(self):
        """Manage and close positions based on EMA signals"""
        closed_positions = []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return closed_positions
                
            for position in positions:
                symbol = position.symbol
                signal = self.get_trading_signal(symbol)
                
                # Close position if signal is opposite to position direction
                if (position.type == mt5.ORDER_TYPE_BUY and signal == 'SELL') or \
                   (position.type == mt5.ORDER_TYPE_SELL and signal == 'BUY'):
                    
                    # Close the position
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                        
                    close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
                    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": position.ticket,
                        "symbol": symbol,
                        "volume": position.volume,
                        "type": close_type,
                        "price": close_price,
                        "deviation": 20,
                        "magic": 123456,
                        "comment": "Close EMA position",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,  # Changed from FOK to IOC
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Closed position {position.ticket} for {symbol} with P&L {position.profit}")
                        closed_positions.append(position)
                    else:
                        logger.error(f"Failed to close position {position.ticket}: {result.retcode} - {result.comment}")
                        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
        
        return closed_positions
    
    def run(self, execution_interval=300):  # 5 minutes
        """Main trading loop"""
        logger.info("Starting EMA Trading Bot")
        
        # Initialize MT5 connection
        if not self.initialize_mt5():
            logger.error("Failed to initialize MT5. Exiting.")
            return
        
        try:
            while True:
                try:
                    logger.info("="*60)
                    logger.info("SCANNING FOR EMA TRADING OPPORTUNITIES")
                    logger.info("="*60)
                    
                    # Manage existing positions
                    self.manage_open_positions()
                    
                    # Check for new trading opportunities
                    trade_count = 0
                    for symbol in self.available_symbols:
                        if trade_count >= 5:  # Limit to 5 trades per cycle
                            break
                            
                        try:
                            signal = self.get_trading_signal(symbol)
                            
                            if signal != 'HOLD':
                                logger.info(f"Signal for {symbol}: {signal}")
                                
                                if self.should_enter_trade(symbol, signal):
                                    if self.execute_trade(symbol, signal):
                                        trade_count += 1
                                        time.sleep(1)  # Small delay between trades
                            else:
                                logger.debug(f"No signal for {symbol}: HOLD")
                                
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                            continue
                    
                    # Log current status
                    account_info = mt5.account_info()
                    if account_info:
                        logger.info(f"Account Status: Equity: ${account_info.equity:.2f} | Balance: ${account_info.balance:.2f} | Profit: ${account_info.profit:.2f}")
                    
                    # Log failed symbols
                    if self.failed_symbols:
                        logger.warning(f"Symbols with trading issues: {list(self.failed_symbols)}")
                    
                    # Wait for next execution cycle
                    logger.info(f"Next analysis in {execution_interval} seconds...")
                    time.sleep(execution_interval)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(60)  # Wait a minute before continuing
                    
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in trading bot: {e}")
        finally:
            mt5.shutdown()
            logger.info("Trading bot shutdown complete")

# Run the bot
if __name__ == "__main__":
    print("Starting EMA Trading Bot for Forex Symbols...")
    bot = EMATradingBot(
        account_number=526136,
        password="Emilosano@60",
        server="EGMSecurities-Demo",
        initial_balance=10000
    )
    bot.run()