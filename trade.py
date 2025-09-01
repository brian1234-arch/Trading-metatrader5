import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS
import warnings
warnings.filterwarnings("ignore")

# Define potential pairs to test (grouped by sector/industry)
pairs_to_test = [
    # ETFs tracking similar indices
    ("SPY", "IVV"),  # S&P 500 ETFs
    ("QQQ", "ONEQ"), # Nasdaq ETFs
    ("DIA", "IYY"),  # Dow Jones ETFs
    
    # Tech stocks
    ("AAPL", "MSFT"),
    ("GOOGL", "GOOG"),  # Different share classes
    ("AMD", "NVDA"),
    
    # Banking sector
    ("JPM", "BAC"),
    ("WFC", "C"),
    
    # Consumer goods
    ("KO", "PEP"),
    ("MCD", "SBUX"),
    
    # E-commerce
    ("AMZN", "EBAY"),
    
    # Social media
    ("META", "TWTR"),  # Note: TWTR is now X, but ticker might still work
    
    # Automotive
    ("TSLA", "NIO"),
    ("F", "GM"),
    
    # Semiconductor
    ("INTC", "TXN"),
    
    # Pharmaceutical
    ("PFE", "JNJ"),
    ("MRK", "ABBV"),
]

# Parameters for testing
initial_capital = 100.0
lot_size = 0.01
start_date = "2020-01-01"
end_date = "2023-01-01"

# Function to test a single pair
def test_pair(ticker1, ticker2):
    try:
        # Download data
        data = yf.download([ticker1, ticker2], start=start_date, end=end_date, auto_adjust=False)
        
        if data.empty:
            return None
        
        # Extract prices
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                adj_close = data.xs('Adj Close', axis=1, level=0)
            else:
                adj_close = data.xs('Close', axis=1, level=0)
        else:
            if 'Adj Close' in data.columns:
                adj_close = data['Adj Close']
            else:
                adj_close = data['Close']
        
        # Clean data
        adj_close = adj_close.dropna()
        if len(adj_close) < 100:  # Need sufficient data
            return None
        
        # Check for cointegration
        score, pvalue, _ = coint(adj_close[ticker1], adj_close[ticker2])
        
        if pvalue >= 0.05:  # Not cointegrated
            return None
        
        # Calculate hedge ratio
        model = OLS(adj_close[ticker1], adj_close[ticker2])
        results = model.fit()
        hedge_ratio = results.params[0]
        
        # Calculate spread and z-score
        spread = adj_close[ticker1] - hedge_ratio * adj_close[ticker2]
        zscore = (spread - spread.mean()) / spread.std()
        
        # Define thresholds
        entry_threshold = 2.0
        exit_threshold = 0.5
        
        # Generate signals
        positions = pd.Series(0, index=zscore.index)
        positions[zscore >= entry_threshold] = -1
        positions[zscore <= -entry_threshold] = 1
        positions[abs(zscore) <= exit_threshold] = 0
        
        # Calculate returns
        returns = pd.DataFrame(index=adj_close.index)
        returns['asset1_returns'] = adj_close[ticker1].pct_change()
        returns['asset2_returns'] = adj_close[ticker2].pct_change()
        returns['spread_returns'] = returns['asset1_returns'] - hedge_ratio * returns['asset2_returns']
        returns['strategy_returns'] = positions.shift(1) * returns['spread_returns']
        
        # Calculate performance metrics
        cumulative_returns = (1 + returns['strategy_returns']).cumprod()
        final_value = initial_capital * cumulative_returns.iloc[-1] if not cumulative_returns.empty else initial_capital
        total_return = (final_value / initial_capital - 1) * 100
        
        # Calculate Sharpe ratio
        daily_returns = returns['strategy_returns'].dropna()
        if daily_returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        # Count number of trades
        trades = positions.diff().fillna(0)
        num_trades = abs(trades).sum() / 2  # Each trade has entry and exit
        
        return {
            'pair': f"{ticker1}-{ticker2}",
            'pvalue': pvalue,
            'hedge_ratio': hedge_ratio,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'final_value': final_value
        }
    
    except Exception as e:
        print(f"Error testing {ticker1}-{ticker2}: {str(e)}")
        return None

# Test all pairs
results = []
for ticker1, ticker2 in pairs_to_test:
    print(f"Testing {ticker1}-{ticker2}...")
    result = test_pair(ticker1, ticker2)
    if result:
        results.append(result)

# Sort results by Sharpe ratio (descending) and then by total return (descending)
results.sort(key=lambda x: (-x['sharpe_ratio'], -x['total_return']))

# Display results
print("\n" + "="*80)
print("PAIR ARBITRAGE TEST RESULTS")
print("="*80)
print(f"{'Pair':<12} {'P-value':<10} {'Hedge Ratio':<12} {'Return %':<10} {'Sharpe':<8} {'Trades':<8} {'Final Value':<12}")
print("-"*80)

for result in results:
    print(f"{result['pair']:<12} {result['pvalue']:.4f}    {result['hedge_ratio']:.4f}      {result['total_return']:>6.2f}%   {result['sharpe_ratio']:>5.2f}   {result['num_trades']:>5.0f}   ${result['final_value']:>9.2f}")

# Display the best pair
if results:
    best_pair = results[0]
    print("\n" + "="*80)
    print("BEST PAIR FOR STATISTICAL ARBITRAGE:")
    print("="*80)
    print(f"Pair: {best_pair['pair']}")
    print(f"Cointegration p-value: {best_pair['pvalue']:.6f}")
    print(f"Hedge ratio: {best_pair['hedge_ratio']:.4f}")
    print(f"Total return: {best_pair['total_return']:.2f}%")
    print(f"Sharpe ratio: {best_pair['sharpe_ratio']:.2f}")
    print(f"Number of trades: {best_pair['num_trades']:.0f}")
    print(f"Final value: ${best_pair['final_value']:.2f} (from ${initial_capital:.2f})")
else:
    print("No cointegrated pairs found.")