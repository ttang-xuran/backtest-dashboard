# Statistical Arbitrage Backtesting Guide

## Overview
You now have a comprehensive backtesting system for your BTC/ETH statistical arbitrage strategy with beta hedging. This guide explains how to use the different components.

## Available Tools

### 1. Simple Backtester (`simple_backtest.py`)
**Best for:** Quick testing and initial validation

```bash
python simple_backtest.py
```

**Features:**
- ✅ Beta hedging implementation
- ✅ Realistic P&L calculation  
- ✅ Trading cost simulation
- ✅ Clean, readable results
- ✅ 3-day test with 418 trades in example

**Sample Output:**
```
Initial Capital:  $100,000.00
Final Value:      $93,461.23
Total Return:     -6.54%
Number of Trades: 418
Win Rate:         64.5%
```

### 2. Full Backtesting Engine (`backtest_engine.py`)
**Best for:** Comprehensive analysis with plots

```bash
python backtest_engine.py
```

**Features:**
- ✅ Advanced performance metrics
- ✅ Sharpe ratio, drawdown analysis
- ✅ Visual plots and charts
- ✅ Multiple data sources (sample, CSV, API)
- ⚠️ Currently has scaling issues (being fixed)

### 3. Parameter Optimizer (`parameter_optimizer.py`)
**Best for:** Finding optimal strategy parameters

```bash
python parameter_optimizer.py
```

**Features:**
- ✅ Grid search optimization
- ✅ Walk-forward analysis
- ✅ Parameter sensitivity testing
- ✅ Parallel processing support

## How to Use

### Basic Backtesting
1. **Quick Test:**
   ```bash
   python simple_backtest.py
   ```

2. **Custom Parameters:**
   Edit `config.json`:
   ```json
   {
     "z_score_entry": 1.5,
     "z_score_exit": 0.3,
     "lookback_period": 20,
     "position_size": 0.25
   }
   ```

### Parameter Optimization
```python
from parameter_optimizer import ParameterOptimizer
from simple_backtest import SimpleBacktester

# Initialize
optimizer = ParameterOptimizer()
backtester = SimpleBacktester()

# Generate data
data = backtester.generate_sample_data(days=10)

# Define parameter ranges
param_ranges = {
    'z_score_entry': [1.0, 1.5, 2.0, 2.5],
    'z_score_exit': [0.1, 0.3, 0.5],
    'lookback_period': [10, 15, 20, 25],
    'position_size': [0.1, 0.15, 0.2, 0.25]
}

# Optimize
best_result = optimizer.grid_search_optimization(
    data, param_ranges, objective="sharpe_ratio"
)

print("Best Parameters:", best_result['parameters'])
```

### Using Real Data
1. **From CSV file:**
   ```python
   # Create CSV with columns: timestamp, btc_price, eth_price
   backtester = SimpleBacktester()
   
   # Modify load_from_csv method to load your data
   data = pd.read_csv('your_data.csv')
   data['timestamp'] = pd.to_datetime(data['timestamp'])
   
   results = backtester.run_backtest(data)
   ```

2. **From API (extend the code):**
   ```python
   # Add API data fetching to backtester
   def fetch_hyperliquid_data(days=7):
       # Implement API calls to get historical data
       pass
   ```

## Key Metrics Explained

### Performance Metrics
- **Total Return:** Overall profit/loss percentage
- **Sharpe Ratio:** Risk-adjusted returns
- **Win Rate:** Percentage of profitable trades
- **Max Drawdown:** Largest peak-to-trough decline

### Strategy Metrics  
- **Z-Score:** Statistical measure of price spread deviation
- **Beta:** ETH volatility relative to BTC (for hedging)
- **Position Size:** Percentage of capital per trade

## Beta Hedging in Action

The backtester implements proper beta hedging:

```python
# Example from recent run:
# ENTER SHORT BTC, LONG ETH | Z-score: 2.64 | Beta: 0.69

# This means:
# - BTC is expensive relative to ETH (positive Z-score)
# - SHORT $25,000 BTC + LONG $17,250 ETH (25k * 0.69 beta)
# - True market neutral position
```

## Performance Analysis

### Sample 3-Day Backtest Results:
- **418 trades** in 3 days (very active)
- **64.5% win rate** (decent for stat arb)
- **-6.54% return** (might need parameter tuning)
- **Beta range:** 0.69 - 1.02 (reasonable for BTC/ETH)

### Optimization Opportunities:
1. **Reduce trading frequency:** Increase Z-score thresholds
2. **Improve risk management:** Add stop-losses
3. **Fine-tune parameters:** Use optimizer to find better settings
4. **Extend holding periods:** Adjust exit thresholds

## Next Steps

1. **Parameter Optimization:**
   ```bash
   python parameter_optimizer.py
   ```

2. **Test with different time periods:**
   ```python
   data = backtester.generate_sample_data(days=30)  # Month-long test
   ```

3. **Add your own data:**
   - Export from TradingView
   - Use exchange APIs
   - Historical data providers

4. **Production Integration:**
   - Connect to live bot for validation
   - Implement paper trading
   - Add real-time monitoring

## File Structure
```
tony-project/
├── simple_backtest.py         # Main backtesting tool
├── backtest_engine.py          # Advanced backtester  
├── parameter_optimizer.py      # Parameter optimization
├── hyperliquid_stat_arb.py    # Live trading bot
├── config.json                # Strategy parameters
└── BACKTESTING_GUIDE.md       # This guide
```

## Tips for Success

1. **Start Simple:** Use `simple_backtest.py` first
2. **Validate Parameters:** Use optimizer before live trading  
3. **Test Different Markets:** Bull/bear/sideways conditions
4. **Monitor Beta:** Should be 0.5-1.5 for BTC/ETH
5. **Risk Management:** Never risk more than you can afford to lose

Your backtesting system is now ready for comprehensive strategy validation! 🚀