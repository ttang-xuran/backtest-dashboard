#!/usr/bin/env python3
"""
Strategy Comparison: Original vs Improved
Quick comparison of performance improvements
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class QuickComparisonTest:
    def __init__(self):
        """Initialize comparison test"""
        
        # ORIGINAL STRATEGY (problematic)
        self.original_params = {
            'position_pct': 0.25,      # Too high risk
            'z_entry': 1.5,            # Too low (overtrading)
            'z_exit': 0.3,             # Too quick exit
            'lookback': 20,            # Too short
            'fee': 0.0005,             # Higher fees
            'max_trades_per_day': 999  # No limit
        }
        
        # IMPROVED STRATEGY (optimized)
        self.improved_params = {
            'position_pct': 0.15,      # Reduced risk
            'z_entry': 2.0,            # Higher conviction
            'z_exit': 0.5,             # Let profits run
            'lookback': 40,            # More stable
            'fee': 0.0003,             # Better fees
            'max_trades_per_day': 10   # Limit overtrading
        }

    def generate_test_data(self, days: int = 3) -> pd.DataFrame:
        """Generate consistent test data"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days * 24 * 120,
            freq='30s'
        )
        
        n = len(timestamps)
        np.random.seed(42)  # Fixed seed for comparison
        
        # Realistic BTC/ETH data
        btc_returns = np.random.normal(0.0001, 0.02, n)
        btc_prices = [118000]
        for ret in btc_returns[1:]:
            btc_prices.append(btc_prices[-1] * (1 + ret))
        
        # Correlated ETH
        correlation = 0.75
        eth_returns = []
        for btc_ret in btc_returns:
            corr_part = correlation * btc_ret
            noise_part = np.sqrt(1 - correlation**2) * np.random.normal(0, 0.018)
            eth_returns.append(corr_part + noise_part)
        
        eth_prices = [3800]
        for ret in eth_returns[1:]:
            eth_prices.append(eth_prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'btc_price': btc_prices,
            'eth_price': eth_prices
        })

    def run_strategy(self, data: pd.DataFrame, params: dict, strategy_name: str) -> dict:
        """Run a strategy with given parameters"""
        initial_capital = 100000
        capital = initial_capital
        
        btc_history = []
        eth_history = []
        
        position = None
        entry_capital = 0
        entry_btc_price = 0
        entry_eth_price = 0
        
        trades = []
        daily_trades = {}
        
        for _, row in data.iterrows():
            btc_price = row['btc_price']
            eth_price = row['eth_price']
            timestamp = row['timestamp']
            
            btc_history.append(btc_price)
            eth_history.append(eth_price)
            
            # Keep lookback period
            if len(btc_history) > params['lookback']:
                btc_history = btc_history[-params['lookback']:]
                eth_history = eth_history[-params['lookback']:]
            
            # Need minimum data
            if len(btc_history) < params['lookback']:
                continue
            
            # Calculate Z-score
            spreads = [np.log(btc_history[i] / eth_history[i]) for i in range(len(btc_history))]
            current_spread = spreads[-1]
            mean_spread = np.mean(spreads)
            std_spread = np.std(spreads)
            
            z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0
            
            # Trading logic
            if position is None and abs(z_score) > params['z_entry']:
                # Check daily trade limit
                date_key = timestamp.strftime('%Y-%m-%d')
                daily_count = daily_trades.get(date_key, 0)
                
                if daily_count < params['max_trades_per_day']:
                    # Enter trade
                    entry_capital = capital * params['position_pct']
                    trading_cost = entry_capital * params['fee'] * 2
                    capital -= trading_cost
                    
                    entry_btc_price = btc_price
                    entry_eth_price = eth_price
                    
                    daily_trades[date_key] = daily_count + 1
                    
                    if z_score > params['z_entry']:
                        position = 'short_btc'
                    else:
                        position = 'long_btc'
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'ENTER',
                        'z_score': z_score,
                        'capital': capital
                    })
                    
            elif position is not None and abs(z_score) < params['z_exit']:
                # Exit trade
                if position == 'long_btc':
                    btc_pnl = (btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = -(eth_price - entry_eth_price) / entry_eth_price
                    total_pnl = (btc_pnl + eth_pnl) * entry_capital
                else:
                    btc_pnl = -(btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = (eth_price - entry_eth_price) / entry_eth_price
                    total_pnl = (btc_pnl + eth_pnl) * entry_capital
                
                trading_cost = entry_capital * params['fee'] * 2
                capital += total_pnl - trading_cost
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'EXIT',
                    'pnl': total_pnl,
                    'capital': capital
                })
                
                position = None
        
        # Calculate results
        final_value = capital
        total_return = (final_value - initial_capital) / initial_capital
        
        num_trades = len([t for t in trades if t['action'] == 'ENTER'])
        exit_trades = [t for t in trades if t['action'] == 'EXIT']
        winning_trades = len([t for t in exit_trades if t['pnl'] > 0])
        win_rate = winning_trades / max(len(exit_trades), 1)
        
        return {
            'strategy': strategy_name,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': final_value - initial_capital,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'trades': trades
        }

    def run_comparison(self):
        """Run side-by-side comparison"""
        print("🔥 STRATEGY PERFORMANCE COMPARISON")
        print("="*60)
        
        # Generate test data
        data = self.generate_test_data(days=3)
        print(f"📊 Testing with {len(data)} data points (3 days)")
        
        # Run original strategy
        print("\n⏳ Running original strategy...")
        original_results = self.run_strategy(data, self.original_params, "Original")
        
        # Run improved strategy
        print("⏳ Running improved strategy...")
        improved_results = self.run_strategy(data, self.improved_params, "Improved")
        
        # Display comparison
        self.print_comparison(original_results, improved_results)
        
        return original_results, improved_results

    def print_comparison(self, original: dict, improved: dict):
        """Print side-by-side comparison"""
        print("\n" + "="*80)
        print("📈 PERFORMANCE COMPARISON RESULTS")
        print("="*80)
        
        metrics = [
            ('Final Value', 'final_value', '${:,.2f}'),
            ('Total Return', 'total_return', '{:.2f}%'),
            ('Total P&L', 'total_pnl', '${:,.2f}'),
            ('Number of Trades', 'num_trades', '{:,}'),
            ('Win Rate', 'win_rate', '{:.1f}%')
        ]
        
        print(f"{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
        print("-" * 65)
        
        for metric_name, key, fmt in metrics:
            orig_val = original[key]
            impr_val = improved[key]
            
            if key == 'total_return':
                orig_display = fmt.format(orig_val * 100)
                impr_display = fmt.format(impr_val * 100)
                change = impr_val - orig_val
                change_display = f"{change*100:+.2f}%"
            elif key == 'win_rate':
                orig_display = fmt.format(orig_val * 100)
                impr_display = fmt.format(impr_val * 100)
                change = impr_val - orig_val
                change_display = f"{change*100:+.1f}%"
            else:
                orig_display = fmt.format(orig_val)
                impr_display = fmt.format(impr_val)
                if key in ['final_value', 'total_pnl']:
                    change = impr_val - orig_val
                    change_display = f"${change:+,.2f}"
                else:
                    change = impr_val - orig_val
                    change_display = f"{change:+,}"
            
            print(f"{metric_name:<20} {orig_display:<15} {impr_display:<15} {change_display:<15}")
        
        print("="*80)
        
        # Key improvements summary
        print("\n🎯 KEY IMPROVEMENTS:")
        
        return_improvement = (improved['total_return'] - original['total_return']) * 100
        trade_reduction = original['num_trades'] - improved['num_trades']
        
        if return_improvement > 0:
            print(f"✅ Return improved by {return_improvement:.2f} percentage points")
        else:
            print(f"❌ Return decreased by {abs(return_improvement):.2f} percentage points")
        
        if trade_reduction > 0:
            print(f"✅ Reduced overtrading by {trade_reduction} trades ({trade_reduction/original['num_trades']*100:.1f}% fewer)")
        
        win_rate_improvement = (improved['win_rate'] - original['win_rate']) * 100
        if win_rate_improvement > 0:
            print(f"✅ Win rate improved by {win_rate_improvement:.1f} percentage points")
        
        print(f"\n💡 STRATEGY CHANGES:")
        print(f"   📉 Reduced position size: 25% → 15% (lower risk)")
        print(f"   📈 Higher entry threshold: 1.5 → 2.0 (less noise)")
        print(f"   ⏱️  Better exit timing: 0.3 → 0.5 (let profits run)")
        print(f"   📊 Longer lookback: 20 → 40 periods (more stable)")
        print(f"   💰 Lower fees: 0.05% → 0.03% (better execution)")
        print(f"   🚫 Trade limit: ∞ → 10/day (prevent overtrading)")


def main():
    """Run the comparison test"""
    tester = QuickComparisonTest()
    original_results, improved_results = tester.run_comparison()
    return original_results, improved_results


if __name__ == "__main__":
    results = main()