#!/usr/bin/env python3
"""
Improved Statistical Arbitrage Backtester
Optimized version with better parameters and risk management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class ImprovedBacktester:
    def __init__(self, config_path: str = "config.json"):
        """Initialize improved backtester with optimized parameters"""
        self.config = self.load_config(config_path)
        
        # IMPROVED STRATEGY PARAMETERS
        self.position_pct = 0.15  # Reduced from 25% to 15% (less risk)
        self.z_entry = 2.0        # Increased from 1.5 to 2.0 (higher conviction)
        self.z_exit = 0.5         # Increased from 0.3 to 0.5 (let profits run)
        self.lookback = 40        # Increased from 20 to 40 (more stable statistics)
        
        # RISK MANAGEMENT IMPROVEMENTS
        self.max_holding_minutes = 120  # Maximum 2 hours per trade
        self.min_holding_minutes = 30   # Minimum 30 minutes (avoid noise)
        self.daily_trade_limit = 10     # Max 10 trades per day
        self.stop_loss_pct = 0.03       # 3% stop loss
        
        # TRADING COST OPTIMIZATION
        self.fee = 0.0003  # Reduced fees (0.03% - achievable with volume)
        
        # ADAPTIVE PARAMETERS
        self.volatility_threshold = 0.05  # Don't trade in high volatility
        self.correlation_min = 0.6        # Minimum BTC-ETH correlation
        
        # Results tracking
        self.trades = []
        self.daily_trades = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def generate_sample_data(self, days: int = 7, volatility_regime: str = "normal") -> pd.DataFrame:
        """Generate realistic sample data with different market regimes"""
        self.logger.info(f"Generating {days} days of {volatility_regime} market data...")
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days * 24 * 120,  # 30-second intervals
            freq='30s'
        )
        
        n = len(timestamps)
        np.random.seed(42)
        
        # Adjust volatility by regime
        if volatility_regime == "low":
            btc_vol, eth_vol = 0.015, 0.012
        elif volatility_regime == "high":
            btc_vol, eth_vol = 0.04, 0.035
        else:  # normal
            btc_vol, eth_vol = 0.02, 0.018
        
        # Generate correlated price movements with regime
        correlation = 0.75  # Slightly higher correlation
        
        # BTC prices
        btc_returns = np.random.normal(0.0001, btc_vol, n)
        btc_prices = [118000]
        for ret in btc_returns[1:]:
            btc_prices.append(btc_prices[-1] * (1 + ret))
        
        # ETH prices (correlated)
        eth_returns = []
        for btc_ret in btc_returns:
            corr_part = correlation * btc_ret
            noise_part = np.sqrt(1 - correlation**2) * np.random.normal(0, eth_vol)
            eth_returns.append(corr_part + noise_part)
        
        eth_prices = [3800]
        for ret in eth_returns[1:]:
            eth_prices.append(eth_prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'btc_price': btc_prices,
            'eth_price': eth_prices
        })

    def calculate_market_regime(self, btc_returns: list, eth_returns: list) -> dict:
        """Calculate current market regime metrics"""
        if len(btc_returns) < 20:
            return {'volatility': 0.02, 'correlation': 0.75, 'regime': 'normal'}
        
        btc_vol = np.std(btc_returns[-20:]) * np.sqrt(24 * 120)  # Annualized
        eth_vol = np.std(eth_returns[-20:]) * np.sqrt(24 * 120)
        correlation = np.corrcoef(btc_returns[-20:], eth_returns[-20:])[0, 1]
        
        avg_vol = (btc_vol + eth_vol) / 2
        
        if avg_vol > self.volatility_threshold:
            regime = 'high_vol'
        elif correlation < self.correlation_min:
            regime = 'low_corr'
        else:
            regime = 'normal'
        
        return {
            'volatility': avg_vol,
            'correlation': correlation,
            'regime': regime
        }

    def calculate_enhanced_beta(self, btc_returns: list, eth_returns: list) -> float:
        """Enhanced beta calculation with regime awareness"""
        if len(btc_returns) < 20:
            return 1.0
        
        # Use longer lookback for more stable beta
        lookback = min(len(btc_returns), self.lookback)
        btc_arr = np.array(btc_returns[-lookback:])
        eth_arr = np.array(eth_returns[-lookback:])
        
        # Robust beta calculation
        try:
            covariance = np.cov(eth_arr, btc_arr)[0, 1]
            btc_variance = np.var(btc_arr)
            
            if btc_variance > 0:
                beta = covariance / btc_variance
                # Clamp beta to reasonable range
                return max(0.3, min(2.0, beta))
            else:
                return 1.0
        except:
            return 1.0

    def calculate_adaptive_z_score(self, btc_prices: list, eth_prices: list, market_regime: dict) -> float:
        """Calculate Z-score with regime-based adjustments"""
        if len(btc_prices) < self.lookback:
            return 0
        
        # Use full lookback period
        lookback_btc = btc_prices[-self.lookback:]
        lookback_eth = eth_prices[-self.lookback:]
        
        # Calculate log price ratio (spread)
        spreads = [np.log(lookback_btc[i] / lookback_eth[i]) for i in range(len(lookback_btc))]
        
        if len(spreads) < 2:
            return 0
        
        current_spread = spreads[-1]
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        if std_spread == 0:
            return 0
        
        base_z_score = (current_spread - mean_spread) / std_spread
        
        # Adjust for market regime
        if market_regime['regime'] == 'high_vol':
            # Scale down Z-score in high volatility (less reliable)
            return base_z_score * 0.7
        elif market_regime['regime'] == 'low_corr':
            # Scale down Z-score when correlation is low
            return base_z_score * 0.5
        else:
            return base_z_score

    def should_trade(self, market_regime: dict, current_time: datetime) -> bool:
        """Enhanced trading filter"""
        # Don't trade in adverse market conditions
        if market_regime['regime'] in ['high_vol', 'low_corr']:
            return False
        
        # Daily trade limit
        date_key = current_time.strftime('%Y-%m-%d')
        daily_count = self.daily_trades.get(date_key, 0)
        if daily_count >= self.daily_trade_limit:
            return False
        
        return True

    def calculate_position_size(self, capital: float, market_regime: dict, beta: float) -> tuple:
        """Enhanced position sizing based on market conditions"""
        base_size = capital * self.position_pct
        
        # Reduce size in uncertain conditions
        if market_regime['regime'] == 'high_vol':
            base_size *= 0.5
        elif market_regime['correlation'] < 0.7:
            base_size *= 0.7
        
        btc_dollar_size = base_size
        eth_dollar_size = base_size * abs(beta)
        
        return btc_dollar_size, eth_dollar_size

    def run_backtest(self, data: pd.DataFrame) -> dict:
        """Run improved backtesting simulation"""
        self.logger.info("Starting improved backtest...")
        
        initial_capital = 100000
        capital = initial_capital
        
        btc_history = []
        eth_history = []
        btc_returns = []
        eth_returns = []
        
        position = None
        entry_time = None
        entry_capital = 0
        entry_btc_price = 0
        entry_eth_price = 0
        
        results = []
        self.trades = []
        self.daily_trades = {}
        
        for idx, row in data.iterrows():
            btc_price = row['btc_price']
            eth_price = row['eth_price']
            timestamp = row['timestamp']
            
            # Update price histories
            btc_history.append(btc_price)
            eth_history.append(eth_price)
            
            # Calculate returns
            if len(btc_history) >= 2:
                btc_ret = (btc_price - btc_history[-2]) / btc_history[-2]
                eth_ret = (eth_price - eth_history[-2]) / eth_history[-2]
                btc_returns.append(btc_ret)
                eth_returns.append(eth_ret)
            
            # Keep lookback period
            if len(btc_history) > self.lookback:
                btc_history = btc_history[-self.lookback:]
                eth_history = eth_history[-self.lookback:]
                btc_returns = btc_returns[-self.lookback:]
                eth_returns = eth_returns[-self.lookback:]
            
            # Need minimum data
            if len(btc_history) < self.lookback:
                continue
            
            # Calculate market metrics
            market_regime = self.calculate_market_regime(btc_returns, eth_returns)
            beta = self.calculate_enhanced_beta(btc_returns, eth_returns)
            z_score = self.calculate_adaptive_z_score(btc_history, eth_history, market_regime)
            
            # Current portfolio value
            if position is None:
                portfolio_value = capital
            else:
                # Calculate current P&L
                if position == 'long_btc':
                    btc_pnl = (btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = -(eth_price - entry_eth_price) / entry_eth_price * beta
                    total_pnl = (btc_pnl + eth_pnl) * entry_capital
                    portfolio_value = capital + total_pnl
                else:  # short_btc
                    btc_pnl = -(btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = (eth_price - entry_eth_price) / entry_eth_price * beta
                    total_pnl = (btc_pnl + eth_pnl) * entry_capital
                    portfolio_value = capital + total_pnl
            
            # Trading logic with enhanced filters
            if position is None and abs(z_score) > self.z_entry and self.should_trade(market_regime, timestamp):
                # Enter position
                btc_dollar_size, eth_dollar_size = self.calculate_position_size(capital, market_regime, beta)
                entry_capital = btc_dollar_size
                
                trading_cost = (btc_dollar_size + eth_dollar_size) * self.fee
                capital -= trading_cost
                
                entry_btc_price = btc_price
                entry_eth_price = eth_price
                entry_time = timestamp
                
                # Update daily trade count
                date_key = timestamp.strftime('%Y-%m-%d')
                self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
                
                if z_score > self.z_entry:
                    position = 'short_btc'
                    action = "SHORT BTC, LONG ETH"
                else:
                    position = 'long_btc'
                    action = "LONG BTC, SHORT ETH"
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'ENTER',
                    'position': action,
                    'z_score': z_score,
                    'beta': beta,
                    'capital': capital,
                    'cost': trading_cost,
                    'market_regime': market_regime['regime'],
                    'volatility': market_regime['volatility'],
                    'correlation': market_regime['correlation']
                })
                
            elif position is not None:
                # Check exit conditions
                minutes_held = (timestamp - entry_time).total_seconds() / 60
                should_exit = False
                exit_reason = ""
                
                # Normal exit: Z-score mean reversion
                if abs(z_score) < self.z_exit:
                    should_exit = True
                    exit_reason = "Z-score mean reversion"
                
                # Time-based exit
                elif minutes_held > self.max_holding_minutes:
                    should_exit = True
                    exit_reason = "Maximum holding time"
                
                # Stop loss
                elif position == 'long_btc':
                    btc_pnl = (btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = -(eth_price - entry_eth_price) / entry_eth_price * beta
                    total_return = btc_pnl + eth_pnl
                    if total_return < -self.stop_loss_pct:
                        should_exit = True
                        exit_reason = "Stop loss"
                else:  # short_btc
                    btc_pnl = -(btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = (eth_price - entry_eth_price) / entry_eth_price * beta
                    total_return = btc_pnl + eth_pnl
                    if total_return < -self.stop_loss_pct:
                        should_exit = True
                        exit_reason = "Stop loss"
                
                # Minimum holding time check
                if should_exit and minutes_held < self.min_holding_minutes:
                    should_exit = False
                
                if should_exit:
                    # Exit position
                    if position == 'long_btc':
                        btc_pnl = (btc_price - entry_btc_price) / entry_btc_price
                        eth_pnl = -(eth_price - entry_eth_price) / entry_eth_price * beta
                        total_pnl = (btc_pnl + eth_pnl) * entry_capital
                    else:
                        btc_pnl = -(btc_price - entry_btc_price) / entry_btc_price
                        eth_pnl = (eth_price - entry_eth_price) / entry_eth_price * beta
                        total_pnl = (btc_pnl + eth_pnl) * entry_capital
                    
                    trading_cost = entry_capital * self.fee * 2
                    capital += total_pnl - trading_cost
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'EXIT',
                        'position': position,
                        'z_score': z_score,
                        'beta': beta,
                        'pnl': total_pnl,
                        'capital': capital,
                        'cost': trading_cost,
                        'holding_minutes': minutes_held,
                        'exit_reason': exit_reason,
                        'market_regime': market_regime['regime']
                    })
                    
                    position = None
                    entry_capital = 0
            
            # Record portfolio state
            results.append({
                'timestamp': timestamp,
                'btc_price': btc_price,
                'eth_price': eth_price,
                'z_score': z_score,
                'beta': beta,
                'position': position,
                'portfolio_value': portfolio_value,
                'cash': capital,
                'market_regime': market_regime['regime'],
                'volatility': market_regime['volatility'],
                'correlation': market_regime['correlation']
            })
        
        # Calculate final metrics
        final_value = results[-1]['portfolio_value']
        total_return = (final_value - initial_capital) / initial_capital
        
        num_trades = len([t for t in self.trades if t['action'] == 'ENTER'])
        exit_trades = [t for t in self.trades if t['action'] == 'EXIT']
        winning_trades = len([t for t in exit_trades if t['pnl'] > 0])
        win_rate = winning_trades / max(len(exit_trades), 1)
        
        # Advanced metrics
        if exit_trades:
            avg_holding_time = np.mean([t['holding_minutes'] for t in exit_trades])
            profit_factor = sum([t['pnl'] for t in exit_trades if t['pnl'] > 0]) / abs(sum([t['pnl'] for t in exit_trades if t['pnl'] < 0])) if any(t['pnl'] < 0 for t in exit_trades) else float('inf')
        else:
            avg_holding_time = 0
            profit_factor = 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': final_value - initial_capital,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_holding_time': avg_holding_time,
            'profit_factor': profit_factor,
            'trades': self.trades,
            'portfolio_data': pd.DataFrame(results)
        }

    def print_enhanced_results(self, results: dict):
        """Print comprehensive results"""
        print("\n" + "="*60)
        print("🚀 IMPROVED STATISTICAL ARBITRAGE BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Capital:     ${results['initial_capital']:,.2f}")
        print(f"Final Value:         ${results['final_value']:,.2f}")
        print(f"Total Return:        {results['total_return']*100:.2f}%")
        print(f"Total P&L:           ${results['total_pnl']:,.2f}")
        print()
        print(f"Number of Trades:    {results['num_trades']}")
        print(f"Win Rate:            {results['win_rate']*100:.1f}%")
        print(f"Avg Holding Time:    {results['avg_holding_time']:.0f} minutes")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print("="*60)
        
        # Enhanced trade analysis
        exit_trades = [t for t in self.trades if t['action'] == 'EXIT']
        if exit_trades:
            trade_pnls = [t['pnl'] for t in exit_trades]
            print(f"\n📊 ENHANCED TRADE ANALYSIS:")
            print(f"Total Completed Trades: {len(exit_trades)}")
            print(f"Winning Trades:         {len([p for p in trade_pnls if p > 0])}")
            print(f"Losing Trades:          {len([p for p in trade_pnls if p < 0])}")
            print(f"Average Trade P&L:      ${np.mean(trade_pnls):,.2f}")
            print(f"Best Trade:             ${max(trade_pnls):,.2f}")
            print(f"Worst Trade:            ${min(trade_pnls):,.2f}")
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in exit_trades:
                reason = trade.get('exit_reason', 'Unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            print(f"\n📈 EXIT REASON BREAKDOWN:")
            for reason, count in exit_reasons.items():
                percentage = count / len(exit_trades) * 100
                print(f"  {reason}: {count} trades ({percentage:.1f}%)")
        
        # Daily trading analysis
        daily_summary = {}
        for trade in [t for t in self.trades if t['action'] == 'ENTER']:
            date = trade['timestamp'].strftime('%Y-%m-%d')
            daily_summary[date] = daily_summary.get(date, 0) + 1
        
        print(f"\n📅 DAILY TRADING ACTIVITY:")
        for date, count in daily_summary.items():
            print(f"  {date}: {count} trades")


def main():
    """Run improved backtest"""
    print("🚀 Improved Statistical Arbitrage Backtest")
    print("="*50)
    
    backtester = ImprovedBacktester()
    
    # Test with different market regimes
    regimes = ["normal", "low", "high"]
    
    for regime in regimes:
        print(f"\n📊 Testing {regime.upper()} volatility regime...")
        
        # Generate data for this regime
        data = backtester.generate_sample_data(days=7, volatility_regime=regime)
        
        # Run backtest
        results = backtester.run_backtest(data)
        
        # Show results
        print(f"\n{regime.upper()} REGIME RESULTS:")
        print(f"Return: {results['total_return']*100:.2f}%, Trades: {results['num_trades']}, Win Rate: {results['win_rate']*100:.1f}%")
    
    # Run detailed analysis on normal regime
    print(f"\n" + "="*50)
    print("🎯 DETAILED ANALYSIS - NORMAL REGIME")
    print("="*50)
    
    data = backtester.generate_sample_data(days=7, volatility_regime="normal")
    results = backtester.run_backtest(data)
    backtester.print_enhanced_results(results)
    
    return results


if __name__ == "__main__":
    results = main()