#!/usr/bin/env python3
"""
Simple Statistical Arbitrage Backtester
A streamlined version for testing the BTC/ETH pairs trading strategy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

class SimpleBacktester:
    def __init__(self, config_path: str = "config.json"):
        """Initialize simple backtester"""
        self.config = self.load_config(config_path)
        
        # Strategy parameters
        self.position_pct = self.config.get("position_size", 0.25)  # 25% of capital per side
        self.z_entry = self.config.get("z_score_entry", 1.5)
        self.z_exit = self.config.get("z_score_exit", 0.3)
        self.lookback = self.config.get("lookback_period", 20)
        
        # Trading costs
        self.fee = 0.0005  # 0.05% per trade
        
        # Results tracking
        self.trades = []
        self.portfolio_values = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def generate_sample_data(self, days: int = 7) -> pd.DataFrame:
        """Generate realistic BTC/ETH price data"""
        self.logger.info(f"Generating {days} days of sample data...")
        
        # 30-second intervals
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days * 24 * 120,  # 120 periods per hour (30-sec intervals)
            freq='30S'
        )
        
        n = len(timestamps)
        np.random.seed(42)
        
        # BTC prices with realistic volatility
        btc_returns = np.random.normal(0, 0.02, n)  # 2% volatility
        btc_prices = [118000]
        for ret in btc_returns[1:]:
            btc_prices.append(btc_prices[-1] * (1 + ret))
        
        # ETH prices correlated with BTC
        correlation = 0.8
        eth_returns = []
        for btc_ret in btc_returns:
            corr_part = correlation * btc_ret
            noise_part = np.sqrt(1 - correlation**2) * np.random.normal(0, 0.015)
            eth_returns.append(corr_part + noise_part)
        
        eth_prices = [3800]
        for ret in eth_returns[1:]:
            eth_prices.append(eth_prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'btc_price': btc_prices,
            'eth_price': eth_prices
        })

    def calculate_beta(self, btc_returns: list, eth_returns: list) -> float:
        """Calculate beta (ETH relative to BTC)"""
        if len(btc_returns) < 10:
            return 1.0
        
        btc_arr = np.array(btc_returns)
        eth_arr = np.array(eth_returns)
        
        covariance = np.cov(eth_arr, btc_arr)[0, 1]
        btc_variance = np.var(btc_arr)
        
        return covariance / btc_variance if btc_variance > 0 else 1.0

    def calculate_z_score(self, btc_prices: list, eth_prices: list) -> float:
        """Calculate Z-score of price spread"""
        if len(btc_prices) < 2:
            return 0
        
        # Log price ratio (spread)
        spreads = [np.log(btc_prices[i] / eth_prices[i]) for i in range(len(btc_prices))]
        
        if len(spreads) < 2:
            return 0
        
        current_spread = spreads[-1]
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        return (current_spread - mean_spread) / std_spread if std_spread > 0 else 0

    def run_backtest(self, data: pd.DataFrame) -> dict:
        """Run the backtesting simulation"""
        self.logger.info("Starting backtest...")
        
        initial_capital = 100000
        capital = initial_capital
        
        btc_history = []
        eth_history = []
        btc_returns = []
        eth_returns = []
        
        position = None  # None, 'long_btc', 'short_btc'
        entry_capital = 0
        entry_btc_price = 0
        entry_eth_price = 0
        
        results = []
        
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
            
            # Keep only lookback period
            if len(btc_history) > self.lookback:
                btc_history = btc_history[-self.lookback:]
                eth_history = eth_history[-self.lookback:]
                btc_returns = btc_returns[-self.lookback:]
                eth_returns = eth_returns[-self.lookback:]
            
            # Need minimum data
            if len(btc_history) < self.lookback:
                continue
            
            # Calculate current metrics
            z_score = self.calculate_z_score(btc_history, eth_history)
            beta = self.calculate_beta(btc_returns, eth_returns)
            
            # Current portfolio value
            if position is None:
                portfolio_value = capital
            else:
                # Calculate current P&L
                if position == 'long_btc':
                    # Long BTC, Short ETH
                    btc_pnl = (btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = -(eth_price - entry_eth_price) / entry_eth_price * beta
                    total_pnl = (btc_pnl + eth_pnl) * entry_capital
                    portfolio_value = capital + total_pnl
                else:  # short_btc
                    # Short BTC, Long ETH  
                    btc_pnl = -(btc_price - entry_btc_price) / entry_btc_price
                    eth_pnl = (eth_price - entry_eth_price) / entry_eth_price * beta
                    total_pnl = (btc_pnl + eth_pnl) * entry_capital
                    portfolio_value = capital + total_pnl
            
            # Trading logic
            if position is None and abs(z_score) > self.z_entry:
                # Enter position
                entry_capital = capital * self.position_pct
                trading_cost = entry_capital * self.fee * 2  # Buy and sell
                capital -= trading_cost
                
                entry_btc_price = btc_price
                entry_eth_price = eth_price
                
                if z_score > self.z_entry:
                    position = 'short_btc'  # BTC expensive vs ETH
                    action = "SHORT BTC, LONG ETH"
                else:
                    position = 'long_btc'   # ETH expensive vs BTC  
                    action = "LONG BTC, SHORT ETH"
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'ENTER',
                    'position': action,
                    'z_score': z_score,
                    'beta': beta,
                    'capital': capital,
                    'cost': trading_cost
                })
                
            elif position is not None and abs(z_score) < self.z_exit:
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
                    'cost': trading_cost
                })
                
                position = None
                entry_capital = 0
            
            # Record portfolio value
            results.append({
                'timestamp': timestamp,
                'btc_price': btc_price,
                'eth_price': eth_price,
                'z_score': z_score,
                'beta': beta,
                'position': position,
                'portfolio_value': portfolio_value,
                'cash': capital
            })
        
        # Calculate final metrics
        final_value = results[-1]['portfolio_value']
        total_return = (final_value - initial_capital) / initial_capital
        
        num_trades = len([t for t in self.trades if t['action'] == 'ENTER'])
        winning_trades = len([t for t in self.trades if t['action'] == 'EXIT' and t['pnl'] > 0])
        win_rate = winning_trades / max(len([t for t in self.trades if t['action'] == 'EXIT']), 1)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': final_value - initial_capital,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'trades': self.trades,
            'portfolio_data': pd.DataFrame(results)
        }

    def print_results(self, results: dict):
        """Print backtest results"""
        print("\n" + "="*50)
        print("STATISTICAL ARBITRAGE BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Capital:  ${results['initial_capital']:,.2f}")
        print(f"Final Value:      ${results['final_value']:,.2f}")
        print(f"Total Return:     {results['total_return']*100:.2f}%")
        print(f"Total P&L:        ${results['total_pnl']:,.2f}")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Win Rate:         {results['win_rate']*100:.1f}%")
        print("="*50)
        
        # Show recent trades
        print("\nRecent Trades:")
        for trade in self.trades[-10:]:
            action = trade['action']
            if action == 'ENTER':
                print(f"{trade['timestamp']:%Y-%m-%d %H:%M} | ENTER {trade['position']} | Z-score: {trade['z_score']:.2f} | Beta: {trade['beta']:.2f}")
            else:
                print(f"{trade['timestamp']:%Y-%m-%d %H:%M} | EXIT | P&L: ${trade['pnl']:,.2f} | Capital: ${trade['capital']:,.2f}")

    def plot_pnl_chart(self, results: dict):
        """Plot comprehensive P&L and trading analysis charts"""
        portfolio_df = results['portfolio_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Arbitrage Backtest Analysis', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
                linewidth=2, color='navy', label='Portfolio Value')
        ax1.axhline(y=results['initial_capital'], color='red', linestyle='--', 
                   alpha=0.7, label='Initial Capital')
        
        # Highlight trading periods
        current_position = None
        entry_time = None
        
        for idx, row in portfolio_df.iterrows():
            if row['position'] != current_position:
                if current_position is not None:
                    # End of previous position
                    ax1.axvspan(entry_time, row['timestamp'], alpha=0.1, 
                              color='green' if current_position == 'long_btc' else 'red')
                
                if row['position'] is not None:
                    # Start of new position
                    entry_time = row['timestamp']
                
                current_position = row['position']
        
        ax1.set_title('Portfolio Value Over Time', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Cumulative P&L
        ax2 = axes[0, 1]
        cumulative_pnl = portfolio_df['portfolio_value'] - results['initial_capital']
        ax2.plot(portfolio_df['timestamp'], cumulative_pnl, 
                linewidth=2, color='darkgreen', label='Cumulative P&L')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(portfolio_df['timestamp'], cumulative_pnl, 0, 
                        where=(cumulative_pnl >= 0), color='green', alpha=0.3, label='Profit')
        ax2.fill_between(portfolio_df['timestamp'], cumulative_pnl, 0, 
                        where=(cumulative_pnl < 0), color='red', alpha=0.3, label='Loss')
        
        ax2.set_title('Cumulative P&L', fontweight='bold')
        ax2.set_ylabel('P&L ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 3. Z-Score and Trade Signals
        ax3 = axes[1, 0]
        ax3.plot(portfolio_df['timestamp'], portfolio_df['z_score'], 
                linewidth=1, color='blue', label='Z-Score')
        
        # Entry/exit thresholds
        ax3.axhline(y=self.z_entry, color='red', linestyle='--', alpha=0.7, label=f'Entry (+{self.z_entry})')
        ax3.axhline(y=-self.z_entry, color='red', linestyle='--', alpha=0.7, label=f'Entry (-{self.z_entry})')
        ax3.axhline(y=self.z_exit, color='green', linestyle='--', alpha=0.7, label=f'Exit (+{self.z_exit})')
        ax3.axhline(y=-self.z_exit, color='green', linestyle='--', alpha=0.7, label=f'Exit (-{self.z_exit})')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark trade entries
        for trade in self.trades:
            if trade['action'] == 'ENTER':
                color = 'red' if 'SHORT BTC' in trade['position'] else 'blue'
                ax3.scatter(trade['timestamp'], trade['z_score'], 
                          color=color, s=50, alpha=0.8, zorder=5)
        
        ax3.set_title('Z-Score and Trade Signals', fontweight='bold')
        ax3.set_ylabel('Z-Score')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # 4. Beta Over Time
        ax4 = axes[1, 1]
        ax4.plot(portfolio_df['timestamp'], portfolio_df['beta'], 
                linewidth=2, color='purple', label='Beta (ETH/BTC)')
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Beta = 1.0')
        
        ax4.set_title('Beta (ETH/BTC) Over Time', fontweight='bold')
        ax4.set_ylabel('Beta')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('backtest_pnl_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional trade-by-trade P&L chart
        self.plot_trade_pnl()

    def plot_trade_pnl(self):
        """Plot individual trade P&L"""
        exit_trades = [t for t in self.trades if t['action'] == 'EXIT']
        
        if not exit_trades:
            print("No completed trades to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Trade P&L over time
        trade_times = [t['timestamp'] for t in exit_trades]
        trade_pnls = [t['pnl'] for t in exit_trades]
        colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
        
        ax1.bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Individual Trade P&L', fontweight='bold')
        ax1.set_ylabel('P&L ($)')
        ax1.set_xlabel('Trade Number')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Cumulative trade P&L
        cumulative_trade_pnl = np.cumsum(trade_pnls)
        ax2.plot(range(len(cumulative_trade_pnl)), cumulative_trade_pnl, 
                linewidth=2, color='navy', marker='o', markersize=3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(range(len(cumulative_trade_pnl)), cumulative_trade_pnl, 0, 
                        where=np.array(cumulative_trade_pnl) >= 0, color='green', alpha=0.3)
        ax2.fill_between(range(len(cumulative_trade_pnl)), cumulative_trade_pnl, 0, 
                        where=np.array(cumulative_trade_pnl) < 0, color='red', alpha=0.3)
        
        ax2.set_title('Cumulative Trade P&L', fontweight='bold')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.set_xlabel('Trade Number')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig('individual_trade_pnl.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print trade statistics
        print(f"\n📊 TRADE ANALYSIS:")
        print(f"Total Trades: {len(exit_trades)}")
        print(f"Winning Trades: {len([p for p in trade_pnls if p > 0])}")
        print(f"Losing Trades: {len([p for p in trade_pnls if p < 0])}")
        print(f"Average Trade P&L: ${np.mean(trade_pnls):,.2f}")
        print(f"Best Trade: ${max(trade_pnls):,.2f}")
        print(f"Worst Trade: ${min(trade_pnls):,.2f}")
        print(f"Total Trading P&L: ${sum(trade_pnls):,.2f}")


def main():
    """Run simple backtest"""
    print("Simple Statistical Arbitrage Backtest")
    print("="*40)
    
    backtester = SimpleBacktester()
    
    # Generate sample data
    data = backtester.generate_sample_data(days=3)  # 3 days for quick test
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    # Show results
    backtester.print_results(results)
    
    # Generate P&L charts
    print("\n📈 Generating P&L charts...")
    backtester.plot_pnl_chart(results)
    
    return results


if __name__ == "__main__":
    results = main()