#!/usr/bin/env python3
"""
Statistical Arbitrage Backtesting Engine
Backtests the BTC/ETH pairs trading strategy with beta hedging
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StatArbBacktester:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the backtesting engine
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        
        # Strategy parameters
        self.position_size = self.config.get("position_size", 0.25)
        self.z_score_entry = self.config.get("z_score_entry", 1.5)
        self.z_score_exit = self.config.get("z_score_exit", 0.3)
        self.lookback_period = self.config.get("lookback_period", 20)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.02)
        
        # Backtesting parameters
        self.trading_fee = 0.0005  # 0.05% per trade (Hyperliquid typical)
        self.slippage = 0.0001     # 0.01% slippage
        
        # Data storage
        self.price_data = None
        self.trades = []
        self.positions = {"BTC": 0, "ETH": 0}
        self.portfolio_value = []
        self.drawdowns = []
        
        # Beta hedging
        self.beta_history = []
        self.min_beta_periods = 10
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}

    def load_historical_data(self, data_source: str = "sample") -> pd.DataFrame:
        """
        Load historical price data for BTC and ETH
        
        Args:
            data_source: "sample" for generated data, "file" for CSV, "api" for live data
            
        Returns:
            DataFrame with BTC and ETH prices
        """
        if data_source == "sample":
            return self.generate_sample_data()
        elif data_source == "file":
            return self.load_from_csv()
        elif data_source == "api":
            return self.fetch_live_data()
        else:
            raise ValueError("Invalid data source")

    def generate_sample_data(self, days: int = 30, interval_minutes: int = 1) -> pd.DataFrame:
        """
        Generate realistic sample price data for backtesting
        
        Args:
            days: Number of days of data
            interval_minutes: Data interval in minutes
            
        Returns:
            DataFrame with timestamp, BTC_price, ETH_price
        """
        self.logger.info(f"Generating {days} days of sample data...")
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=days)
        timestamps = pd.date_range(
            start=start_time, 
            periods=days * 24 * 60 // interval_minutes,
            freq=f'{interval_minutes}min'
        )
        
        n_points = len(timestamps)
        
        # Generate correlated price movements
        np.random.seed(42)  # For reproducible results
        
        # BTC price simulation (random walk with drift)
        btc_returns = np.random.normal(0.0001, 0.02, n_points)  # Higher volatility
        btc_prices = [118000]  # Starting price
        
        for ret in btc_returns[1:]:
            btc_prices.append(btc_prices[-1] * (1 + ret))
        
        # ETH price simulation (correlated with BTC)
        correlation = 0.8
        eth_returns = []
        
        for i, btc_ret in enumerate(btc_returns):
            # Correlated return with some independent noise
            corr_component = correlation * btc_ret
            noise_component = np.sqrt(1 - correlation**2) * np.random.normal(0, 0.015)
            eth_ret = corr_component + noise_component
            eth_returns.append(eth_ret)
        
        eth_prices = [3800]  # Starting price
        for ret in eth_returns[1:]:
            eth_prices.append(eth_prices[-1] * (1 + ret))
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'BTC_price': btc_prices,
            'ETH_price': eth_prices
        })
        
        self.logger.info(f"Generated {len(data)} data points")
        return data

    def load_from_csv(self, filepath: str = "historical_data.csv") -> pd.DataFrame:
        """Load historical data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            self.logger.info(f"Loaded {len(data)} data points from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return self.generate_sample_data()

    def calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate price returns"""
        if len(prices) < 2:
            return []
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

    def calculate_beta(self, btc_returns: List[float], eth_returns: List[float]) -> float:
        """
        Calculate beta (ETH relative to BTC) for hedging
        
        Args:
            btc_returns: BTC price returns
            eth_returns: ETH price returns
            
        Returns:
            Beta coefficient
        """
        if len(btc_returns) < self.min_beta_periods or len(eth_returns) < self.min_beta_periods:
            return 1.0
        
        btc_array = np.array(btc_returns)
        eth_array = np.array(eth_returns)
        
        covariance = np.cov(eth_array, btc_array)[0, 1]
        btc_variance = np.var(btc_array)
        
        if btc_variance > 0:
            return covariance / btc_variance
        return 1.0

    def calculate_spread_statistics(self, btc_prices: List[float], eth_prices: List[float]) -> Tuple[float, float, float]:
        """
        Calculate spread statistics for Z-score computation
        
        Returns:
            (current_spread, mean_spread, std_spread)
        """
        if len(btc_prices) < 2 or len(eth_prices) < 2:
            return 0, 0, 1
        
        # Calculate log price ratio (spread)
        spreads = [np.log(btc_prices[i] / eth_prices[i]) for i in range(len(btc_prices))]
        
        current_spread = spreads[-1]
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        if std_spread == 0:
            std_spread = 1
        
        return current_spread, mean_spread, std_spread

    def calculate_z_score(self, btc_prices: List[float], eth_prices: List[float]) -> float:
        """Calculate Z-score of current spread"""
        current_spread, mean_spread, std_spread = self.calculate_spread_statistics(btc_prices, eth_prices)
        return (current_spread - mean_spread) / std_spread

    def get_position_sizes(self, base_size: float, beta: float) -> Tuple[float, float]:
        """
        Calculate beta-hedged position sizes
        
        Args:
            base_size: Base position size for BTC
            beta: Current beta coefficient
            
        Returns:
            (btc_size, eth_size)
        """
        btc_size = base_size
        eth_size = base_size * abs(beta)
        return btc_size, eth_size

    def execute_trade(self, action: str, btc_price: float, eth_price: float, 
                     btc_size: float, eth_size: float, timestamp: datetime, 
                     z_score: float, beta: float) -> Dict:
        """
        Execute a trade and record it
        
        Args:
            action: "enter_long_btc", "enter_short_btc", "exit"
            btc_price, eth_price: Current prices
            btc_size, eth_size: Position sizes
            timestamp: Trade timestamp
            z_score: Current Z-score
            beta: Current beta
            
        Returns:
            Trade record dictionary
        """
        # Calculate trading costs
        btc_cost = btc_price * btc_size * (self.trading_fee + self.slippage)
        eth_cost = eth_price * eth_size * (self.trading_fee + self.slippage)
        total_cost = btc_cost + eth_cost
        
        trade = {
            'timestamp': timestamp,
            'action': action,
            'btc_price': btc_price,
            'eth_price': eth_price,
            'btc_size': btc_size,
            'eth_size': eth_size,
            'z_score': z_score,
            'beta': beta,
            'trading_cost': total_cost,
            'btc_position': self.positions["BTC"],
            'eth_position': self.positions["ETH"]
        }
        
        self.trades.append(trade)
        return trade

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run the complete backtesting simulation
        
        Args:
            data: Historical price data
            
        Returns:
            Backtest results dictionary
        """
        self.logger.info("Starting backtest...")
        
        # Initialize tracking variables
        btc_price_history = []
        eth_price_history = []
        btc_returns_history = []
        eth_returns_history = []
        
        initial_capital = 100000  # $100k starting capital
        current_capital = initial_capital
        
        is_trading = False
        entry_z_score = 0
        self.entry_prices = {'BTC': 0, 'ETH': 0}
        
        # Main backtesting loop
        for idx, row in data.iterrows():
            timestamp = row['timestamp']
            btc_price = row['BTC_price']
            eth_price = row['ETH_price']
            
            # Update price histories
            btc_price_history.append(btc_price)
            eth_price_history.append(eth_price)
            
            # Calculate returns
            if len(btc_price_history) >= 2:
                btc_ret = (btc_price - btc_price_history[-2]) / btc_price_history[-2]
                eth_ret = (eth_price - eth_price_history[-2]) / eth_price_history[-2]
                btc_returns_history.append(btc_ret)
                eth_returns_history.append(eth_ret)
            
            # Keep only lookback period
            if len(btc_price_history) > self.lookback_period:
                btc_price_history = btc_price_history[-self.lookback_period:]
                eth_price_history = eth_price_history[-self.lookback_period:]
                btc_returns_history = btc_returns_history[-self.lookback_period:]
                eth_returns_history = eth_returns_history[-self.lookback_period:]
            
            # Need minimum data points for trading
            if len(btc_price_history) < self.lookback_period:
                continue
            
            # Calculate current metrics
            z_score = self.calculate_z_score(btc_price_history, eth_price_history)
            beta = self.calculate_beta(btc_returns_history, eth_returns_history)
            self.beta_history.append(beta)
            
            # Calculate dollar position sizes
            btc_dollar_size = current_capital * self.position_size
            eth_dollar_size = btc_dollar_size * abs(beta)
            
            # Convert to contract sizes
            btc_contracts = btc_dollar_size / btc_price
            eth_contracts = eth_dollar_size / eth_price
            
            # Trading logic
            if not is_trading and abs(z_score) > self.z_score_entry:
                # Enter trade
                if z_score > self.z_score_entry:
                    # Short BTC, Long ETH
                    action = "enter_short_btc"
                    self.positions["BTC"] = -btc_contracts
                    self.positions["ETH"] = eth_contracts
                elif z_score < -self.z_score_entry:
                    # Long BTC, Short ETH
                    action = "enter_long_btc" 
                    self.positions["BTC"] = btc_contracts
                    self.positions["ETH"] = -eth_contracts
                
                # Store entry prices
                self.entry_prices = {'BTC': btc_price, 'ETH': eth_price}
                
                trade = self.execute_trade(action, btc_price, eth_price, btc_contracts, 
                                         eth_contracts, timestamp, z_score, beta)
                current_capital -= trade['trading_cost']
                is_trading = True
                entry_z_score = z_score
                
            elif is_trading and (abs(z_score) < self.z_score_exit or 
                               self.check_stop_loss(btc_price_history, eth_price_history, entry_z_score)):
                # Exit trade
                trade = self.execute_trade("exit", btc_price, eth_price, 
                                         abs(self.positions["BTC"]), abs(self.positions["ETH"]), 
                                         timestamp, z_score, beta)
                
                # Calculate P&L
                pnl = self.calculate_pnl(btc_price, eth_price)
                current_capital += pnl - trade['trading_cost']
                
                # Reset positions
                self.positions["BTC"] = 0
                self.positions["ETH"] = 0
                is_trading = False
            
            # Track portfolio value
            portfolio_val = current_capital
            if is_trading:
                portfolio_val += self.calculate_unrealized_pnl(btc_price, eth_price)
            
            self.portfolio_value.append({
                'timestamp': timestamp,
                'value': portfolio_val,
                'capital': current_capital,
                'is_trading': is_trading,
                'z_score': z_score,
                'beta': beta
            })
        
        # Calculate final results
        results = self.calculate_performance_metrics(initial_capital)
        self.logger.info("Backtest completed!")
        
        return results

    def check_stop_loss(self, btc_prices: List[float], eth_prices: List[float], entry_z_score: float) -> bool:
        """Check if stop loss should be triggered"""
        if not btc_prices or not eth_prices:
            return False
        
        current_z_score = self.calculate_z_score(btc_prices, eth_prices)
        
        # Stop loss if Z-score moves significantly against the trade
        if entry_z_score > 0 and current_z_score > entry_z_score * (1 + self.stop_loss_pct):
            return True
        elif entry_z_score < 0 and current_z_score < entry_z_score * (1 + self.stop_loss_pct):
            return True
        
        return False

    def calculate_pnl(self, current_btc_price: float, current_eth_price: float, 
                      entry_btc_price: float = None, entry_eth_price: float = None) -> float:
        """Calculate realized P&L for current positions"""
        if not hasattr(self, 'entry_prices') or not self.entry_prices:
            return 0
        
        # Calculate P&L based on price changes since entry
        btc_pnl = self.positions["BTC"] * (current_btc_price - self.entry_prices['BTC'])
        eth_pnl = self.positions["ETH"] * (current_eth_price - self.entry_prices['ETH'])
        return btc_pnl + eth_pnl

    def calculate_unrealized_pnl(self, current_btc_price: float, current_eth_price: float) -> float:
        """Calculate unrealized P&L for current positions"""
        return self.calculate_pnl(current_btc_price, current_eth_price)

    def calculate_performance_metrics(self, initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_value:
            return {}
        
        # Convert to DataFrame for easier analysis
        portfolio_df = pd.DataFrame(self.portfolio_value)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
        
        # Risk metrics
        volatility = portfolio_df['daily_return'].std() * np.sqrt(365 * 24 * 60)  # Annualized
        sharpe_ratio = (total_return * 365) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        running_max = portfolio_df['value'].expanding().max()
        drawdown = (portfolio_df['value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        num_trades = len(trades_df)
        win_trades = 0
        total_pnl = final_value - initial_capital
        
        if num_trades > 0:
            # Calculate individual trade P&L (simplified)
            win_trades = num_trades // 2  # Placeholder
        
        win_rate = win_trades / (num_trades / 2) if num_trades > 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'avg_beta': np.mean(self.beta_history) if self.beta_history else 1.0,
            'portfolio_data': portfolio_df,
            'trades_data': trades_df
        }
        
        return results

    def plot_results(self, results: Dict):
        """Generate comprehensive performance plots"""
        portfolio_df = results['portfolio_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Arbitrage Backtest Results', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df['timestamp'], portfolio_df['value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Z-score over time
        axes[0, 1].plot(portfolio_df['timestamp'], portfolio_df['z_score'])
        axes[0, 1].axhline(y=self.z_score_entry, color='r', linestyle='--', label='Entry')
        axes[0, 1].axhline(y=-self.z_score_entry, color='r', linestyle='--')
        axes[0, 1].axhline(y=self.z_score_exit, color='g', linestyle='--', label='Exit')
        axes[0, 1].axhline(y=-self.z_score_exit, color='g', linestyle='--')
        axes[0, 1].set_title('Z-Score Over Time')
        axes[0, 1].set_ylabel('Z-Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Beta over time
        axes[1, 0].plot(portfolio_df['timestamp'], portfolio_df['beta'])
        axes[1, 0].set_title('Beta (ETH/BTC) Over Time')
        axes[1, 0].set_ylabel('Beta')
        axes[1, 0].grid(True)
        
        # Drawdown
        running_max = portfolio_df['value'].expanding().max()
        drawdown = (portfolio_df['value'] - running_max) / running_max
        axes[1, 1].fill_between(portfolio_df['timestamp'], drawdown, 0, alpha=0.3, color='red')
        axes[1, 1].set_title('Drawdown')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self, results: Dict):
        """Print formatted backtest summary"""
        print("\n" + "="*60)
        print("STATISTICAL ARBITRAGE BACKTEST SUMMARY")
        print("="*60)
        print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"Final Value:        ${results['final_value']:,.2f}")
        print(f"Total Return:       {results['total_return']*100:.2f}%")
        print(f"Total P&L:          ${results['total_pnl']:,.2f}")
        print()
        print(f"Number of Trades:   {results['num_trades']}")
        print(f"Win Rate:           {results['win_rate']*100:.1f}%")
        print(f"Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"Volatility:         {results['volatility']*100:.2f}%")
        print(f"Max Drawdown:       {results['max_drawdown']*100:.2f}%")
        print(f"Average Beta:       {results['avg_beta']:.3f}")
        print("="*60)


def main():
    """Run a sample backtest"""
    print("Statistical Arbitrage Backtesting Engine")
    print("="*50)
    
    # Initialize backtester
    backtester = StatArbBacktester()
    
    # Load historical data (sample data for demonstration)
    data = backtester.load_historical_data("sample")
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    # Display results
    backtester.print_summary(results)
    
    # Generate plots
    backtester.plot_results(results)
    
    return results


if __name__ == "__main__":
    results = main()