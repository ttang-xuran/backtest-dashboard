#!/usr/bin/env python3
"""
Statistical Arbitrage Trading Bot for BTC/USD and ETH/USD on Hyperliquid
This is a defensive trading tool for statistical analysis and pairs trading.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import time

try:
    from hyperliquid import HyperliquidAsync
except ImportError:
    print("Please install hyperliquid: pip install hyperliquid")
    exit(1)

from cointegration_tests import CointegrationTester

class StatisticalArbitrageBot:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Statistical Arbitrage Bot
        
        Args:
            config_path: Path to configuration file containing API credentials
        """
        self.config = self.load_config(config_path)
        self.hl = None
        self.btc_symbol = "BTC"
        self.eth_symbol = "ETH"
        self.position_size = self.config.get("position_size", 0.01)
        self.z_score_entry = self.config.get("z_score_entry", 2.0)
        self.z_score_exit = self.config.get("z_score_exit", 0.5)
        self.lookback_period = self.config.get("lookback_period", 100)
        self.price_history = {"BTC": [], "ETH": []}
        self.positions = {"BTC": 0, "ETH": 0}
        self.spread_history = []
        self.is_trading = False
        self.cointegration_tester = CointegrationTester()
        self.cointegration_validated = False
        self.last_cointegration_check = None
        self.cointegration_check_interval = self.config.get("cointegration_check_hours", 24) * 3600  # seconds
        
        # Beta hedging variables
        self.beta_eth_btc = 1.0  # Default beta, will be calculated
        self.returns_history = {"BTC": [], "ETH": []}  # Store returns for beta calculation
        self.min_beta_periods = 10  # Minimum periods to calculate reliable beta
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stat_arb_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file {config_path} not found. Creating template...")
            template_config = {
                "account_address": "YOUR_PUBLIC_KEY_HERE",
                "secret_key": "YOUR_PRIVATE_KEY_HERE",
                "position_size": 0.01,
                "z_score_entry": 2.0,
                "z_score_exit": 0.5,
                "lookback_period": 100,
                "max_position_size": 0.1,
                "stop_loss_pct": 0.05
            }
            with open(config_path, 'w') as f:
                json.dump(template_config, f, indent=4)
            self.logger.info(f"Template config created at {config_path}. Please update with your credentials.")
            exit(1)

    async def initialize(self):
        """Initialize the Hyperliquid connection"""
        try:
            self.hl = HyperliquidAsync({
                "account_address": self.config["account_address"],
                "secret_key": self.config["secret_key"]
            })
            self.logger.info("Successfully connected to Hyperliquid")
        except Exception as e:
            self.logger.error(f"Failed to initialize Hyperliquid connection: {e}")
            raise

    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            # Get all tickers from the market
            tickers = await self.hl.fetch_tickers()
            
            # Construct the symbol in the format used by Hyperliquid
            market_symbol = f"{symbol}/USD:USD"
            
            if market_symbol in tickers:
                # Use 'close' price from ticker
                return float(tickers[market_symbol]['close'])
            
            # Alternative: try to find in markets data
            markets = await self.hl.fetch_markets()
            for market in markets:
                if market['base'] == symbol:
                    # Use markPx from market info
                    return float(market['info']['markPx'])
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None

    async def update_price_history(self):
        """Update price history for both BTC and ETH"""
        btc_price = await self.get_market_price(self.btc_symbol)
        eth_price = await self.get_market_price(self.eth_symbol)
        
        if btc_price and eth_price:
            # Calculate returns if we have previous prices
            if len(self.price_history["BTC"]) > 0:
                btc_return = (btc_price - self.price_history["BTC"][-1]) / self.price_history["BTC"][-1]
                eth_return = (eth_price - self.price_history["ETH"][-1]) / self.price_history["ETH"][-1]
                
                self.returns_history["BTC"].append(btc_return)
                self.returns_history["ETH"].append(eth_return)
                
                # Keep only the last lookback_period returns
                if len(self.returns_history["BTC"]) > self.lookback_period:
                    self.returns_history["BTC"] = self.returns_history["BTC"][-self.lookback_period:]
                    self.returns_history["ETH"] = self.returns_history["ETH"][-self.lookback_period:]
            
            self.price_history["BTC"].append(btc_price)
            self.price_history["ETH"].append(eth_price)
            
            # Keep only the last lookback_period prices
            if len(self.price_history["BTC"]) > self.lookback_period:
                self.price_history["BTC"] = self.price_history["BTC"][-self.lookback_period:]
                self.price_history["ETH"] = self.price_history["ETH"][-self.lookback_period:]
            
            # Update beta calculation
            self.update_beta()
            
            return True
        return False

    def calculate_spread_statistics(self) -> Tuple[float, float, float]:
        """
        Calculate the spread between BTC and ETH prices and its statistics
        Returns: (current_spread, mean_spread, std_spread)
        """
        if len(self.price_history["BTC"]) < 2 or len(self.price_history["ETH"]) < 2:
            return 0, 0, 0
        
        btc_prices = np.array(self.price_history["BTC"])
        eth_prices = np.array(self.price_history["ETH"])
        
        # Calculate log price ratio (spread)
        spreads = np.log(btc_prices / eth_prices)
        self.spread_history = spreads.tolist()
        
        current_spread = spreads[-1]
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        return current_spread, mean_spread, std_spread

    def calculate_z_score(self) -> float:
        """Calculate Z-score of current spread"""
        current_spread, mean_spread, std_spread = self.calculate_spread_statistics()
        
        if std_spread == 0:
            return 0
        
        z_score = (current_spread - mean_spread) / std_spread
        return z_score

    def update_beta(self):
        """Calculate beta (ETH relative to BTC) for hedging"""
        if len(self.returns_history["BTC"]) < self.min_beta_periods:
            return  # Not enough data for reliable beta
        
        # Convert to numpy arrays for calculation
        btc_returns = np.array(self.returns_history["BTC"])
        eth_returns = np.array(self.returns_history["ETH"])
        
        # Calculate beta: Cov(ETH, BTC) / Var(BTC)
        covariance = np.cov(eth_returns, btc_returns)[0, 1]
        btc_variance = np.var(btc_returns)
        
        if btc_variance > 0:
            self.beta_eth_btc = covariance / btc_variance
        else:
            self.beta_eth_btc = 1.0  # Default fallback

    def get_position_sizes(self, base_position_size: float) -> Tuple[float, float]:
        """
        Calculate beta-hedged position sizes
        
        Args:
            base_position_size: Base position size for BTC
            
        Returns:
            Tuple of (btc_position_size, eth_position_size)
        """
        btc_size = base_position_size
        eth_size = base_position_size * abs(self.beta_eth_btc)
        return btc_size, eth_size

    async def get_account_info(self) -> Dict:
        """Get account information including positions and balance"""
        try:
            return await self.hl.account_state()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}

    async def place_order(self, symbol: str, side: str, size: float, order_type: str = "market") -> bool:
        """
        Place an order on Hyperliquid
        
        Args:
            symbol: Trading symbol (BTC or ETH)
            side: 'buy' or 'sell'
            size: Order size
            order_type: 'market' or 'limit'
        """
        try:
            if order_type == "market":
                result = await self.hl.market_order(symbol, side, size)
            else:
                # For limit orders, you'd need to specify price
                price = await self.get_market_price(symbol)
                if not price:
                    return False
                result = await self.hl.limit_order(symbol, side, size, price)
            
            self.logger.info(f"Order placed: {side} {size} {symbol} - Result: {result}")
            return True
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return False

    async def validate_cointegration(self) -> bool:
        """
        Validate cointegration between BTC and ETH before trading
        """
        current_time = time.time()
        
        # Check if we need to revalidate cointegration
        if (self.last_cointegration_check is None or 
            current_time - self.last_cointegration_check > self.cointegration_check_interval or
            not self.cointegration_validated):
            
            self.logger.info("Validating cointegration between BTC and ETH...")
            
            # Need sufficient data for cointegration test
            if len(self.price_history["BTC"]) < self.lookback_period or len(self.price_history["ETH"]) < self.lookback_period:
                self.logger.warning("Insufficient data for cointegration test")
                return False
            
            # Perform cointegration validation
            is_suitable, reason, test_results = self.cointegration_tester.validate_trading_pair(
                self.price_history["BTC"], 
                self.price_history["ETH"]
            )
            
            self.cointegration_validated = is_suitable
            self.last_cointegration_check = current_time
            
            self.logger.info(f"Cointegration validation result: {is_suitable}")
            self.logger.info(f"Reason: {reason}")
            
            if is_suitable:
                # Log key statistics
                spread_analysis = test_results.get('spread_analysis', {})
                half_life = spread_analysis.get('half_life', 'N/A')
                self.logger.info(f"Half-life: {half_life} periods")
                
                overall = test_results.get('overall_assessment', {})
                confidence = overall.get('confidence', 'unknown')
                self.logger.info(f"Cointegration confidence: {confidence}")
            
            return is_suitable
        
        return self.cointegration_validated

    async def execute_arbitrage_strategy(self):
        """Execute the statistical arbitrage strategy"""
        if len(self.price_history["BTC"]) < self.lookback_period:
            self.logger.info(f"Not enough price history. Have {len(self.price_history['BTC'])}, need {self.lookback_period}")
            return

        # Validate cointegration before trading
        if not await self.validate_cointegration():
            self.logger.warning("Cointegration validation failed - skipping trading")
            return

        z_score = self.calculate_z_score()
        self.logger.info(f"Current Z-score: {z_score:.4f}")

        # Entry conditions
        if abs(z_score) > self.z_score_entry and not self.is_trading:
            # Get beta-hedged position sizes
            btc_size, eth_size = self.get_position_sizes(self.position_size)
            
            if z_score > self.z_score_entry:
                # Spread is high: BTC overpriced relative to ETH
                # Short BTC, Long ETH (beta-hedged)
                self.logger.info(f"Entering trade: SHORT BTC({btc_size:.4f}), LONG ETH({eth_size:.4f}) - Beta: {self.beta_eth_btc:.4f} (Z-score: {z_score:.4f})")
                
                btc_success = await self.place_order(self.btc_symbol, "sell", btc_size)
                eth_success = await self.place_order(self.eth_symbol, "buy", eth_size)
                
                if btc_success and eth_success:
                    self.positions["BTC"] = -btc_size
                    self.positions["ETH"] = eth_size
                    self.is_trading = True
                    
            elif z_score < -self.z_score_entry:
                # Spread is low: ETH overpriced relative to BTC
                # Long BTC, Short ETH (beta-hedged)
                self.logger.info(f"Entering trade: LONG BTC({btc_size:.4f}), SHORT ETH({eth_size:.4f}) - Beta: {self.beta_eth_btc:.4f} (Z-score: {z_score:.4f})")
                
                btc_success = await self.place_order(self.btc_symbol, "buy", btc_size)
                eth_success = await self.place_order(self.eth_symbol, "sell", eth_size)
                
                if btc_success and eth_success:
                    self.positions["BTC"] = btc_size
                    self.positions["ETH"] = -eth_size
                    self.is_trading = True

        # Exit conditions
        elif abs(z_score) < self.z_score_exit and self.is_trading:
            self.logger.info(f"Exiting trade (Z-score: {z_score:.4f})")
            
            # Close positions
            if self.positions["BTC"] > 0:
                await self.place_order(self.btc_symbol, "sell", abs(self.positions["BTC"]))
            elif self.positions["BTC"] < 0:
                await self.place_order(self.btc_symbol, "buy", abs(self.positions["BTC"]))
                
            if self.positions["ETH"] > 0:
                await self.place_order(self.eth_symbol, "sell", abs(self.positions["ETH"]))
            elif self.positions["ETH"] < 0:
                await self.place_order(self.eth_symbol, "buy", abs(self.positions["ETH"]))
            
            # Reset positions
            self.positions = {"BTC": 0, "ETH": 0}
            self.is_trading = False

    async def run_strategy(self, duration_hours: int = 24):
        """
        Run the statistical arbitrage strategy
        
        Args:
            duration_hours: How long to run the strategy in hours
        """
        await self.initialize()
        
        self.logger.info(f"Starting Statistical Arbitrage Bot for {duration_hours} hours")
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # Update price data
                if await self.update_price_history():
                    # Execute strategy
                    await self.execute_arbitrage_strategy()
                    
                    # Log current status
                    btc_price = self.price_history["BTC"][-1] if self.price_history["BTC"] else 0
                    eth_price = self.price_history["ETH"][-1] if self.price_history["ETH"] else 0
                    z_score = self.calculate_z_score()
                    
                    # Get beta info for logging
                    beta_info = f", Beta: {self.beta_eth_btc:.4f}" if len(self.returns_history["BTC"]) >= self.min_beta_periods else ""
                    
                    self.logger.info(
                        f"BTC: ${btc_price:.2f}, ETH: ${eth_price:.2f}, "
                        f"Z-score: {z_score:.4f}{beta_info}, Trading: {self.is_trading}, "
                        f"Positions: BTC={self.positions['BTC']:.4f}, ETH={self.positions['ETH']:.4f}"
                    )
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 seconds between checks
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
        
        self.logger.info("Strategy execution completed")

def main():
    """Main function to run the statistical arbitrage bot"""
    print("Statistical Arbitrage Bot for Hyperliquid")
    print("=" * 50)
    print("This bot implements pairs trading between BTC and ETH")
    print("Make sure to configure your API credentials in config.json")
    print()
    
    try:
        duration = input("Enter duration to run (hours, default 1): ").strip()
        duration = float(duration) if duration else 1.0
        
        bot = StatisticalArbitrageBot()
        asyncio.run(bot.run_strategy(duration_hours=duration))
        
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()