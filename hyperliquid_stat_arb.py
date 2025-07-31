#!/usr/bin/env python3
"""
Statistical Arbitrage Trading Bot for BTC/USD and ETH/USD on Hyperliquid
This is a defensive trading tool for statistical analysis and pairs trading.

SECURITY NOTE:
- Credentials are loaded from local Windows folder (WSL path: /mnt/c/Users/16473/Desktop/Trading/hyperliquid/)
- This path is excluded from Git via .gitignore to prevent accidental credential exposure  
- Never commit credential files to version control
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
    def __init__(self, config_path: str = None):
        """
        Initialize the Statistical Arbitrage Bot
        
        Args:
            config_path: Path to configuration file containing API credentials
        """
        # Set default credential path to Windows hyperliquid folder
        if config_path is None:
            config_path = "/mnt/c/Users/16473/Desktop/Trading/hyperliquid/trade_api.json"
        
        self.config = self.load_config(config_path)
        self.hl = None
        self.btc_symbol = "BTC"
        self.eth_symbol = "ETH"
        # UPDATED PARAMETERS TO MATCH BACKTESTING OPTIMIZATION
        self.position_size = self.config.get("position_size", 0.20)  # Increased from 0.01 to 0.20 (20% per asset)
        self.z_score_entry = self.config.get("z_score_entry", 3.0)   # Increased from 2.0 to 3.0 (ultra-selective)
        self.z_score_exit = self.config.get("z_score_exit", 0.8)     # Increased from 0.5 to 0.8 (let profits run)
        self.lookback_period = self.config.get("lookback_period", 30) # Reduced from 100 to 30 (faster signals)
        
        # CONTRARIAN MODE - Key addition!
        self.contrarian_mode = self.config.get("contrarian_mode", True)  # Default to contrarian mode
        
        # ENHANCED RISK MANAGEMENT (matching backtesting)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.015)      # 1.5% stop loss
        self.max_holding_minutes = self.config.get("max_holding_minutes", 360)  # 6 hours max
        self.min_holding_minutes = self.config.get("min_holding_minutes", 180)  # 3 hours min
        self.daily_trade_limit = self.config.get("daily_trade_limit", 2)  # Max 2 trades per day
        
        # MARKET REGIME FILTERS (matching backtesting)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.02)  # 2% volatility limit
        self.correlation_min = self.config.get("correlation_min", 0.7)      # High correlation required
        self.beta_min = self.config.get("beta_min", 0.6)                    # Beta range
        self.beta_max = self.config.get("beta_max", 1.4)                    # Beta range
        
        self.price_history = {"BTC": [], "ETH": []}
        self.positions = {"BTC": 0, "ETH": 0}
        self.spread_history = []
        self.is_trading = False
        self.cointegration_tester = CointegrationTester()
        self.cointegration_validated = False
        self.last_cointegration_check = None
        self.cointegration_check_interval = self.config.get("cointegration_check_hours", 24) * 3600  # seconds
        
        # TRADE TRACKING (for risk management)
        self.entry_time = None
        self.entry_prices = {"BTC": 0, "ETH": 0}
        self.daily_trades_count = 0
        self.last_trade_date = None
        
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
        
        # Log initialization with contrarian mode info
        mode_name = "CONTRARIAN" if self.contrarian_mode else "TRADITIONAL"
        self.logger.info(f"🔄 {mode_name} Statistical Arbitrage Bot initialized")
        
        if self.contrarian_mode:
            self.logger.info("🎯 CONTRARIAN MODE: Trading OPPOSITE of traditional signals")
            self.logger.info("💡 Logic: High Z-score → Long BTC, Low Z-score → Short BTC")
        else:
            self.logger.info("📈 TRADITIONAL MODE: Standard mean reversion strategy")
        
        self.logger.info(f"Configuration: Position Size: {self.position_size*100:.1f}% per asset")
        self.logger.info(f"Z-Entry: {self.z_score_entry} (ultra-selective), Z-Exit: {self.z_score_exit}")
        self.logger.info(f"Lookback Period: {self.lookback_period}, Daily Trade Limit: {self.daily_trade_limit}")
        self.logger.info(f"Risk Management: Stop Loss: {self.stop_loss_pct*100:.1f}%, Max Hold: {self.max_holding_minutes}min")

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Validate that required keys exist
            if "account_address" not in config or "secret_key" not in config:
                raise ValueError("Config file missing required keys: account_address or secret_key")
                
            return config
            
        except FileNotFoundError:
            print(f"❌ Credential file not found at: {config_path}")
            print("📁 Expected location: C:\\Users\\16473\\Desktop\\Trading\\hyperliquid\\trade_api.json")
            print("🔑 Please ensure your Hyperliquid credentials are stored in the expected location")
            print("📋 Required format: {\"account_address\": \"...\", \"secret_key\": \"...\"}")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in credential file: {e}")
            exit(1)
        except ValueError as e:
            print(f"❌ {e}")
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
    
    def check_market_regime(self) -> bool:
        """Check if market conditions are suitable for trading (matching backtesting logic)"""
        if len(self.returns_history["BTC"]) < 20 or len(self.returns_history["ETH"]) < 20:
            return True  # Default allow if not enough data
        
        # Get recent returns
        btc_returns = np.array(self.returns_history["BTC"][-20:])
        eth_returns = np.array(self.returns_history["ETH"][-20:])
        
        # Calculate volatility
        btc_vol = np.std(btc_returns)
        eth_vol = np.std(eth_returns)
        avg_volatility = (btc_vol + eth_vol) / 2
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(btc_returns, eth_returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0.75  # Default if calculation fails
        except:
            correlation = 0.75
        
        # Check beta range
        beta_in_range = self.beta_min <= abs(self.beta_eth_btc) <= self.beta_max
        
        # Market regime checks
        volatile_regime = avg_volatility > self.volatility_threshold
        low_correlation = correlation < self.correlation_min
        
        tradeable = not volatile_regime and not low_correlation and beta_in_range
        
        if not tradeable:
            if volatile_regime:
                self.logger.warning(f"High volatility detected: {avg_volatility:.4f} > {self.volatility_threshold}")
            if low_correlation:
                self.logger.warning(f"Low correlation detected: {correlation:.4f} < {self.correlation_min}")
            if not beta_in_range:
                self.logger.warning(f"Beta out of range: {self.beta_eth_btc:.4f} not in [{self.beta_min}, {self.beta_max}]")
        
        return tradeable
    
    def check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit has been reached"""
        current_date = datetime.now().date()
        
        # Reset counter if it's a new day
        if self.last_trade_date != current_date:
            self.daily_trades_count = 0
            self.last_trade_date = current_date
        
        if self.daily_trades_count >= self.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached: {self.daily_trades_count}/{self.daily_trade_limit}")
            return False
        
        return True
    
    def check_holding_time_limits(self) -> Tuple[bool, str]:
        """Check if holding time limits should trigger exit"""
        if not self.is_trading or self.entry_time is None:
            return False, ""
        
        holding_minutes = (datetime.now() - self.entry_time).total_seconds() / 60
        
        # Minimum holding time check
        if holding_minutes < self.min_holding_minutes:
            return False, ""
        
        # Maximum holding time check
        if holding_minutes > self.max_holding_minutes:
            return True, f"Maximum holding time reached: {holding_minutes:.1f} minutes"
        
        return False, ""
    
    def check_stop_loss(self) -> Tuple[bool, str]:
        """Check if stop loss should be triggered"""
        if not self.is_trading:
            return False, ""
        
        try:
            current_btc = self.price_history["BTC"][-1]
            current_eth = self.price_history["ETH"][-1]
            entry_btc = self.entry_prices["BTC"]
            entry_eth = self.entry_prices["ETH"]
            
            # Calculate current P&L percentage (simplified)
            if self.positions["BTC"] > 0:  # Long BTC position
                btc_return = (current_btc - entry_btc) / entry_btc
                eth_return = -(current_eth - entry_eth) / entry_eth
            else:  # Short BTC position
                btc_return = -(current_btc - entry_btc) / entry_btc
                eth_return = (current_eth - entry_eth) / entry_eth
            
            total_return = btc_return + eth_return * abs(self.beta_eth_btc)
            
            if total_return < -self.stop_loss_pct:
                return True, f"Stop loss triggered: {total_return*100:.2f}% < -{self.stop_loss_pct*100:.1f}%"
        
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
        
        return False, ""

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
        """Execute the CONTRARIAN statistical arbitrage strategy (matching backtesting)"""
        if len(self.price_history["BTC"]) < self.lookback_period:
            self.logger.info(f"Not enough price history. Have {len(self.price_history['BTC'])}, need {self.lookback_period}")
            return

        # ENHANCED RISK CHECKS (matching backtesting)
        
        # 1. Market regime check
        if not self.check_market_regime():
            return
        
        # 2. Daily trade limit check
        if not self.check_daily_trade_limit():
            return
        
        # 3. Validate cointegration before trading
        if not await self.validate_cointegration():
            self.logger.warning("Cointegration validation failed - skipping trading")
            return

        z_score = self.calculate_z_score()
        self.logger.info(f"Current Z-score: {z_score:.4f} | Market regime: OK | Daily trades: {self.daily_trades_count}/{self.daily_trade_limit}")

        # EXIT CONDITIONS (enhanced with risk management)
        if self.is_trading:
            should_exit = False
            exit_reason = ""
            
            # Check exit conditions in priority order
            stop_loss_exit, stop_loss_reason = self.check_stop_loss()
            time_exit, time_reason = self.check_holding_time_limits()
            
            if stop_loss_exit:
                should_exit = True
                exit_reason = stop_loss_reason
            elif time_exit:
                should_exit = True  
                exit_reason = time_reason
            elif abs(z_score) < self.z_score_exit:
                should_exit = True
                exit_reason = f"Z-score mean reversion: {z_score:.4f} < {self.z_score_exit}"
            
            if should_exit:
                await self.close_positions(exit_reason)
                return

        # ENTRY CONDITIONS (CONTRARIAN LOGIC!)
        if abs(z_score) > self.z_score_entry and not self.is_trading:
            # Get beta-hedged position sizes
            btc_size, eth_size = self.get_position_sizes(self.position_size)
            
            # Store entry information for risk management
            self.entry_time = datetime.now()
            self.entry_prices["BTC"] = self.price_history["BTC"][-1]
            self.entry_prices["ETH"] = self.price_history["ETH"][-1]
            
            if not self.contrarian_mode:
                # TRADITIONAL LOGIC
                if z_score > self.z_score_entry:
                    # Traditional: BTC overpriced → Short BTC, Long ETH
                    self.logger.info(f"TRADITIONAL: Entering SHORT BTC({btc_size:.4f}), LONG ETH({eth_size:.4f}) - Beta: {self.beta_eth_btc:.4f} (Z-score: {z_score:.4f})")
                    btc_success = await self.place_order(self.btc_symbol, "sell", btc_size)
                    eth_success = await self.place_order(self.eth_symbol, "buy", eth_size)
                    
                    if btc_success and eth_success:
                        self.positions["BTC"] = -btc_size
                        self.positions["ETH"] = eth_size
                        self.is_trading = True
                        self.daily_trades_count += 1
                        
                elif z_score < -self.z_score_entry:
                    # Traditional: BTC underpriced → Long BTC, Short ETH
                    self.logger.info(f"TRADITIONAL: Entering LONG BTC({btc_size:.4f}), SHORT ETH({eth_size:.4f}) - Beta: {self.beta_eth_btc:.4f} (Z-score: {z_score:.4f})")
                    btc_success = await self.place_order(self.btc_symbol, "buy", btc_size)
                    eth_success = await self.place_order(self.eth_symbol, "sell", eth_size)
                    
                    if btc_success and eth_success:
                        self.positions["BTC"] = btc_size
                        self.positions["ETH"] = -eth_size
                        self.is_trading = True
                        self.daily_trades_count += 1
                        
            else:
                # CONTRARIAN LOGIC - DO THE OPPOSITE!
                if z_score > self.z_score_entry:
                    # Contrarian: High Z-score → Long BTC (opposite of traditional)
                    self.logger.info(f"CONTRARIAN: Entering LONG BTC({btc_size:.4f}), SHORT ETH({eth_size:.4f}) - Beta: {self.beta_eth_btc:.4f} (Z-score: {z_score:.4f})")
                    btc_success = await self.place_order(self.btc_symbol, "buy", btc_size)
                    eth_success = await self.place_order(self.eth_symbol, "sell", eth_size)
                    
                    if btc_success and eth_success:
                        self.positions["BTC"] = btc_size
                        self.positions["ETH"] = -eth_size
                        self.is_trading = True
                        self.daily_trades_count += 1
                        
                elif z_score < -self.z_score_entry:
                    # Contrarian: Low Z-score → Short BTC (opposite of traditional) 
                    self.logger.info(f"CONTRARIAN: Entering SHORT BTC({btc_size:.4f}), LONG ETH({eth_size:.4f}) - Beta: {self.beta_eth_btc:.4f} (Z-score: {z_score:.4f})")
                    btc_success = await self.place_order(self.btc_symbol, "sell", btc_size)
                    eth_success = await self.place_order(self.eth_symbol, "buy", eth_size)
                    
                    if btc_success and eth_success:
                        self.positions["BTC"] = -btc_size
                        self.positions["ETH"] = eth_size
                        self.is_trading = True
                        self.daily_trades_count += 1
    
    async def close_positions(self, reason: str):
        """Close all positions with specified reason"""
        if not self.is_trading:
            return
        
        self.logger.info(f"Closing positions: {reason}")
        
        # Close BTC position
        if self.positions["BTC"] > 0:
            btc_success = await self.place_order(self.btc_symbol, "sell", abs(self.positions["BTC"]))
        elif self.positions["BTC"] < 0:
            btc_success = await self.place_order(self.btc_symbol, "buy", abs(self.positions["BTC"]))
        else:
            btc_success = True
        
        # Close ETH position
        if self.positions["ETH"] > 0:
            eth_success = await self.place_order(self.eth_symbol, "sell", abs(self.positions["ETH"]))
        elif self.positions["ETH"] < 0:
            eth_success = await self.place_order(self.eth_symbol, "buy", abs(self.positions["ETH"]))
        else:
            eth_success = True
        
        if btc_success and eth_success:
            # Reset position tracking
            self.positions["BTC"] = 0
            self.positions["ETH"] = 0
            self.is_trading = False
            self.entry_time = None
            self.entry_prices = {"BTC": 0, "ETH": 0}
            
            self.logger.info("Positions closed successfully")
        else:
            self.logger.error("Failed to close some positions - manual intervention may be required")

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
        
        # Properly close the connection
        if self.hl:
            await self.hl.close()

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