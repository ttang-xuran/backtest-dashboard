#!/usr/bin/env python3
"""
Modernized Statistical Arbitrage Trading Bot
Refactored architecture with service layers, comprehensive error handling, and monitoring
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional
import traceback

try:
    from hyperliquid import HyperliquidAsync
except ImportError:
    print("Please install hyperliquid: pip install hyperliquid")
    sys.exit(1)

from config.settings import config_manager
from services.data_service import DataService
from services.trading_service import TradingService, SignalType
from services.risk_service import RiskService
from utils.logger import get_logger, get_trade_logger, get_performance_monitor
from utils.error_handler import handle_errors, ErrorCategory, ErrorSeverity, health_monitor
from utils.circuit_breaker import hyperliquid_circuit_breaker, circuit_registry
from utils.validators import market_data_validator, trading_signal_validator


class ModernizedStatArbBot:
    def __init__(self, config_file: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.trade_logger = get_trade_logger(__name__)
        self.perf_monitor = get_performance_monitor(__name__)
        
        if config_file:
            config_manager.load_from_file(config_file)
        
        if not config_manager.validate_config():
            raise ValueError("Invalid configuration")
        
        self.config = config_manager.config
        self.hl: Optional[HyperliquidAsync] = None
        
        self.data_service: Optional[DataService] = None
        self.trading_service: Optional[TradingService] = None
        self.risk_service: Optional[RiskService] = None
        
        self.is_running = False
        self.main_loop_task: Optional[asyncio.Task] = None
        
        self.symbols = ["BTC", "ETH"]
        self.update_interval = 60  # seconds
        
        self._setup_signal_handlers()
        self._register_health_checks()
        
        self.logger.info("ModernizedStatArbBot initialized")
    
    def _setup_signal_handlers(self):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(self.shutdown())
    
    def _register_health_checks(self):
        health_monitor.register_health_check("config", self._health_check_config)
        health_monitor.register_health_check("services", self._health_check_services)
        health_monitor.register_health_check("circuit_breakers", self._health_check_circuit_breakers)
    
    async def _health_check_config(self) -> bool:
        return config_manager.validate_config()
    
    async def _health_check_services(self) -> bool:
        if not all([self.data_service, self.trading_service, self.risk_service]):
            return False
        
        try:
            data_health = await self.data_service.health_check()
            risk_health = await self.risk_service.health_check()
            
            return (data_health.get('status') in ['healthy', 'degraded'] and
                   risk_health.get('status') in ['healthy', 'degraded'])
        except Exception:
            return False
    
    async def _health_check_circuit_breakers(self) -> bool:
        health_summary = circuit_registry.get_health_summary()
        return health_summary['overall_health'] in ['healthy', 'degraded', 'recovering']
    
    @handle_errors(retry_count=3, error_category=ErrorCategory.API, severity=ErrorSeverity.HIGH)
    async def initialize(self):
        try:
            self.logger.info("Initializing Hyperliquid connection...")
            credentials = config_manager.get_credentials()
            
            self.hl = HyperliquidAsync({
                "account_address": credentials["account_address"],
                "secret_key": credentials["secret_key"]
            })
            
            await self._test_connection()
            
            self.data_service = DataService(self.hl)
            self.risk_service = RiskService(self.data_service)
            self.trading_service = TradingService(self.hl, self.data_service)
            self.trading_service.risk_service = self.risk_service
            
            self.logger.info("All services initialized successfully")
            
            # Initial data population
            await self._initial_data_collection()
            
        except Exception as e:
            self.logger.critical(f"Failed to initialize bot: {e}")
            raise
    
    @hyperliquid_circuit_breaker
    async def _test_connection(self):
        account_info = await self.hl.account_state()
        self.logger.info(f"Connection successful. Account info retrieved: {bool(account_info)}")
        return account_info
    
    async def _initial_data_collection(self):
        self.logger.info("Collecting initial market data...")
        
        for _ in range(5):  # Collect 5 data points
            success = await self.data_service.update_price_history(self.symbols)
            if all(success.values()):
                self.logger.info("Initial data collection successful")
                await asyncio.sleep(10)  # 10 second intervals for initial collection
            else:
                self.logger.warning(f"Some data collection failed: {success}")
                await asyncio.sleep(5)
        
        self.logger.info("Initial data collection completed")
    
    async def run(self):
        try:
            await self.initialize()
            
            self.is_running = True
            self.main_loop_task = asyncio.create_task(self._main_loop())
            
            self.logger.info("🚀 ModernizedStatArbBot started successfully")
            self.logger.info(f"Trading symbols: {self.symbols}")
            self.logger.info(f"Update interval: {self.update_interval}s")
            self.logger.info(f"Z-score entry: {self.config.trading.z_score_entry}")
            self.logger.info(f"Z-score exit: {self.config.trading.z_score_exit}")
            
            await self.main_loop_task
            
        except Exception as e:
            self.logger.critical(f"Critical error in main execution: {e}")
            self.logger.critical(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            await self.shutdown()
    
    async def _main_loop(self):
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.is_running:
            try:
                self.perf_monitor.start_timing("main_loop_cycle")
                
                # Update market data
                success = await self._update_market_data()
                if not success:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.critical("Too many consecutive data errors, stopping bot")
                        break
                    continue
                else:
                    consecutive_errors = 0
                
                # Process trading logic
                await self._process_trading_logic()
                
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Periodic health checks
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    await self._periodic_health_check()
                
                # Log performance metrics
                cycle_time = self.perf_monitor.end_timing("main_loop_cycle")
                
                if cycle_time > 30:  # Log slow cycles
                    self.logger.warning(f"Slow main loop cycle: {cycle_time:.2f}s")
                
                # Sleep until next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Error in main loop (consecutive: {consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical("Too many consecutive errors, stopping bot")
                    break
                
                # Exponential backoff
                sleep_time = min(300, 5 * (2 ** consecutive_errors))
                await asyncio.sleep(sleep_time)
    
    @handle_errors(retry_count=2, error_category=ErrorCategory.DATA, severity=ErrorSeverity.MEDIUM)
    async def _update_market_data(self) -> bool:
        try:
            results = await self.data_service.update_price_history(self.symbols)
            
            # Validate market data
            for symbol in self.symbols:
                if symbol in self.data_service.price_history and len(self.data_service.price_history[symbol]) > 0:
                    latest_price = self.data_service.price_history[symbol][-1]
                    
                    market_data = {
                        'symbol': symbol,
                        'price': latest_price,
                        'timestamp': datetime.now()
                    }
                    
                    validation_result = market_data_validator.validate(market_data)
                    
                    if not validation_result.is_valid:
                        self.logger.error(f"Market data validation failed for {symbol}: {validation_result.errors}")
                        return False
                    
                    if validation_result.warnings:
                        self.logger.warning(f"Market data warnings for {symbol}: {validation_result.warnings}")
            
            success_count = sum(1 for success in results.values() if success)
            self.logger.debug(f"Market data update: {success_count}/{len(self.symbols)} successful")
            
            return success_count >= len(self.symbols) * 0.8  # 80% success rate required
            
        except Exception as e:
            self.logger.error(f"Failed to update market data: {e}")
            return False
    
    async def _process_trading_logic(self):
        try:
            if not self.trading_service or not self.risk_service:
                return
            
            # Check if risk service allows trading
            if self.risk_service.is_circuit_breaker_active():
                self.logger.warning("Risk circuit breaker active, skipping trading logic")
                return
            
            # Generate trading signal
            signal = self.trading_service.generate_signal()
            
            # Validate trading signal
            signal_data = {
                'signal_type': signal.signal_type.value,
                'z_score': signal.z_score,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp
            }
            
            validation_result = trading_signal_validator.validate(signal_data)
            
            if not validation_result.is_valid:
                self.logger.error(f"Invalid trading signal: {validation_result.errors}")
                return
            
            if validation_result.warnings:
                self.logger.warning(f"Trading signal warnings: {validation_result.warnings}")
            
            # Process signal based on type
            if signal.signal_type != SignalType.NO_SIGNAL:
                self.logger.info(f"Signal generated: {signal.signal_type.value} "
                               f"(Z-score: {signal.z_score:.3f}, Confidence: {signal.confidence:.3f})")
                
                # Risk validation before execution
                if signal.signal_type in [SignalType.LONG_BTC, SignalType.SHORT_BTC]:
                    can_trade, reason = await self._validate_trade_risk()
                    if not can_trade:
                        self.logger.warning(f"Trade blocked by risk check: {reason}")
                        return
                
                # Execute signal
                success = await self.trading_service.execute_signal(signal)
                
                if success:
                    self.logger.info(f"Successfully executed {signal.signal_type.value} signal")
                else:
                    self.logger.error(f"Failed to execute {signal.signal_type.value} signal")
            
            # Check for risk-based exits
            await self._check_risk_exits()
            
        except Exception as e:
            self.logger.error(f"Error in trading logic: {e}")
    
    async def _validate_trade_risk(self) -> tuple[bool, str]:
        if not self.risk_service or not self.data_service:
            return False, "Services not available"
        
        try:
            # Get current account info for risk calculation
            account_info = await self.hl.account_state()
            account_balance = float(account_info.get('marginSummary', {}).get('accountValue', 0))
            
            if account_balance <= 0:
                return False, "Invalid account balance"
            
            # Get current price for size calculation
            btc_price = await self.data_service.get_market_price("BTC")
            if not btc_price:
                return False, "Cannot get current BTC price"
            
            # Calculate position size
            btc_volatility = self.data_service.get_volatility("BTC") or 0.2
            confidence = 0.8  # Default confidence
            position_size = self.risk_service.calculate_position_size(account_balance, btc_volatility, confidence)
            
            # Check risk limits
            can_trade, risk_reason = self.risk_service.check_risk_limits(
                "BTC", "buy", position_size, btc_price, account_balance
            )
            
            return can_trade, risk_reason
            
        except Exception as e:
            return False, f"Risk validation error: {str(e)}"
    
    async def _check_risk_exits(self):
        try:
            if not self.trading_service.is_trading:
                return
            
            # Check stop loss
            should_exit, reason = self.trading_service.check_stop_loss()
            if should_exit:
                self.logger.warning(f"Stop loss triggered: {reason}")
                await self.trading_service.force_exit_all_positions("Stop loss")
                return
            
            # Check holding time limits
            should_exit, reason = self.trading_service.check_holding_time_limits()
            if should_exit:
                self.logger.info(f"Time-based exit triggered: {reason}")
                await self.trading_service.force_exit_all_positions("Time limit")
                return
            
            # Update unrealized PnL
            await self.trading_service.update_unrealized_pnl()
            
        except Exception as e:
            self.logger.error(f"Error checking risk exits: {e}")
    
    async def _update_risk_metrics(self):
        try:
            if not self.risk_service or not self.trading_service:
                return
            
            # Update risk service with current positions
            for symbol, position in self.trading_service.positions.items():
                current_price = await self.data_service.get_market_price(symbol)
                if current_price:
                    self.risk_service.update_position(
                        symbol, position.size, position.entry_price, 
                        current_price, position.entry_time
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    async def _periodic_health_check(self):
        try:
            health_report = await health_monitor.run_health_checks()
            
            if health_report['overall_status'] == 'unhealthy':
                self.logger.critical(f"System health check failed: {health_report}")
                
                # Consider emergency shutdown
                failed_checks = health_report.get('failed_checks', [])
                if 'services' in failed_checks:
                    self.logger.critical("Core services failed health check, emergency shutdown")
                    await self.emergency_shutdown("Failed health check")
            
            elif health_report['overall_status'] == 'degraded':
                self.logger.warning(f"System health degraded: {health_report}")
            
            # Log circuit breaker status
            cb_summary = circuit_registry.get_health_summary()
            if cb_summary['overall_health'] != 'healthy':
                self.logger.warning(f"Circuit breaker status: {cb_summary['overall_health']}")
            
        except Exception as e:
            self.logger.error(f"Error in periodic health check: {e}")
    
    async def emergency_shutdown(self, reason: str):
        self.logger.critical(f"🚨 EMERGENCY SHUTDOWN: {reason}")
        
        try:
            # Close all positions immediately
            if self.trading_service and self.trading_service.is_trading:
                await self.trading_service.force_exit_all_positions(f"Emergency: {reason}")
            
            # Trigger risk service emergency stop
            if self.risk_service:
                self.risk_service.emergency_stop(reason)
            
        except Exception as e:
            self.logger.critical(f"Error during emergency shutdown: {e}")
        
        finally:
            self.is_running = False
    
    async def shutdown(self):
        self.logger.info("Shutting down ModernizedStatArbBot...")
        
        self.is_running = False
        
        try:
            # Cancel main loop
            if self.main_loop_task and not self.main_loop_task.done():
                self.main_loop_task.cancel()
                try:
                    await self.main_loop_task
                except asyncio.CancelledError:
                    pass
            
            # Close positions if still trading
            if self.trading_service and self.trading_service.is_trading:
                self.logger.info("Closing open positions...")
                await self.trading_service.force_exit_all_positions("Bot shutdown")
            
            # Final logging
            if self.trading_service:
                stats = self.trading_service.get_trading_stats()
                self.logger.info(f"Final trading stats: {stats}")
            
            if self.risk_service:
                risk_summary = self.risk_service.get_risk_summary()
                self.logger.info(f"Final risk summary: Total PnL: {risk_summary['portfolio_metrics']['total_pnl']:.4f}")
            
            # Close connections
            if self.hl:
                # Close any open connections if the library supports it
                pass
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("ModernizedStatArbBot shutdown complete")
    
    async def get_status(self) -> dict:
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'health': await health_monitor.run_health_checks(),
                'circuit_breakers': circuit_registry.get_health_summary(),
            }
            
            if self.trading_service:
                status['trading'] = self.trading_service.get_trading_stats()
            
            if self.risk_service:
                status['risk'] = self.risk_service.get_risk_summary()
            
            if self.data_service:
                status['data'] = self.data_service.get_data_quality_report()
            
            return status
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'is_running': self.is_running
            }


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Modernized Statistical Arbitrage Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Setup logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    bot = None
    try:
        bot = ModernizedStatArbBot(config_file=args.config)
        await bot.run()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bot:
            await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())