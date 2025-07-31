#!/usr/bin/env python3
"""
Enhanced Statistical Arbitrage Trading Bot
Final implementation with regime detection, cost optimization, and adaptive trading
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
from services.enhanced_trading_service import EnhancedTradingService
from services.risk_service import RiskService
from utils.logger import get_logger, get_trade_logger, get_performance_monitor
from utils.error_handler import handle_errors, ErrorCategory, ErrorSeverity, health_monitor
from utils.circuit_breaker import circuit_registry


class EnhancedStatArbBot:
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
        self.trading_service: Optional[EnhancedTradingService] = None
        self.risk_service: Optional[RiskService] = None
        
        self.is_running = False
        self.main_loop_task: Optional[asyncio.Task] = None
        
        self.symbols = ["BTC", "ETH"]
        self.update_interval = 60
        
        # Performance tracking
        self.session_start_time = datetime.now()
        self.total_signals_generated = 0
        self.signals_executed = 0
        self.regime_changes_detected = 0
        
        self._setup_signal_handlers()
        self._register_health_checks()
        
        self.logger.info("🚀 Enhanced Statistical Arbitrage Bot initialized")
        self.logger.info("Features: Regime Detection | Cost Optimization | Adaptive Thresholds | Smart Filtering")
    
    def _setup_signal_handlers(self):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(self.shutdown())
    
    def _register_health_checks(self):
        health_monitor.register_health_check("config", self._health_check_config)
        health_monitor.register_health_check("services", self._health_check_services)
        health_monitor.register_health_check("market_conditions", self._health_check_market_conditions)
    
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
    
    async def _health_check_market_conditions(self) -> bool:
        """Check if market conditions are suitable for trading"""
        if not self.trading_service:
            return False
        
        try:
            regime_summary = self.trading_service.regime_service.get_regime_summary()
            return regime_summary.get('recommendation') == 'TRADE'
        except Exception:
            return False
    
    @handle_errors(retry_count=3, error_category=ErrorCategory.API, severity=ErrorSeverity.HIGH)
    async def initialize(self):
        try:
            self.logger.info("🔌 Initializing Hyperliquid connection...")
            credentials = config_manager.get_credentials()
            
            self.hl = HyperliquidAsync({
                "account_address": credentials["account_address"],
                "secret_key": credentials["secret_key"]
            })
            
            await self._test_connection()
            
            # Initialize services with enhanced architecture
            self.logger.info("🛠️ Initializing enhanced services...")
            self.data_service = DataService(self.hl)
            self.risk_service = RiskService(self.data_service)
            self.trading_service = EnhancedTradingService(self.hl, self.data_service, self.risk_service)
            
            self.logger.info("✅ All enhanced services initialized successfully")
            
            # Initial data population with regime detection
            await self._initial_enhanced_setup()
            
        except Exception as e:
            self.logger.critical(f"❌ Failed to initialize enhanced bot: {e}")
            raise
    
    async def _test_connection(self):
        """Test connection with enhanced error handling"""
        account_info = await self.hl.account_state()
        account_value = float(account_info.get('marginSummary', {}).get('accountValue', 0))
        
        self.logger.info(f"✅ Connection successful - Account Value: ${account_value:,.2f}")
        return account_info
    
    async def _initial_enhanced_setup(self):
        """Enhanced initial setup with regime detection"""
        self.logger.info("📊 Performing initial market analysis...")
        
        # Collect initial data
        for i in range(10):  # Collect more data points for better regime detection
            success = await self.data_service.update_price_history(self.symbols)
            if all(success.values()):
                self.logger.info(f"✅ Data collection {i+1}/10 successful")
            else:
                self.logger.warning(f"⚠️ Some data collection failed: {success}")
            
            await asyncio.sleep(6)  # 6 second intervals
        
        # Initial regime detection
        self.logger.info("🧠 Performing initial regime analysis...")
        regime_summary = self.trading_service.regime_service.get_regime_summary()
        
        self.logger.info(f"📈 Market Regime: {regime_summary['market_regime']}")
        self.logger.info(f"🔗 Correlation: {regime_summary['current_correlation']:.3f} ({regime_summary['correlation_regime']})")
        self.logger.info(f"📊 Adaptive Entry Threshold: {regime_summary['adaptive_thresholds']['entry_threshold']:.2f}")
        self.logger.info(f"🎯 Trading Recommendation: {regime_summary['recommendation']}")
        
        if regime_summary['recommendation'] != 'TRADE':
            self.logger.warning(f"⚠️ Initial conditions not optimal for trading: {regime_summary['adaptive_thresholds']['reasoning']}")
        
        self.logger.info("🚀 Enhanced setup completed successfully")
    
    async def run(self):
        try:
            await self.initialize()
            
            self.is_running = True
            self.main_loop_task = asyncio.create_task(self._enhanced_main_loop())
            
            self._log_startup_summary()
            
            await self.main_loop_task
            
        except Exception as e:
            self.logger.critical(f"💥 Critical error in enhanced bot execution: {e}")
            self.logger.critical(f"📍 Traceback: {traceback.format_exc()}")
            raise
        finally:
            await self.shutdown()
    
    def _log_startup_summary(self):
        """Log comprehensive startup summary"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 ENHANCED STATISTICAL ARBITRAGE BOT STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"📈 Trading Symbols: {self.symbols}")
        self.logger.info(f"⏱️ Update Interval: {self.update_interval}s")
        self.logger.info(f"🎯 Base Z-Entry: {self.config.trading.z_score_entry} (Adaptive)")
        self.logger.info(f"🎯 Base Z-Exit: {self.config.trading.z_score_exit} (Adaptive)")
        self.logger.info(f"💰 Base Position Size: {self.config.trading.position_size*100:.1f}% (Adaptive)")
        self.logger.info("🧠 Enhanced Features:")
        self.logger.info("   ✅ Market Regime Detection")
        self.logger.info("   ✅ Correlation Stability Filtering") 
        self.logger.info("   ✅ Adaptive Z-Score Thresholds")
        self.logger.info("   ✅ Transaction Cost Optimization")
        self.logger.info("   ✅ Signal Quality Tracking")
        self.logger.info("   ✅ Circuit Breakers & Risk Management")
        self.logger.info("=" * 80)
    
    async def _enhanced_main_loop(self):
        """Enhanced main loop with regime detection and cost optimization"""
        consecutive_errors = 0
        max_consecutive_errors = 5  # Reduced threshold for enhanced error handling
        
        last_regime_check = datetime.now()
        regime_check_interval = 300  # Check regime every 5 minutes
        
        while self.is_running:
            try:
                self.perf_monitor.start_timing("enhanced_main_loop_cycle")
                
                # Update market data
                success = await self._update_market_data()
                if not success:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.critical("💥 Too many consecutive data errors, emergency shutdown")
                        break
                    continue
                else:
                    consecutive_errors = 0
                
                # Enhanced trading logic with regime detection
                await self._process_enhanced_trading_logic()
                
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Periodic regime and health checks
                now = datetime.now()
                if (now - last_regime_check).total_seconds() > regime_check_interval:
                    await self._periodic_enhanced_checks()
                    last_regime_check = now
                
                # Performance logging
                cycle_time = self.perf_monitor.end_timing("enhanced_main_loop_cycle")
                
                if cycle_time > 30:
                    self.logger.warning(f"🐌 Slow enhanced cycle: {cycle_time:.2f}s")
                
                # Adaptive sleep based on market conditions
                sleep_time = await self._get_adaptive_sleep_time()
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"💥 Error in enhanced main loop (consecutive: {consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical("💥 Too many consecutive errors, emergency shutdown")
                    break
                
                # Enhanced backoff strategy
                sleep_time = min(300, 10 * (2 ** consecutive_errors))
                self.logger.info(f"⏰ Enhanced backoff: sleeping {sleep_time}s")
                await asyncio.sleep(sleep_time)
    
    async def _process_enhanced_trading_logic(self):
        """Enhanced trading logic with regime detection and cost optimization"""
        try:
            # Generate enhanced signal with all filters
            enhanced_signal = await self.trading_service.generate_enhanced_signal()
            self.total_signals_generated += 1
            
            # Log signal analysis
            if enhanced_signal.final_recommendation != "SKIP":
                self.logger.info(f"📊 Enhanced Signal Generated:")
                self.logger.info(f"   🎯 Type: {enhanced_signal.base_signal.signal_type.value}")
                self.logger.info(f"   📈 Z-Score: {enhanced_signal.base_signal.z_score:.3f}")
                self.logger.info(f"   🧠 Regime: {enhanced_signal.regime_analysis.get('market_regime', 'unknown')}")
                self.logger.info(f"   🔗 Correlation: {enhanced_signal.regime_analysis.get('current_correlation', 0):.3f}")
                self.logger.info(f"   💰 Adjusted Size: {enhanced_signal.position_size_adjusted:.4f}")
                self.logger.info(f"   ✅ Recommendation: {enhanced_signal.final_recommendation}")
            
            # Execute enhanced signal
            if enhanced_signal.final_recommendation in ["TRADE", "TRADE_SMALL"]:
                success = await self.trading_service.execute_enhanced_signal(enhanced_signal)
                
                if success:
                    self.signals_executed += 1
                    self.logger.info(f"✅ Enhanced signal executed successfully")
                else:
                    self.logger.error(f"❌ Enhanced signal execution failed")
            else:
                self.logger.debug(f"⏭️ Enhanced signal skipped: {'; '.join(enhanced_signal.reasoning)}")
            
            # Check for enhanced risk exits
            await self._check_enhanced_risk_exits()
            
        except Exception as e:
            self.logger.error(f"💥 Error in enhanced trading logic: {e}")
    
    async def _check_enhanced_risk_exits(self):
        """Enhanced risk exit checking with regime awareness"""
        try:
            if not self.trading_service.is_trading:
                return
            
            # Standard risk checks
            should_exit, reason = self.trading_service.check_stop_loss()
            if should_exit:
                self.logger.warning(f"🛑 Stop loss triggered: {reason}")
                await self.trading_service.force_exit_with_tracking("Stop loss")
                return
            
            should_exit, reason = self.trading_service.check_holding_time_limits()
            if should_exit:
                self.logger.info(f"⏰ Time-based exit triggered: {reason}")
                await self.trading_service.force_exit_with_tracking("Time limit")
                return
            
            # Enhanced regime-based exit
            regime_summary = self.trading_service.regime_service.get_regime_summary()
            if regime_summary['recommendation'] != 'TRADE':
                self.logger.warning(f"🧠 Regime-based exit triggered: {regime_summary['adaptive_thresholds']['reasoning']}")
                await self.trading_service.force_exit_with_tracking("Regime change")
                self.regime_changes_detected += 1
                return
            
            # Update tracking
            await self.trading_service.update_unrealized_pnl()
            
        except Exception as e:
            self.logger.error(f"💥 Error checking enhanced risk exits: {e}")
    
    async def _update_market_data(self) -> bool:
        """Enhanced market data update with validation"""
        try:
            results = await self.data_service.update_price_history(self.symbols)
            
            success_count = sum(1 for success in results.values() if success)
            self.logger.debug(f"📊 Market data update: {success_count}/{len(self.symbols)} successful")
            
            return success_count >= len(self.symbols) * 0.8
            
        except Exception as e:
            self.logger.error(f"💥 Failed to update market data: {e}")
            return False
    
    async def _update_risk_metrics(self):
        """Enhanced risk metrics update"""
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
            self.logger.error(f"💥 Error updating enhanced risk metrics: {e}")
    
    async def _periodic_enhanced_checks(self):
        """Comprehensive periodic health and regime checks"""
        try:
            self.logger.info("🔍 Running periodic enhanced checks...")
            
            # Health checks
            health_report = await health_monitor.run_health_checks()
            self._log_health_status(health_report)
            
            # Regime analysis summary
            regime_summary = self.trading_service.regime_service.get_regime_summary()
            self._log_regime_status(regime_summary)
            
            # Cost efficiency report
            cost_report = self.trading_service.cost_service.get_cost_efficiency_report()
            self._log_cost_efficiency(cost_report)
            
            # Circuit breaker status
            cb_summary = circuit_registry.get_health_summary()
            if cb_summary['overall_health'] != 'healthy':
                self.logger.warning(f"⚠️ Circuit breaker status: {cb_summary['overall_health']}")
            
            # Performance summary
            self._log_session_performance()
            
        except Exception as e:
            self.logger.error(f"💥 Error in periodic enhanced checks: {e}")
    
    def _log_health_status(self, health_report: Dict):
        """Log health status summary"""
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'unhealthy': '❌'
        }
        
        overall_emoji = status_emoji.get(health_report['overall_status'], '❓')
        self.logger.info(f"{overall_emoji} System Health: {health_report['overall_status'].upper()}")
        
        if health_report['overall_status'] != 'healthy':
            failed_checks = health_report.get('failed_checks', [])
            if failed_checks:
                self.logger.warning(f"⚠️ Failed checks: {', '.join(failed_checks)}")
    
    def _log_regime_status(self, regime_summary: Dict):
        """Log current regime status"""
        regime_emojis = {
            'mean_reverting': '📉',
            'trending': '📈', 
            'choppy': '🌊',
            'uncertain': '❓'
        }
        
        regime = regime_summary.get('market_regime', 'uncertain')
        emoji = regime_emojis.get(regime, '❓')
        
        self.logger.info(f"{emoji} Market Regime: {regime.upper()}")
        self.logger.info(f"🔗 Correlation: {regime_summary.get('current_correlation', 0):.3f} "
                        f"({regime_summary.get('correlation_regime', 'unknown')})")
        self.logger.info(f"🎯 Adaptive Entry Threshold: {regime_summary.get('adaptive_thresholds', {}).get('entry_threshold', 0):.2f}")
        
        recommendation = regime_summary.get('recommendation', 'PAUSE')
        rec_emoji = '✅' if recommendation == 'TRADE' else '⏸️'
        self.logger.info(f"{rec_emoji} Trading Recommendation: {recommendation}")
    
    def _log_cost_efficiency(self, cost_report: Dict):
        """Log cost efficiency summary"""
        if cost_report.get('status') == 'insufficient_data':
            return
        
        efficiency_score = cost_report.get('cost_efficiency_score', 0)
        score_emoji = '✅' if efficiency_score > 0.7 else '⚠️' if efficiency_score > 0.4 else '❌'
        
        self.logger.info(f"{score_emoji} Cost Efficiency Score: {efficiency_score:.3f}")
        self.logger.info(f"💰 Win Rate: {cost_report.get('win_rate', 0):.1%}")
        self.logger.info(f"💸 Avg Cost/Trade: {cost_report.get('avg_cost_per_trade', 0):.6f}")
    
    def _log_session_performance(self):
        """Log session performance summary"""
        uptime = datetime.now() - self.session_start_time
        execution_rate = (self.signals_executed / max(self.total_signals_generated, 1)) * 100
        
        self.logger.info(f"📊 Session Performance:")
        self.logger.info(f"   ⏱️ Uptime: {uptime}")
        self.logger.info(f"   📡 Signals Generated: {self.total_signals_generated}")
        self.logger.info(f"   ✅ Signals Executed: {self.signals_executed}")
        self.logger.info(f"   📈 Execution Rate: {execution_rate:.1f}%")
        self.logger.info(f"   🧠 Regime Changes: {self.regime_changes_detected}")
    
    async def _get_adaptive_sleep_time(self) -> float:
        """Get adaptive sleep time based on market conditions"""
        
        base_sleep = self.update_interval
        
        try:
            # Get regime info
            regime_summary = self.trading_service.regime_service.get_regime_summary()
            
            # Sleep longer in choppy/uncertain markets
            if regime_summary.get('market_regime') in ['choppy', 'uncertain']:
                return base_sleep * 1.5
            
            # Sleep shorter in good trading conditions
            if regime_summary.get('recommendation') == 'TRADE' and regime_summary.get('confidence', 0) > 0.8:
                return base_sleep * 0.8
            
        except Exception:
            pass
        
        return base_sleep
    
    async def emergency_shutdown(self, reason: str):
        """Enhanced emergency shutdown"""
        self.logger.critical(f"🚨 ENHANCED EMERGENCY SHUTDOWN: {reason}")
        
        try:
            # Force exit all positions with tracking
            if self.trading_service and self.trading_service.is_trading:
                await self.trading_service.force_exit_with_tracking(f"Emergency: {reason}")
            
            # Trigger enhanced risk controls
            if self.risk_service:
                self.risk_service.emergency_stop(reason)
            
            # Pause trading
            if self.trading_service:
                self.trading_service.pause_trading(reason, 24.0)  # 24 hour pause
            
        except Exception as e:
            self.logger.critical(f"💥 Error during emergency shutdown: {e}")
        
        finally:
            self.is_running = False
    
    async def shutdown(self):
        """Enhanced graceful shutdown"""
        self.logger.info("🛑 Shutting down Enhanced Statistical Arbitrage Bot...")
        
        self.is_running = False
        
        try:
            # Cancel main loop
            if self.main_loop_task and not self.main_loop_task.done():
                self.main_loop_task.cancel()
                try:
                    await self.main_loop_task
                except asyncio.CancelledError:
                    pass
            
            # Enhanced position closing with tracking
            if self.trading_service and self.trading_service.is_trading:
                self.logger.info("🔄 Closing positions with enhanced tracking...")
                await self.trading_service.force_exit_with_tracking("Bot shutdown")
            
            # Final enhanced reporting
            if self.trading_service:
                enhanced_stats = self.trading_service.get_enhanced_status()
                self.logger.info("📊 Final Enhanced Stats:")
                self.logger.info(f"   💰 Total PnL: {enhanced_stats['base_trading_stats']['total_pnl']:.4f}")
                self.logger.info(f"   📈 Win Rate: {enhanced_stats['base_trading_stats']['win_rate']:.1f}%")
                self.logger.info(f"   🧠 Final Regime: {enhanced_stats['regime_analysis']['market_regime']}")
                self.logger.info(f"   💸 Cost Efficiency: {enhanced_stats['cost_efficiency'].get('cost_efficiency_score', 0):.3f}")
            
            # Session summary
            self._log_session_performance()
            
        except Exception as e:
            self.logger.error(f"💥 Error during enhanced shutdown: {e}")
        
        self.logger.info("✅ Enhanced Statistical Arbitrage Bot shutdown complete")
    
    async def get_enhanced_status(self) -> dict:
        """Get comprehensive enhanced status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'uptime_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                'session_stats': {
                    'signals_generated': self.total_signals_generated,
                    'signals_executed': self.signals_executed,
                    'regime_changes_detected': self.regime_changes_detected,
                    'execution_rate': (self.signals_executed / max(self.total_signals_generated, 1)) * 100
                },
                'health': await health_monitor.run_health_checks(),
                'circuit_breakers': circuit_registry.get_health_summary(),
            }
            
            if self.trading_service:
                status['enhanced_trading'] = self.trading_service.get_enhanced_status()
            
            return status
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'is_running': self.is_running
            }


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Statistical Arbitrage Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--demo-mode', action='store_true', help='Run in demo mode (no real trading)')
    args = parser.parse_args()
    
    # Setup logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    if args.demo_mode:
        print("🎭 Demo mode enabled - no real trading will occur")
    
    bot = None
    try:
        bot = EnhancedStatArbBot(config_file=args.config)
        await bot.run()
    except KeyboardInterrupt:
        print("\n⚠️ Received interrupt signal")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bot:
            await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())