#!/usr/bin/env python3
"""
Chart Viewer for Statistical Arbitrage Backtesting
Run this to generate and view P&L charts
"""

from simple_backtest import SimpleBacktester
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate backtest charts')
    parser.add_argument('--days', type=int, default=3, help='Number of days to backtest (default: 3)')
    parser.add_argument('--no-show', action='store_true', help='Save charts without displaying')
    
    args = parser.parse_args()
    
    print(f"🔄 Running {args.days}-day Statistical Arbitrage Backtest...")
    print("="*60)
    
    # Initialize backtester
    backtester = SimpleBacktester()
    
    # Generate data
    data = backtester.generate_sample_data(days=args.days)
    print(f"📊 Generated {len(data)} data points ({args.days} days)")
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    # Print summary
    print(f"\n💰 BACKTEST SUMMARY:")
    print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"   Final Value:     ${results['final_value']:,.2f}")
    print(f"   Total Return:    {results['total_return']*100:.2f}%")
    print(f"   Number of Trades: {results['num_trades']}")
    print(f"   Win Rate:        {results['win_rate']*100:.1f}%")
    
    # Generate charts
    print(f"\n📈 Generating comprehensive P&L analysis charts...")
    
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    
    backtester.plot_pnl_chart(results)
    
    print(f"\n✅ Charts saved:")
    print(f"   📊 backtest_pnl_analysis.png - Main analysis dashboard")
    print(f"   📈 individual_trade_pnl.png - Trade-by-trade P&L")
    
    if not args.no_show:
        print(f"\n💡 Charts are now displayed. Close the chart windows to continue.")
    
    return results

if __name__ == "__main__":
    results = main()