#!/usr/bin/env python3
"""
Real-time Backtest Dashboard
Flask web application with Socket.IO for live backtest monitoring
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime
import asyncio
import queue
import logging
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from comprehensive_backtest import ComprehensiveBacktester

app = Flask(__name__)
app.config['SECRET_KEY'] = 'backtest_dashboard_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for data sharing
backtest_data = {
    'portfolio_history': [],
    'trades': [],
    'current_metrics': {},
    'z_scores': [],
    'market_regimes': [],
    'is_running': False,
    'progress': 0
}

data_queue = queue.Queue()
backtest_thread = None

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get current backtest data"""
    return jsonify(backtest_data)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'socketio': 'ready'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to backtest dashboard'})
    # Send current data to newly connected client
    emit('data_update', backtest_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_backtest')
def handle_start_backtest(data):
    """Start a new backtest"""
    global backtest_thread, backtest_data
    
    if backtest_data['is_running']:
        emit('error', {'message': 'Backtest already running'})
        return
    
    # Reset data
    backtest_data = {
        'portfolio_history': [],
        'trades': [],
        'current_metrics': {},
        'z_scores': [],
        'market_regimes': [],
        'is_running': True,
        'progress': 0
    }
    
    # Start backtest in separate thread
    backtest_thread = threading.Thread(target=run_backtest_async, args=(data.get('days', 30),))
    backtest_thread.daemon = True
    backtest_thread.start()
    
    emit('status', {'message': 'Backtest started'})

def run_backtest_async(days):
    """Run backtest in async context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_live_backtest(days))
    finally:
        loop.close()

async def run_live_backtest(days=30):
    """Run backtest with live updates to dashboard"""
    global backtest_data
    
    try:
        # Initialize backtester
        backtester = LiveBacktester()
        
        # Fetch data
        socketio.emit('status', {'message': f'Fetching {days} days of market data...'})
        data = await backtester.fetch_hyperliquid_data(days=days)
        
        # Run backtest with live updates
        await backtester.run_live_backtest(data, update_callback=update_dashboard)
        
        backtest_data['is_running'] = False
        socketio.emit('backtest_complete', {'message': 'Backtest completed successfully'})
        
    except Exception as e:
        backtest_data['is_running'] = False
        socketio.emit('error', {'message': f'Backtest failed: {str(e)}'})
        print(f"Backtest error: {e}")

def update_dashboard(portfolio_point, trade=None, metrics=None, progress=0):
    """Callback function to update dashboard with new data"""
    global backtest_data
    
    # Update portfolio history
    backtest_data['portfolio_history'].append(portfolio_point)
    
    # Update trades if provided
    if trade:
        backtest_data['trades'].append(trade)
    
    # Update current metrics
    if metrics:
        backtest_data['current_metrics'] = metrics
    
    # Update progress
    backtest_data['progress'] = progress
    
    # Emit update to all connected clients
    socketio.emit('data_update', {
        'portfolio_point': portfolio_point,
        'trade': trade,
        'metrics': metrics,
        'progress': progress,
        'total_trades': len(backtest_data['trades']),
        'current_value': portfolio_point.get('portfolio_value', 0)
    })

class LiveBacktester(ComprehensiveBacktester):
    """Extended backtester for live dashboard updates"""
    
    async def run_live_backtest(self, data, update_callback=None):
        """Run backtest with live updates"""
        self.logger.info("🚀 Starting live backtest with dashboard updates...")
        
        # Initialize capital
        initial_capital = 100000
        current_capital = initial_capital
        
        # Price and return histories
        btc_price_history = []
        eth_price_history = []
        btc_return_history = []
        eth_return_history = []
        
        # Position tracking
        current_position = None
        entry_time = None
        entry_capital = 0
        entry_btc_price = 0
        entry_eth_price = 0
        entry_beta = 1.0
        
        # Results tracking
        self.trades = []
        self.daily_trades = {}
        self.portfolio_history = []
        
        total_rows = len(data)
        
        # Main backtesting loop with live updates
        for idx, row in data.iterrows():
            timestamp = row['timestamp']
            btc_price = row['btc_price']
            eth_price = row['eth_price']
            
            # Update price histories
            btc_price_history.append(btc_price)
            eth_price_history.append(eth_price)
            
            # Update return histories
            if len(btc_price_history) >= 2:
                btc_ret = (btc_price - btc_price_history[-2]) / btc_price_history[-2]
                eth_ret = (eth_price - eth_price_history[-2]) / eth_price_history[-2]
                btc_return_history.append(btc_ret)
                eth_return_history.append(eth_ret)
            
            # Maintain lookback window
            if len(btc_price_history) > self.lookback:
                btc_price_history = btc_price_history[-self.lookback:]
                eth_price_history = eth_price_history[-self.lookback:]
            
            if len(btc_return_history) > self.beta_lookback:
                btc_return_history = btc_return_history[-self.beta_lookback:]
                eth_return_history = eth_return_history[-self.beta_lookback:]
            
            # Need minimum data points
            if len(btc_price_history) < self.lookback:
                continue
            
            # Calculate current market metrics
            market_regime = self.calculate_market_regime(btc_return_history, eth_return_history)
            beta = self.calculate_robust_beta(btc_return_history, eth_return_history)
            z_score = self.calculate_adaptive_z_score(btc_price_history, eth_price_history, market_regime)
            
            # Calculate current portfolio value
            if current_position is None:
                portfolio_value = current_capital
                unrealized_pnl = 0
                unrealized_pnl_pct = 0
            else:
                unrealized_pnl, unrealized_pnl_pct = self.calculate_trade_pnl(
                    current_position, entry_btc_price, entry_eth_price,
                    btc_price, eth_price, entry_beta, entry_capital
                )
                portfolio_value = current_capital + unrealized_pnl
            
            # Create portfolio point for dashboard
            portfolio_point = {
                'timestamp': timestamp.isoformat(),
                'btc_price': btc_price,
                'eth_price': eth_price,
                'z_score': z_score,
                'beta': beta,
                'position': current_position,
                'portfolio_value': portfolio_value,
                'cash': current_capital,
                'unrealized_pnl': unrealized_pnl,
                'market_regime': market_regime['regime'],
                'volatility': market_regime['volatility'],
                'correlation': market_regime['correlation'],
                'tradeable': market_regime['tradeable']
            }
            
            # TRADING LOGIC (same as original)
            trade_info = None
            
            # Entry logic
            if (current_position is None and 
                self.should_enter_trade(z_score, market_regime, timestamp)):
                
                # Calculate position sizes
                btc_dollar_size, eth_dollar_size = self.calculate_adaptive_position_size(
                    current_capital, market_regime, beta
                )
                
                # Calculate trading costs
                total_position_size = btc_dollar_size + eth_dollar_size
                trading_cost = total_position_size * self.fee
                
                # Deduct trading costs
                current_capital -= trading_cost
                
                # Store entry information
                entry_capital = btc_dollar_size
                entry_btc_price = btc_price
                entry_eth_price = eth_price
                entry_time = timestamp
                entry_beta = beta
                
                # Determine position direction
                if z_score > self.z_entry:
                    current_position = 'short_btc'
                    position_description = "SHORT BTC, LONG ETH"
                else:
                    current_position = 'long_btc'
                    position_description = "LONG BTC, SHORT ETH"
                
                # Update daily trade count
                date_key = timestamp.strftime('%Y-%m-%d')
                self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
                
                # Create trade info for dashboard
                trade_info = {
                    'timestamp': timestamp.isoformat(),
                    'action': 'ENTER',
                    'position': position_description,
                    'z_score': z_score,
                    'beta': beta,
                    'capital_at_risk': entry_capital,
                    'btc_price': btc_price,
                    'eth_price': eth_price,
                    'trading_cost': trading_cost,
                    'remaining_capital': current_capital,
                    'market_regime': market_regime['regime']
                }
                
                self.trades.append(trade_info)
            
            # Exit logic
            elif current_position is not None:
                should_exit, exit_reason = self.should_exit_trade(
                    z_score, entry_time, timestamp, unrealized_pnl_pct, current_position
                )
                
                if should_exit:
                    # Calculate final P&L
                    final_pnl, final_return_pct = self.calculate_trade_pnl(
                        current_position, entry_btc_price, entry_eth_price,
                        btc_price, eth_price, entry_beta, entry_capital
                    )
                    
                    # Calculate exit trading costs
                    exit_cost = entry_capital * self.fee
                    
                    # Update capital
                    current_capital += final_pnl - exit_cost
                    
                    # Calculate holding time
                    holding_minutes = (timestamp - entry_time).total_seconds() / 60
                    
                    # Create exit trade info
                    trade_info = {
                        'timestamp': timestamp.isoformat(),
                        'action': 'EXIT',
                        'position': current_position,
                        'z_score': z_score,
                        'beta': beta,
                        'pnl': final_pnl,
                        'return_pct': final_return_pct,
                        'trading_cost': exit_cost,
                        'final_capital': current_capital,
                        'holding_minutes': holding_minutes,
                        'exit_reason': exit_reason,
                        'btc_price': btc_price,
                        'eth_price': eth_price,
                        'market_regime': market_regime['regime']
                    }
                    
                    self.trades.append(trade_info)
                    
                    # Reset position
                    current_position = None
                    entry_capital = 0
            
            # Calculate current metrics
            current_metrics = {
                'total_return': (portfolio_value - initial_capital) / initial_capital,
                'total_pnl': portfolio_value - initial_capital,
                'num_trades': len([t for t in self.trades if t['action'] == 'ENTER']),
                'current_drawdown': (portfolio_value - max([p.get('portfolio_value', initial_capital) for p in self.portfolio_history] + [initial_capital])) / max([p.get('portfolio_value', initial_capital) for p in self.portfolio_history] + [initial_capital])
            }
            
            # Update dashboard every 100 data points or on trade
            progress = (idx / total_rows) * 100
            if idx % 100 == 0 or trade_info or idx == total_rows - 1:
                if update_callback:
                    update_callback(portfolio_point, trade_info, current_metrics, progress)
            
            # Store portfolio history
            self.portfolio_history.append(portfolio_point)
            
            # Small delay to make updates visible
            if idx % 1000 == 0:
                await asyncio.sleep(0.01)
        
        self.logger.info("✅ Live backtest completed!")

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting Backtest Dashboard - Updated Version")
    print("📊 Access the dashboard at: http://localhost:5000")
    print("🌐 Dashboard will be accessible to others on your network")
    
    # Run the Flask-SocketIO app
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)