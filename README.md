# Statistical Arbitrage Trading Bot for Hyperliquid

A defensive statistical arbitrage trading bot that implements pairs trading between BTC/USD and ETH/USD on the Hyperliquid platform.

## Features

- **Statistical Arbitrage**: Uses mean reversion strategy based on price spread Z-scores
- **Risk Management**: Comprehensive risk controls including stop losses, position sizing, and drawdown limits
- **Real-time Monitoring**: Continuous price monitoring and logging
- **Defensive Design**: Built for security analysis and risk management

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your API credentials:
```bash
cp config_template.json config.json
# Edit config.json with your Hyperliquid API credentials
```

## Configuration

Edit `config.json` with your settings:

### API Credentials
- `account_address`: Your Hyperliquid public key
- `secret_key`: Your Hyperliquid private key  

### Position Sizing (50% Account Capital)
- `position_size`: 0.25 (25% per asset = 50% total exposure)
- `max_position_size`: 0.25 (Maximum 25% per asset)

### Strategy Parameters
- `z_score_entry`: 2.5 (Higher threshold for 50% exposure)
- `z_score_exit`: 0.3 (Quicker exits for risk management)
- `lookback_period`: 100 (Number of price points for analysis)

### Risk Management (Enhanced for 50% Exposure)
- `max_daily_loss`: 0.05 (5% daily loss limit)
- `stop_loss_pct`: 0.02 (2% stop loss - tighter control)
- `max_drawdown`: 0.10 (10% maximum drawdown)
- `max_open_positions`: 1 (One pairs trade at a time)
- `max_daily_trades`: 50 (Maximum trades per day)

### Cointegration Settings
- `correlation_threshold`: 0.7 (Minimum correlation for trading)
- `cointegration_check_hours`: 24 (Revalidate cointegration daily)

## Usage

Run the statistical arbitrage bot:

```bash
python hyperliquid_stat_arb.py
```

The bot will:
1. Connect to Hyperliquid API
2. Monitor BTC and ETH prices
3. Calculate spread statistics and Z-scores
4. Execute pairs trades when thresholds are met
5. Apply comprehensive risk management
6. Log all activities

## Strategy Overview

The bot implements a statistical arbitrage strategy:

1. **Data Collection**: Continuously monitors BTC and ETH prices
2. **Spread Calculation**: Calculates log price ratio between BTC and ETH
3. **Statistical Analysis**: Computes Z-score of current spread vs historical mean
4. **Signal Generation**: 
   - Enter trade when |Z-score| > entry threshold
   - Exit trade when |Z-score| < exit threshold
5. **Risk Management**: Applies position sizing, stop losses, and risk limits

## Files

- `hyperliquid_stat_arb.py`: Main trading bot
- `risk_manager.py`: Risk management module
- `config_template.json`: Configuration template
- `requirements.txt`: Python dependencies

## Position Sizing Example

**Example with 50% Capital:**
- Account balance: $10,000
- Position size: 0.25 = $2,500 per asset
- Total exposure: $5,000 (50% of account)
- Trade: SHORT $2,500 BTC + LONG $2,500 ETH

The sizes are equal in dollar value to maintain market neutrality in the pairs trade.

## Risk Warning for 50% Exposure

**High Risk Configuration**: Using 50% of account capital significantly increases risk. The configuration includes enhanced risk controls:

- **Tighter Stop Loss**: 2% (vs standard 5%)
- **Higher Entry Threshold**: Z-score 2.5 (vs standard 2.0)  
- **Quicker Exits**: Z-score 0.3 (vs standard 0.5)
- **Single Position**: Only one pairs trade at a time

**Recommendation**: Consider starting with lower exposure (10-20%) and gradually increasing as you gain experience.

## Risk Disclaimer

This is a defensive trading tool designed for statistical analysis and educational purposes. Trading cryptocurrencies involves substantial risk. Always test with small amounts and understand the risks involved.

## Security

- Never commit API credentials to version control
- Use strong API key security practices
- Monitor positions and logs regularly
- Set appropriate risk limits