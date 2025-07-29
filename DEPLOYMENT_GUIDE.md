# 🚀 Live Backtest Dashboard - Deployment Guide

## Overview
This is a real-time web dashboard for monitoring statistical arbitrage backtests with live updates and interactive charts.

## Features
- ✅ Real-time backtest monitoring with Socket.IO
- ✅ 4 interactive charts matching comprehensive_backtest_results.png:
  - Portfolio Value Over Time
  - Cumulative P&L 
  - Z-Score Analysis
  - Beta Analysis
- ✅ Real-time trade table and metrics
- ✅ Progress tracking and status updates
- ✅ Responsive design with dark theme

## Free Deployment Options

### 1. Heroku (Recommended - Free Tier Available)

**Step 1: Install Heroku CLI**
```bash
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

**Step 2: Deploy to Heroku**
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-backtest-dashboard

# Set Python version
heroku stack:set heroku-24

# Deploy
git add .
git commit -m "Deploy backtest dashboard"
git push heroku main

# Open your app
heroku open
```

**One-Click Deploy:**
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

### 2. Railway (Free Tier)

**Step 1: Connect GitHub**
- Go to [Railway.app](https://railway.app)
- Connect your GitHub account
- Select this repository

**Step 2: Deploy**
```bash
# Railway will automatically detect Python and use Procfile
# No additional configuration needed
```

### 3. Render (Free Tier)

**Step 1: Create Web Service**
- Go to [Render.com](https://render.com)
- Create new Web Service from GitHub
- Select this repository

**Step 2: Configuration**
```
Build Command: pip install -r requirements.txt
Start Command: gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app
```

### 4. Fly.io (Free Tier)

**Step 1: Install Fly CLI**
```bash
# Download from https://fly.io/docs/getting-started/installing-flyctl/
```

**Step 2: Deploy**
```bash
flyctl auth login
flyctl launch
flyctl deploy
```

## Local Development

**Step 1: Setup Environment**
```bash
python -m venv backtest_env
source backtest_env/bin/activate  # Linux/Mac
# or
backtest_env\Scripts\activate     # Windows

pip install -r requirements.txt
```

**Step 2: Run Dashboard**
```bash
python app.py
```

**Step 3: Access Dashboard**
- Open http://localhost:5000
- Dashboard is accessible on your network at http://YOUR_IP:5000

## Usage

1. **Start Dashboard**: Access the web interface
2. **Configure Backtest**: Set number of days (1-90)
3. **Run Backtest**: Click "Start Backtest" button  
4. **Monitor Progress**: Watch real-time updates on charts and metrics
5. **View Results**: Analyze performance in real-time

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │◄──►│  Flask + SocketIO │◄──►│  Backtest Engine │
│   (Dashboard)   │    │   (Real-time)     │    │  (Hyperliquid)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Files

- `app.py` - Flask application with SocketIO
- `templates/dashboard.html` - Web dashboard interface
- `comprehensive_backtest.py` - Enhanced backtest engine
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku deployment configuration

## Environment Variables

No environment variables required for basic operation. All configuration is handled through the web interface.

## Troubleshooting

**Issue: Charts not updating**
- Check browser console for errors
- Ensure WebSocket connection is established
- Verify firewall settings

**Issue: Deployment fails**
- Check requirements.txt for version conflicts
- Verify Procfile syntax
- Check deployment logs

**Issue: Real-time updates not working**
- Ensure eventlet is installed
- Check Socket.IO version compatibility
- Verify CORS settings

## Performance Notes

- Dashboard handles up to 1000 data points per chart
- Uses data decimation for large datasets
- Optimized for real-time updates without memory leaks

## Security

- No authentication required for demo purposes
- Add authentication for production use
- CORS enabled for development (restrict in production)

## Support

For issues or questions:
1. Check the deployment logs
2. Verify all requirements are installed
3. Test locally first before deploying

---

**🎉 Your backtest dashboard will be live and accessible to anyone with the URL!**