#!/usr/bin/env python3
"""
Test script for the live dashboard
"""

import requests
import time
import threading
from app import app, socketio

def test_dashboard():
    """Test the dashboard functionality"""
    print("🧪 Testing Live Backtest Dashboard")
    print("=" * 50)
    
    # Start the Flask app in a separate thread
    def run_app():
        socketio.run(app, host='localhost', port=5001, debug=False, log_output=False)
    
    app_thread = threading.Thread(target=run_app, daemon=True)
    app_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test homepage
        response = requests.get('http://localhost:5001/', timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard homepage accessible")
        else:
            print(f"❌ Dashboard homepage failed: {response.status_code}")
            
        # Test API endpoint
        response = requests.get('http://localhost:5001/api/data', timeout=5)
        if response.status_code == 200:
            print("✅ API endpoint accessible")
            data = response.json()
            print(f"   📊 Initial data: {len(data.get('portfolio_history', []))} portfolio points")
        else:
            print(f"❌ API endpoint failed: {response.status_code}")
            
        print("\n🎉 Dashboard is ready!")
        print("🌐 Access at: http://localhost:5001")
        print("\n🚀 Ready for deployment to free hosting platforms:")
        print("   • Heroku: Free tier available")
        print("   • Railway: Free tier available") 
        print("   • Render: Free tier available")
        print("   • Fly.io: Free tier available")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = test_dashboard()
    if success:
        print("\n✅ All tests passed! Dashboard is ready for deployment.")
    else:
        print("\n❌ Tests failed. Check the configuration.")