# Credential Setup Instructions

## Security First! 🔐

This project is configured to keep your trading credentials secure and **never commit them to Git**.

## Expected Credential Location

The script looks for your Hyperliquid credentials at:
```
C:\Users\16473\Desktop\Trading\hyperliquid\trade_api.json
```

## Required Format

Your `trade_api.json` should contain:
```json
{
    "account_address": "your_hyperliquid_public_key",
    "secret_key": "your_hyperliquid_private_key"
}
```

## Security Features

✅ **Credential folder excluded from Git** via `.gitignore`  
✅ **No credentials in source code**  
✅ **Local-only credential access**  
✅ **Clear error messages** if credentials not found  

## Important Notes

- **Never** commit credential files to version control
- **Never** share your `trade_api.json` file
- Keep your credentials in the designated folder only
- The script will fail gracefully if credentials are missing

## Running the Bot

Once credentials are properly set up, simply run:
```bash
python3 hyperliquid_stat_arb.py
```

The bot will automatically load credentials from the secure location.