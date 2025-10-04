from pybit.unified_trading import HTTP

# Create a session for testnet or live environment (set testnet=True for demo)
session = HTTP(testnet=False)

# Fetch recent public trade history for BTCUSDT spot category, limiting to 5 records
response = session.get_public_trade_history(category="spot", symbol="BTCUSDT", limit=5)

print(response)
