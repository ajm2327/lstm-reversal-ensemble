# clients.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from config import ALPACA_API_KEY, ALPACA_API_SECRET, GOOGLE_API_KEY

# Create clients once at module level
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

trading_client = TradingClient(
    api_key=ALPACA_API_KEY, 
    secret_key=ALPACA_API_SECRET, 
    paper=True
)
