from flask import Flask, request, jsonify, render_template
from flask_mongoengine import MongoEngine
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import pandas as pd
import requests
import logging
from dotenv import load_dotenv
import os
from functools import lru_cache

# Load environment variables
load_dotenv()

PASSWORD = os.getenv('PASSWORD')

# Initialize Flask app
app = Flask(__name__)

# Configure MongoDB
app.config['MONGODB_SETTINGS'] = {
    'db': 'OKh5hxnPlLbMm2Yb',  # Database name
    'host': os.getenv('MONGODB_URI')  # MongoDB connection string
}

# Initialize MongoEngine
db = MongoEngine(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models using MongoEngine
class Position(db.Document):
    symbol = db.StringField(required=True, max_length=10)
    buy_price = db.FloatField(required=True)
    target_price = db.FloatField(required=True)
    buy_date = db.DateTimeField(required=True)
    is_open = db.BooleanField(default=True)

    meta = {
        'collection': 'positions'
    }

class TriggeredBuy(db.Document):
    symbol = db.StringField(required=True, unique=True, max_length=10)
    trigger_date = db.DateTimeField(required=True)

    meta = {
        'collection': 'triggered_buys',
        'indexes': [
            {'fields': ['symbol'], 'unique': True}
        ]
    }

class Trade(db.Document):
    symbol = db.StringField(required=True, max_length=10)
    action = db.StringField(required=True, choices=['BUY', 'SELL'])
    price = db.FloatField(required=True)
    timestamp = db.DateTimeField(required=True, default=datetime.utcnow)

    meta = {
        'collection': 'trades'
    }

# Create database tables (collections)
with app.app_context():
    db.create_collection(Position)
    db.create_collection(TriggeredBuy)
    db.create_collection(Trade)

# Dashboard route
@app.route('/dashboard')
def dashboard():
    # Fetch trade history
    trades = Trade.objects.order_by('-timestamp')

    # Fetch open positions
    open_positions = Position.objects(is_open=True)

    # Calculate average rates of return
    avg_returns = calculate_average_returns()

    return render_template('dashboard.html', trades=trades, open_positions=open_positions, avg_returns=avg_returns)

# API endpoint for trade data
@app.route('/api/trades')
def api_trades():
    trades = Trade.objects.order_by('timestamp')
    trade_data = {
        'trades': [],
        'timestamps': [],
        'buyVolumes': [],
        'sellVolumes': [],
        'profitLoss': []
    }

    profit = 0.0
    # Initialize a dictionary to track open buys
    open_buys = {}

    for trade in trades:
        trade_data['trades'].append({
            'action': trade.action,
            'symbol': trade.symbol,
            'price': trade.price,
            'timestamp': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

        # Aggregate buy and sell volumes over time (e.g., per day)
        timestamp = trade.timestamp.strftime('%Y-%m-%d')
        if timestamp not in trade_data['timestamps']:
            trade_data['timestamps'].append(timestamp)
            trade_data['buyVolumes'].append(0)
            trade_data['sellVolumes'].append(0)
            trade_data['profitLoss'].append(profit)

        index = trade_data['timestamps'].index(timestamp)

        if trade.action == 'BUY':
            trade_data['buyVolumes'][index] += 1
            open_buys[trade.symbol] = trade.price
        elif trade.action == 'SELL' and trade.symbol in open_buys:
            buy_price = open_buys.pop(trade.symbol)
            profit += (trade.price - buy_price)
            trade_data['sellVolumes'][index] += 1

        trade_data['profitLoss'][index] = profit

    return jsonify(trade_data)

# Endpoint to run trades
@app.route('/run-trades', methods=['POST'])
def run_trades():
    data = request.get_json()
    if not data or data.get('password') != PASSWORD:
        logger.warning("Unauthorized access attempt.")
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        run_trading_logic()
        logger.info("Trading logic executed successfully.")
        return jsonify({'status': 'Trading logic executed successfully'}), 200
    except Exception as e:
        logger.error(f"Error executing trading logic: {e}")
        return jsonify({'error': str(e)}), 500

def fetch_multiple_stock_data(symbols):
    try:
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        start_date = now - timedelta(days=1)
        end_date = now

        # Fetch data for all symbols at once
        stocks = yf.download(tickers=' '.join(symbols), interval='1m', start=start_date, end=end_date, group_by='ticker')

        stock_data = {}
        for symbol in symbols:
            # Check if the symbol exists in the fetched data
            if symbol not in stocks.columns.levels[0]:
                logger.warning(f"No data for {symbol}")
                continue
            hist = stocks[symbol]
            if hist.empty:
                logger.warning(f"No data for {symbol}")
                continue
            current_price = hist['Close'][-1]
            opening_price = hist['Open'][0]
            intraday_low = hist['Low'].min()
            stock_data[symbol] = {
                'current_price': current_price,
                'opening_price': opening_price,
                'intraday_low': intraday_low
            }
        return stock_data
    except Exception as e:
        logger.error(f"Error fetching multiple stock data: {e}")
        return {}

def is_market_open():
    """
    Checks if the US stock market is currently open.

    Returns:
        bool: True if market is open, else False.
    """
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    # US stock markets are typically open from 9:30 AM to 4:00 PM EST
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close

@lru_cache(maxsize=1)
def get_sp500_symbols_cached():
    return get_sp500_symbols()

def run_trading_logic():
    if not is_market_open():
        logger.info("Market is closed. Skipping trading logic.")
        return

    sp500_symbols = get_sp500_symbols_cached()
    if not sp500_symbols:
        logger.error("No S&P 500 symbols fetched. Aborting trading logic.")
        return

    # Fetch all stock data at once
    stock_data = fetch_multiple_stock_data(sp500_symbols)

    # Check for new buy opportunities
    for symbol in sp500_symbols:
        if (TriggeredBuy.objects(symbol=symbol).first() or
            Position.objects(symbol=symbol, is_open=True).first()):
            continue

        data = stock_data.get(symbol)
        if data and check_buy_condition(data):
            # Record triggered buy condition
            new_trigger = TriggeredBuy(
                symbol=symbol,
                trigger_date=datetime.utcnow()
            )
            new_trigger.save()

            # Open a new position
            buy_price = data['current_price']
            target_price = buy_price * 1.005  # Target 0.5% increase
            new_position = Position(
                symbol=symbol,
                buy_price=buy_price,
                target_price=target_price,
                buy_date=datetime.utcnow()
            )
            new_position.save()

            # Log the buy trade
            buy_trade = Trade(
                symbol=symbol,
                action='BUY',
                price=buy_price,
                timestamp=datetime.utcnow()
            )
            buy_trade.save()

            logger.info(f"Bought {symbol} at {buy_price:.2f}, target {target_price:.2f}")

    # Check for sell opportunities
    open_positions = Position.objects(is_open=True)
    symbols_to_fetch = [position.symbol for position in open_positions]
    sell_data = fetch_multiple_stock_data(symbols_to_fetch)

    for position in open_positions:
        data = sell_data.get(position.symbol)
        if data and data['current_price'] >= position.target_price:
            # Close the position
            position.update(set__is_open=False)
            logger.info(f"Sold {position.symbol} at {data['current_price']:.2f}")

            # Log the sell trade
            sell_trade = Trade(
                symbol=position.symbol,
                action='SELL',
                price=data['current_price'],
                timestamp=datetime.utcnow()
            )
            sell_trade.save()

            # Remove from triggered buys
            TriggeredBuy.objects(symbol=position.symbol).delete()
            logger.info(f"Removed TriggeredBuy for {position.symbol}")

def check_buy_condition(data):
    """
    Checks if the stock has decreased by more than 10% from opening to intraday low.

    Args:
        data (dict): Dictionary containing 'opening_price' and 'intraday_low'.

    Returns:
        bool: True if condition met, else False.
    """
    try:
        drop_percentage = ((data['opening_price'] - data['intraday_low']) / data['opening_price']) * 100
        return drop_percentage >= 10
    except Exception as e:
        logger.error(f"Error checking buy condition: {e}")
        return False

def get_sp500_symbols():
    """
    Fetches the current list of S&P 500 symbols from Wikipedia.

    Returns:
        list: A list of ticker symbols.
    """
    try:
        # URL of the Wikipedia page containing S&P 500 companies
        wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

        # Use pandas to read all tables from the Wikipedia page
        response = requests.get(wiki_url)
        response.raise_for_status()  # Raise an error for bad status codes

        tables = pd.read_html(response.text)

        # The first table usually contains the list of S&P 500 companies
        sp500_table = tables[0]

        # Extract the ticker symbols
        symbols = sp500_table['Symbol'].tolist()

        # Clean symbols (some have dots which yfinance expects as dashes)
        symbols = [symbol.replace('.', '-') for symbol in symbols]

        logger.info(f"Fetched {len(symbols)} S&P 500 symbols.")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {e}")
        return []

def calculate_average_returns():
    avg_returns = {}
    now = datetime.utcnow()

    periods = {
        'daily': now - timedelta(days=1),
        'weekly': now - timedelta(weeks=1),
        'monthly': now - timedelta(days=30),
        'yearly': now - timedelta(days=365)
    }

    for period_name, start_date in periods.items():
        trades = Trade.objects(timestamp__gte=start_date)
        buy_prices = {}
        returns = []

        for trade in trades:
            if trade.action == 'BUY':
                buy_prices[trade.symbol] = trade.price
            elif trade.action == 'SELL' and trade.symbol in buy_prices:
                buy_price = buy_prices.pop(trade.symbol)
                sell_price = trade.price
                ret = ((sell_price - buy_price) / buy_price) * 100
                returns.append(ret)

        if returns:
            avg = sum(returns) / len(returns)
        else:
            avg = 0.0

        avg_returns[period_name] = round(avg, 2)

    return avg_returns

if __name__ == '__main__':
    app.run(debug=True)