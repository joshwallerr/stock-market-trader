from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient, DESCENDING
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import pandas as pd
import requests
import logging
from dotenv import load_dotenv
import os
from functools import lru_cache
from io import StringIO



# Load environment variables
load_dotenv()

PASSWORD = os.getenv('PASSWORD')
MONGODB_URI = os.getenv('MONGODB_URI')

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MongoDB client
client = MongoClient(MONGODB_URI)
db = client.trading_bot  # Database name

# Define collections
positions_col = db.positions
triggered_buys_col = db.triggered_buys
trades_col = db.trades

# Ensure unique index on symbol in triggered_buys
triggered_buys_col.create_index('symbol', unique=True)

# Indexes for trades collection
trades_col.create_index([('timestamp', DESCENDING)])


@app.route('/dashboard')
def dashboard():
    # Fetch trade history
    trades = list(trades_col.find().sort('timestamp', DESCENDING))

    # Fetch open positions
    open_positions = list(positions_col.find({'is_open': True}))

    # Calculate average rates of return
    avg_returns = calculate_average_returns()

    return render_template('dashboard.html', trades=trades, open_positions=open_positions, avg_returns=avg_returns)

@app.route('/api/trades')
def api_trades():
    trades = list(trades_col.find().sort('timestamp', 1))
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
            'action': trade['action'],
            'symbol': trade['symbol'],
            'price': trade['price'],
            'timestamp': trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        })

        # Aggregate buy and sell volumes over time (e.g., per day)
        timestamp = trade['timestamp'].strftime('%Y-%m-%d')
        if timestamp not in trade_data['timestamps']:
            trade_data['timestamps'].append(timestamp)
            trade_data['buyVolumes'].append(0)
            trade_data['sellVolumes'].append(0)
            trade_data['profitLoss'].append(profit)

        index = trade_data['timestamps'].index(timestamp)

        if trade['action'] == 'BUY':
            trade_data['buyVolumes'][index] += 1
            open_buys[trade['symbol']] = trade['price']
        elif trade['action'] == 'SELL' and trade['symbol'] in open_buys:
            buy_price = open_buys.pop(trade['symbol'])
            profit += (trade['price'] - buy_price)
            trade_data['sellVolumes'][index] += 1

        trade_data['profitLoss'][index] = profit

    return jsonify(trade_data)


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
    if not symbols:
        logger.warning("No symbols provided to fetch_multiple_stock_data.")
        return {}
    
    try:
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        start_date = now - timedelta(days=1)
        end_date = now

        # Fetch data for all symbols at once
        stocks = yf.download(
            tickers=' '.join(symbols), 
            interval='1m', 
            start=start_date, 
            end=end_date, 
            group_by='ticker',
            threads=True,  # Enable multi-threading for faster downloads
            progress=False  # Disable progress bar for cleaner logs
        )

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
            # Use .iloc for positional indexing to avoid FutureWarning
            current_price = hist['Close'].iloc[-1]
            opening_price = hist['Open'].iloc[0]
            intraday_low = hist['Low'].min()
            
            # Check for nan values
            if pd.isna(current_price) or pd.isna(opening_price) or pd.isna(intraday_low):
                logger.warning(f"Incomplete data for {symbol}. Data: Current Price={current_price}, Opening Price={opening_price}, Intraday Low={intraday_low}")
                continue

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
        # Check if there's a triggered buy or an open position
        if (triggered_buys_col.find_one({'symbol': symbol}) or
            positions_col.find_one({'symbol': symbol, 'is_open': True})):
            continue

        data = stock_data.get(symbol)
        if data and check_buy_condition(data):
            # Record triggered buy condition
            new_trigger = {
                'symbol': symbol,
                'trigger_date': datetime.utcnow()
            }
            try:
                triggered_buys_col.insert_one(new_trigger)
            except Exception as e:
                logger.error(f"Error inserting TriggeredBuy for {symbol}: {e}")
                continue

            # Open a new position
            buy_price = data['current_price']
            target_price = buy_price * 1.005  # Target 0.5% increase

            if pd.isna(buy_price) or pd.isna(target_price):
                logger.warning(f"Invalid buy_price or target_price for {symbol}. Skipping trade.")
                continue

            new_position = {
                'symbol': symbol,
                'buy_price': buy_price,
                'target_price': target_price,
                'buy_date': datetime.utcnow(),
                'is_open': True
            }
            try:
                positions_col.insert_one(new_position)
            except Exception as e:
                logger.error(f"Error inserting Position for {symbol}: {e}")
                continue

            # Log the buy trade
            buy_trade = {
                'symbol': symbol,
                'action': 'BUY',
                'price': buy_price,
                'timestamp': datetime.utcnow()
            }
            try:
                trades_col.insert_one(buy_trade)
            except Exception as e:
                logger.error(f"Error logging BUY trade for {symbol}: {e}")
                continue

            logger.info(f"Bought {symbol} at {buy_price:.2f}, target {target_price:.2f}")

    # Check for sell opportunities
    open_positions = list(positions_col.find({'is_open': True}))
    symbols_to_fetch = [position['symbol'] for position in open_positions]

    if symbols_to_fetch:
        sell_data = fetch_multiple_stock_data(symbols_to_fetch)
    else:
        sell_data = {}
        logger.info("No open positions to check for sell opportunities.")

    for position in open_positions:
        symbol = position['symbol']
        data = sell_data.get(symbol)
        if data and data['current_price'] >= position['target_price']:
            # Close the position
            try:
                positions_col.update_one(
                    {'_id': position['_id']},
                    {'$set': {'is_open': False}}
                )
            except Exception as e:
                logger.error(f"Error closing Position for {symbol}: {e}")
                continue

            logger.info(f"Sold {symbol} at {data['current_price']:.2f}")

            # Log the sell trade
            sell_trade = {
                'symbol': symbol,
                'action': 'SELL',
                'price': data['current_price'],
                'timestamp': datetime.utcnow()
            }
            try:
                trades_col.insert_one(sell_trade)
            except Exception as e:
                logger.error(f"Error logging SELL trade for {symbol}: {e}")
                continue

            # Remove from triggered buys
            try:
                triggered_buys_col.delete_one({'symbol': symbol})
            except Exception as e:
                logger.error(f"Error deleting TriggeredBuy for {symbol}: {e}")
                continue

def check_buy_condition(data):
    """
    Checks if the stock has decreased by more than 5% from opening to intraday low.
    
    Args:
        data (dict): Dictionary containing 'opening_price' and 'intraday_low'.
    
    Returns:
        bool: True if condition met, else False.
    """
    try:
        # Ensure no nan values are present
        if any(pd.isna(v) for v in data.values()):
            logger.warning(f"Data contains NaN values: {data}")
            return False

        drop_percentage = ((data['opening_price'] - data['intraday_low']) / data['opening_price']) * 100
        return drop_percentage >= 5  # Threshold set to 5%
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

        tables = pd.read_html(StringIO(response.text))

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
        # Fetch trades within the period
        trades = list(trades_col.find({'timestamp': {'$gte': start_date}}).sort('timestamp', 1))

        buy_prices = {}
        returns = []

        for trade in trades:
            if trade['action'] == 'BUY':
                buy_prices[trade['symbol']] = trade['price']
            elif trade['action'] == 'SELL' and trade['symbol'] in buy_prices:
                buy_price = buy_prices.pop(trade['symbol'])
                sell_price = trade['price']
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