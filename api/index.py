from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import pandas as pd
import requests
import logging
from dotenv import load_dotenv
import os
from functools import lru_cache

load_dotenv()

PASSWORD = os.getenv('PASSWORD')


# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trading_bot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
class Position(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    buy_price = db.Column(db.Float, nullable=False)
    target_price = db.Column(db.Float, nullable=False)
    buy_date = db.Column(db.DateTime, nullable=False)
    is_open = db.Column(db.Boolean, default=True)

class TriggeredBuy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    trigger_date = db.Column(db.DateTime, nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

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
            if symbol not in stocks:
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
        if (TriggeredBuy.query.filter_by(symbol=symbol).first() or
            Position.query.filter_by(symbol=symbol, is_open=True).first()):
            continue

        data = stock_data.get(symbol)
        if data and check_buy_condition(data):
            # Record triggered buy condition
            new_trigger = TriggeredBuy(symbol=symbol, trigger_date=datetime.utcnow())
            db.session.add(new_trigger)

            # Open a new position
            buy_price = data['current_price']
            target_price = buy_price * 1.005  # Target 0.5% increase
            new_position = Position(
                symbol=symbol,
                buy_price=buy_price,
                target_price=target_price,
                buy_date=datetime.utcnow()
            )
            db.session.add(new_position)
            db.session.commit()
            logger.info(f"Bought {symbol} at {buy_price:.2f}, target {target_price:.2f}")

    # Check for sell opportunities
    open_positions = Position.query.filter_by(is_open=True).all()
    symbols_to_fetch = [position.symbol for position in open_positions]
    sell_data = fetch_multiple_stock_data(symbols_to_fetch)

    for position in open_positions:
        data = sell_data.get(position.symbol)
        if data and data['current_price'] >= position.target_price:
            # Close the position
            position.is_open = False
            db.session.commit()
            logger.info(f"Sold {position.symbol} at {data['current_price']:.2f}")

            # Remove from triggered buys
            triggered_buy = TriggeredBuy.query.filter_by(symbol=position.symbol).first()
            if triggered_buy:
                db.session.delete(triggered_buy)
                db.session.commit()




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

if __name__ == '__main__':
    app.run(debug=True)