from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient, DESCENDING
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import pandas as pd
import requests
import logging
from dotenv import load_dotenv
import os
from functools import lru_cache
from io import StringIO
import time

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
db = client.trading_bot_testing  # Database name

# Define collections
positions_col = db.positions
triggered_buys_col = db.triggered_buys
trades_col = db.trades
portfolio_col = db.portfolio  # New collection for portfolio

# Ensure unique index on symbol in triggered_buys
triggered_buys_col.create_index('symbol', unique=True)

# Indexes for trades collection
trades_col.create_index([('timestamp', DESCENDING)])

# Initialize portfolio
def initialize_portfolio():
    portfolio = portfolio_col.find_one({'_id': 'portfolio'})
    if not portfolio:
        portfolio = {
            '_id': 'portfolio',
            'starting_cash': 250.0,  # Starting cash value
            'current_cash': 250.0,    # Current cash balance
            'created_at': datetime.utcnow()
        }
        portfolio_col.insert_one(portfolio)
        logger.info("Initialized portfolio with $250 starting cash.")
    else:
        logger.info("Portfolio already initialized.")

initialize_portfolio()

@app.route('/dashboard')
def dashboard():
    # Fetch trade history
    trades = list(trades_col.find().sort('timestamp', DESCENDING))

    # Fetch open positions
    open_positions = list(positions_col.find({'is_open': True}))

    # Fetch portfolio data
    portfolio = portfolio_col.find_one({'_id': 'portfolio'})
    if portfolio:
        starting_cash = portfolio.get('starting_cash', 0.0)
        current_cash = portfolio.get('current_cash', 0.0)
    else:
        starting_cash = 0.0
        current_cash = 0.0

    # Calculate average rates of return
    avg_returns = calculate_average_returns()

    # Calculate total portfolio value
    total_invested = 0.0
    portfolio_value = current_cash

    for position in open_positions:
        symbol = position['symbol']
        shares = position.get('shares', 0.0)
        current_price = fetch_current_price(symbol)
        position['current_price'] = current_price  # Attach current_price to position
        if current_price is not None:
            position_value = shares * current_price
            portfolio_value += position_value
            total_invested += shares * position['buy_price']

    # Calculate profit/loss
    profit_loss = portfolio_value - starting_cash

    return render_template(
        'dashboard.html', 
        trades=trades, 
        open_positions=open_positions, 
        avg_returns=avg_returns,
        starting_cash=starting_cash,
        current_cash=current_cash,
        total_invested=total_invested,
        portfolio_value=round(portfolio_value, 2),
        profit_loss=round(profit_loss, 2)
    )

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
            'shares': trade['shares'],
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

def fetch_multiple_stock_data(symbols, retries=3, delay=5):
    if not symbols:
        logger.warning("No symbols provided to fetch_multiple_stock_data.")
        return {}
    
    try:
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        start_date = (now - timedelta(days=2)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')

        stock_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1m')
                if data.empty:
                    logger.warning(f"No data fetched for {symbol}.")
                    continue

                # Reset index to access datetime
                data = data.reset_index()

                # Ensure at least two days of data
                data['Date'] = data['Datetime'].dt.date
                grouped = data.groupby('Date')
                if len(grouped) < 2:
                    logger.warning(f"Not enough days of data for {symbol}.")
                    continue

                days = sorted(grouped.groups.keys())
                previous_day = days[-2]
                current_day = days[-1]

                previous_day_data = grouped.get_group(previous_day)
                current_day_data = grouped.get_group(current_day)

                previous_close = previous_day_data['Close'].iloc[-1]
                opening_price = current_day_data['Open'].iloc[0]
                intraday_low = current_day_data['Low'].min()
                current_price = current_day_data['Close'].iloc[-1]

                # Check for NaN values
                if pd.isna(current_price) or pd.isna(opening_price) or pd.isna(intraday_low) or pd.isna(previous_close):
                    logger.warning(
                        f"Incomplete data for {symbol}. Data: Current Price=${current_price}, "
                        f"Opening Price=${opening_price}, Intraday Low=${intraday_low}, "
                        f"Previous Close=${previous_close}"
                    )
                    continue

                stock_data[symbol] = {
                    'current_price': current_price,
                    'opening_price': opening_price,
                    'intraday_low': intraday_low,
                    'previous_close': previous_close
                }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue

        return stock_data
    except Exception as e:
        logger.error(f"Error fetching multiple stock data: {e}")
        if retries > 0:
            logger.info(f"Retrying in {delay} seconds... ({retries} retries left)")
            time.sleep(delay)
            return fetch_multiple_stock_data(symbols, retries - 1, delay)
        else:
            logger.error("Max retries exceeded. Skipping data fetch.")
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

    # Fetch portfolio data
    portfolio = portfolio_col.find_one({'_id': 'portfolio'})
    if not portfolio:
        logger.error("Portfolio not initialized. Aborting trading logic.")
        return

    current_cash = portfolio.get('current_cash', 0.0)

    # Check for new buy opportunities
    for symbol in sp500_symbols:
        # Check if there's a triggered buy or an open position
        if (triggered_buys_col.find_one({'symbol': symbol}) or
            positions_col.find_one({'symbol': symbol, 'is_open': True})):
            continue

        data = stock_data.get(symbol)
        if not data:
            continue

        condition_type = get_buy_condition_type(data)
        if not condition_type:
            continue  # No condition met

        # Determine target price based on condition type
        if condition_type == 'drop':
            target_price = data['current_price'] * 1.005  # 0.5% increase
            buy_amount = 5.0  # $5 investment
        elif condition_type == 'gap_down':
            target_price = data['current_price'] * 1.005   # Approximately 0.5% increase
            buy_amount = 5.0  # $5 investment
        else:
            logger.warning(f"Unknown condition type for {symbol}: {condition_type}")
            continue

        # Check if there's enough cash to buy
        if current_cash < buy_amount:
            logger.warning(f"Not enough cash to buy {symbol}. Current cash: ${current_cash}")
            continue

        # Record triggered buy condition
        new_trigger = {
            'symbol': symbol,
            'trigger_date': datetime.utcnow(),
            'condition_type': condition_type
        }
        try:
            triggered_buys_col.insert_one(new_trigger)
        except Exception as e:
            logger.error(f"Error inserting TriggeredBuy for {symbol}: {e}")
            continue

        # Open a new position
        buy_price = data['current_price']
        # target_price already determined

        shares = buy_amount / buy_price  # Calculate number of shares

        new_position = {
            'symbol': symbol,
            'buy_price': buy_price,
            'target_price': target_price,
            'buy_date': datetime.utcnow(),
            'is_open': True,
            'shares': shares,
            'condition_type': condition_type  # Store condition type
        }
        try:
            positions_col.insert_one(new_position)
            # Deduct buy_amount from current_cash
            portfolio_col.update_one({'_id': 'portfolio'}, {'$inc': {'current_cash': -buy_amount}})
            current_cash -= buy_amount
        except Exception as e:
            logger.error(f"Error inserting Position for {symbol}: {e}")
            continue

        # Log the buy trade
        buy_trade = {
            'symbol': symbol,
            'action': 'BUY',
            'price': buy_price,
            'shares': shares,
            'timestamp': datetime.utcnow(),
            'condition_type': condition_type  # Optional: log condition type
        }
        try:
            if validate_trade_data(buy_trade):
                trades_col.insert_one(buy_trade)
            else:
                logger.warning(f"Invalid trade data: {buy_trade}")
        except Exception as e:
            logger.error(f"Error logging BUY trade for {symbol}: {e}")
            continue

        logger.info(f"Bought {symbol} at ${buy_price:.2f}, target ${target_price:.2f}, shares: {shares:.4f}, condition: {condition_type}")

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
        shares = position.get('shares', 0.0)
        data = sell_data.get(symbol)
        if data and data['current_price'] >= position['target_price']:
            # Close the position
            proceeds = data['current_price'] * shares

            try:
                positions_col.update_one(
                    {'_id': position['_id']},
                    {'$set': {'is_open': False}}
                )
                # Add proceeds to current_cash
                portfolio_col.update_one({'_id': 'portfolio'}, {'$inc': {'current_cash': proceeds}})
            except Exception as e:
                logger.error(f"Error closing Position for {symbol}: {e}")
                continue

            logger.info(f"Sold {symbol} at ${data['current_price']:.2f}, shares: {shares:.4f}, proceeds: ${proceeds:.2f}")

            # Log the sell trade
            sell_trade = {
                'symbol': symbol,
                'action': 'SELL',
                'price': data['current_price'],
                'shares': shares,
                'timestamp': datetime.utcnow(),
                'condition_type': position.get('condition_type', 'unknown')  # Optional: log condition type
            }
            try:
                if validate_trade_data(sell_trade):
                    trades_col.insert_one(sell_trade)
                else:
                    logger.warning(f"Invalid trade data: {sell_trade}")
            except Exception as e:
                logger.error(f"Error logging SELL trade for {symbol}: {e}")
                continue

            # Remove from triggered buys
            try:
                triggered_buys_col.delete_one({'symbol': symbol})
            except Exception as e:
                logger.error(f"Error deleting TriggeredBuy for {symbol}: {e}")
                continue

def get_buy_condition_type(data, drop_threshold=5, gap_threshold=5):
    """
    Determines the buy condition type based on the stock data.

    Args:
        data (dict): Dictionary containing 'opening_price', 'intraday_low', 'previous_close'.
        drop_threshold (float): Percentage drop threshold.
        gap_threshold (float): Percentage gap down threshold.

    Returns:
        str or None: 'drop' or 'gap_down' if condition met, else None.
    """
    try:
        # Check for drop condition
        if 'opening_price' in data and 'intraday_low' in data:
            drop_percentage = ((data['opening_price'] - data['intraday_low']) / data['opening_price']) * 100
            if drop_percentage >= drop_threshold:
                return 'drop'

        # Check for gap down condition
        if 'previous_close' in data and 'opening_price' in data:
            gap_percentage = ((data['opening_price'] - data['previous_close']) / data['previous_close']) * 100
            if gap_percentage <= -gap_threshold:
                return 'gap_down'

        return None
    except Exception as e:
        logger.error(f"Error determining buy condition type: {e}")
        return None

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

def fetch_current_price(symbol):
    """
    Fetches the current price for a given symbol.

    Args:
        symbol (str): The stock ticker symbol.

    Returns:
        float or None: Current price if fetched successfully, else None.
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if data.empty:
            logger.warning(f"No current price data available for {symbol}.")
            return None
        current_price = data['Close'].iloc[-1]
        return current_price
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {e}")
        return None

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

def validate_trade_data(trade):
    """
    Validates trade data to ensure all required fields are present and valid.

    Args:
        trade (dict): The trade data dictionary.

    Returns:
        bool: True if valid, False otherwise.
    """
    required_fields = ['symbol', 'action', 'price', 'shares', 'timestamp']
    for field in required_fields:
        if field not in trade or pd.isna(trade[field]):
            logger.warning(f"Trade data missing or invalid for field '{field}': {trade}")
            return False
    return True

if __name__ == '__main__':
    app.run(debug=True)