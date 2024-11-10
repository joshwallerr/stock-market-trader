from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import pandas as pd
import requests






app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trading_bot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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

with app.app_context():
    db.create_all()






@app.route('/run-trades', methods=['POST'])
def run_trades():
    data = request.get_json()
    if not data or data.get('password') != 'rellaw':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        run_trading_logic()
        return jsonify({'status': 'Trading logic executed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500




def fetch_stock_data(symbol):
    try:
        # Set timezone to US/Eastern for US markets
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        start_date = now - timedelta(days=1)
        end_date = now

        # Fetch intraday data with 1-minute intervals
        stock = yf.Ticker(symbol)
        hist = stock.history(interval='1m', start=start_date, end=end_date)

        if hist.empty:
            return None

        # Get current and opening prices
        current_price = hist['Close'][-1]
        opening_price = hist['Open'][0]
        intraday_low = hist['Low'].min()

        return {
            'current_price': current_price,
            'opening_price': opening_price,
            'intraday_low': intraday_low
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None





def run_trading_logic():
    sp500_symbols = get_sp500_symbols()

    # Check for new buy opportunities
    for symbol in sp500_symbols:
        # Skip if already triggered or position is open
        if (TriggeredBuy.query.filter_by(symbol=symbol).first() or
            Position.query.filter_by(symbol=symbol, is_open=True).first()):
            continue

        data = fetch_stock_data(symbol)
        if data and check_buy_condition(data):
            # Record triggered buy condition
            new_trigger = TriggeredBuy(symbol=symbol, trigger_date=datetime.utcnow())
            db.session.add(new_trigger)
            db.session.commit()

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

    # Check for sell opportunities
    open_positions = Position.query.filter_by(is_open=True).all()
    for position in open_positions:
        data = fetch_stock_data(position.symbol)
        if data and data['current_price'] >= position.target_price:
            # Close the position
            position.is_open = False
            db.session.commit()

            # Remove from triggered buys
            triggered_buy = TriggeredBuy.query.filter_by(symbol=position.symbol).first()
            if triggered_buy:
                db.session.delete(triggered_buy)
                db.session.commit()



def check_buy_condition(data):
    drop_percentage = ((data['opening_price'] - data['intraday_low']) / data['opening_price']) * 100
    return drop_percentage >= 10



def get_sp500_symbols():
    """
    Fetches the current list of S&P 500 symbols from Wikipedia.
    
    Returns:
        list: A list of ticker symbols.
    """
    try:
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
        
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []



if __name__ == '__main__':
    app.run(debug=True)