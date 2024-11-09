import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import datetime
import time

DATABASE = 'trades.db'

# Global parameters
trading_parameters = {
    'decrease_threshold': -10,  # Percentage decrease to trigger buy
    'gap_up_threshold': 5,       # Gap up percentage to trigger buy
    'short_window': 20,          # Short moving average window
    'long_window': 50,           # Long moving average window
}

# Sell target profits for each buy rule
sell_targets = {
    'intraday_decrease': 0.5,   # Sell at 0.5% increase
    'gap_up': 5,                # Sell at 5% increase
    'ma_crossover': 2           # Sell at 2% increase
}

def run_trading_bot(stop_event, decrease_threshold=-10):
    tickers = get_sp500_tickers()
    init_db()
    while not stop_event.is_set():
        try:
            check_for_trades(
                tickers,
                decrease_threshold=decrease_threshold,
                gap_up_threshold=trading_parameters['gap_up_threshold'],
                short_window=trading_parameters['short_window'],
                long_window=trading_parameters['long_window']
            )
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(60)  # Wait for 60 seconds before next check
    print("Trading bot stopped.")

def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            buy_price REAL NOT NULL,
            sell_price REAL,
            buy_date TEXT NOT NULL,
            sell_date TEXT,
            status TEXT NOT NULL,
            investment REAL NOT NULL DEFAULT 1000,
            return_pct REAL,
            buy_rule TEXT,
            target_profit REAL
        )
    ''')
    conn.commit()
    conn.close()

def check_for_trades(tickers, decrease_threshold, gap_up_threshold, short_window, long_window):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT DISTINCT ticker FROM trades WHERE status = 'OPEN'")
    open_trades = c.fetchall()
    open_tickers = [t[0] for t in open_trades]
    available_tickers = [ticker for ticker in tickers if ticker not in open_tickers]

    if not available_tickers:
        conn.close()
        return

    for ticker in available_tickers:
        try:
            # Fetch daily data for the last 3 months (approx 90 days)
            data_daily = yf.download(ticker, period='3mo', interval='1d', progress=False)
            if data_daily.empty:
                print(f"No daily data for {ticker}")
                continue

            # Fetch intraday data for today (1d, 1m)
            data_intraday = yf.download(ticker, period='1d', interval='1m', progress=False)
            if data_intraday.empty or 'Close' not in data_intraday.columns:
                print(f"No intraday data for {ticker}")
                continue

            bought = False  # Flag to check if we've already bought the stock

            # Rule 1: Decrease threshold rule
            ticker_data_intraday = data_intraday['Close']
            if isinstance(ticker_data_intraday, pd.DataFrame):
                # Handle multi-level columns
                ticker_data_intraday = ticker_data_intraday.iloc[:, 0]

            ticker_data_intraday = ticker_data_intraday.dropna()
            if not ticker_data_intraday.empty:
                # Extract scalar values
                try:
                    current_price = float(ticker_data_intraday.iloc[-1].item())
                    day_open = float(ticker_data_intraday.iloc[0].item())
                except Exception as e:
                    print(f"Error extracting prices for {ticker}: {e}")
                    continue

                percent_change = (current_price - day_open) / day_open * 100

                if percent_change <= decrease_threshold:
                    # Buy logic here
                    buy_price = current_price
                    buy_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    investment = 1000
                    buy_rule = 'intraday_decrease'
                    target_profit = sell_targets.get(buy_rule, 0.5)
                    c.execute('''
                        INSERT INTO trades (ticker, buy_price, buy_date, status, investment, buy_rule, target_profit)
                        VALUES (?, ?, ?, 'OPEN', ?, ?, ?)
                    ''', (ticker, buy_price, buy_date, investment, buy_rule, target_profit))
                    print(f"Bought {ticker} at ${buy_price:.2f} due to decrease threshold rule")
                    bought = True

            if not bought:
                # Rule 2: Gap up rule
                if len(data_daily) >= 2:
                    today_data = data_daily.iloc[-1]
                    yesterday_data = data_daily.iloc[-2]

                    today_open = today_data['Open']
                    yesterday_close = yesterday_data['Close']

                    try:
                        # Ensure scalar values
                        today_open = float(today_open.iloc[0]) if isinstance(today_open, pd.Series) else float(today_open)
                        yesterday_close = float(yesterday_close.iloc[0]) if isinstance(yesterday_close, pd.Series) else float(yesterday_close)
                    except Exception as e:
                        print(f"Error extracting open/close prices for {ticker}: {e}")
                        continue

                    gap_percentage = (today_open - yesterday_close) / yesterday_close * 100

                    if gap_percentage >= gap_up_threshold:
                        # Buy logic here
                        buy_price = today_open
                        buy_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        investment = 1000
                        buy_rule = 'gap_up'
                        target_profit = sell_targets.get(buy_rule, 5)
                        c.execute('''
                            INSERT INTO trades (ticker, buy_price, buy_date, status, investment, buy_rule, target_profit)
                            VALUES (?, ?, ?, 'OPEN', ?, ?, ?)
                        ''', (ticker, buy_price, buy_date, investment, buy_rule, target_profit))
                        print(f"Bought {ticker} at ${buy_price:.2f} due to gap up rule")
                        bought = True
                else:
                    print(f"Not enough data for gap up rule for {ticker}")

            if not bought:
                # Rule 3: Moving average crossover rule
                if len(data_daily) >= long_window:
                    close_prices = data_daily['Close']
                    if isinstance(close_prices, pd.DataFrame):
                        # Handle multi-level columns
                        close_prices = close_prices.iloc[:, 0]

                    data_daily['Short_MA'] = close_prices.rolling(window=short_window).mean()
                    data_daily['Long_MA'] = close_prices.rolling(window=long_window).mean()

                    # Generate signals
                    data_daily['Signal'] = 0
                    try:
                        # Instead of slicing with .loc, compute 'Signal' as 1 where Short_MA > Long_MA
                        data_daily['Signal'] = (data_daily['Short_MA'] > data_daily['Long_MA']).astype(int)
                    except Exception as e:
                        print(f"Error generating Signal for {ticker}: {e}")
                        continue

                    data_daily['Position'] = data_daily['Signal'].diff()

                    # Check if there was a bullish crossover today
                    try:
                        position = data_daily['Position'].iloc[-1]
                        # Ensure position is a scalar
                        if isinstance(position, pd.Series):
                            position = float(position.iloc[0])
                        else:
                            position = float(position)

                        if position == 1:
                            # Buy logic here
                            buy_price = float(close_prices.iloc[-1].item())
                            buy_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            investment = 1000
                            buy_rule = 'ma_crossover'
                            target_profit = sell_targets.get(buy_rule, 2)
                            c.execute('''
                                INSERT INTO trades (ticker, buy_price, buy_date, status, investment, buy_rule, target_profit)
                                VALUES (?, ?, ?, 'OPEN', ?, ?, ?)
                            ''', (ticker, buy_price, buy_date, investment, buy_rule, target_profit))
                            print(f"Bought {ticker} at ${buy_price:.2f} due to moving average crossover")
                            bought = True
                    except Exception as e:
                        print(f"Error checking crossover position for {ticker}: {e}")
                else:
                    print(f"Not enough data for moving average crossover for {ticker}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Now process sell conditions for open trades
    try:
        c.execute("SELECT id, ticker, buy_price, investment, target_profit FROM trades WHERE status = 'OPEN'")
        open_trades = c.fetchall()
        for trade_id, ticker, buy_price, investment, target_profit in open_trades:
            try:
                ticker_data = yf.download(ticker, period='1d', interval='1m', progress=False)
                if ticker_data.empty or 'Close' not in ticker_data.columns:
                    print(f"No data for open trade {trade_id} for ticker {ticker}")
                    continue
                close_prices = ticker_data['Close']
                if isinstance(close_prices, pd.DataFrame):
                    # Handle multi-level columns
                    close_prices = close_prices.iloc[:, 0]

                close_prices = close_prices.dropna()
                if close_prices.empty:
                    print(f"No close prices for open trade {trade_id} for ticker {ticker}")
                    continue
                try:
                    current_price = float(close_prices.iloc[-1].item())
                except Exception as e:
                    print(f"Error extracting current price for {ticker}: {e}")
                    continue
                target_price = float(buy_price) * (1 + target_profit / 100)

                if current_price >= target_price:
                    sell_price = current_price
                    sell_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    return_pct = (sell_price - buy_price) / buy_price * 100
                    c.execute('''
                        UPDATE trades
                        SET sell_price = ?, sell_date = ?, status = 'CLOSED', return_pct = ?
                        WHERE id = ?
                    ''', (sell_price, sell_date, return_pct, trade_id))
                    print(f"Sold {ticker} at ${sell_price:.2f} with return {return_pct:.2f}%")
            except Exception as e:
                print(f"Error processing trade {trade_id} for {ticker}: {e}")
    except Exception as e:
        print(f"Error processing sell conditions: {e}")

    conn.commit()
    conn.close()