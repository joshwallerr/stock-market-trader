from flask import Flask, render_template, redirect, url_for, request, g
import threading
import sqlite3
import datetime
import json
import pandas as pd
from trading_script import run_trading_bot, trading_parameters, init_db
import sys
import os


DATABASE = 'trades.db'

app = Flask(__name__)

trading_thread = None
stop_trading = threading.Event()

# Initialize database if 'init_db' is passed as a command-line argument
if "init_db" in sys.argv:
    print("Removing old database...")
    try:
        os.remove(DATABASE)
    except FileNotFoundError:
        pass

    print("Initializing database...")
    init_db()


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.route('/')
def index():
    conn = get_db()
    c = conn.cursor()
    
    # Fetch closed trades
    c.execute("SELECT * FROM trades WHERE status = 'CLOSED'")
    trades = c.fetchall()
    trades_df = pd.DataFrame(trades, columns=['id', 'ticker', 'buy_price', 'sell_price', 'buy_date', 'sell_date', 'status', 'investment', 'return_pct', 'buy_rule', 'target_profit'])
    
    # Ensure datetime columns are properly parsed
    if not trades_df.empty:
        trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
        trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])
        
        # Calculate metrics
        today = datetime.date.today()
        one_day_ago = today - datetime.timedelta(days=1)
        one_week_ago = today - datetime.timedelta(weeks=1)
        one_month_ago = today - datetime.timedelta(days=30)
        
        daily_return = calculate_return(trades_df, one_day_ago, today)
        weekly_return = calculate_return(trades_df, one_week_ago, today)
        monthly_return = calculate_return(trades_df, one_month_ago, today)
        
        total_profit = trades_df['sell_price'].sum() - trades_df['buy_price'].sum()
        total_investment = trades_df['investment'].sum()
        total_return_pct = (total_profit / total_investment) * 100 if total_investment > 0 else 0

        # Trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    else:
        daily_return = weekly_return = monthly_return = total_profit = total_return_pct = 0
        total_trades = winning_trades = losing_trades = win_rate = 0
    
    # Prepare data for charts
    if not trades_df.empty:
        # Convert 'sell_date' to date
        trades_df['sell_date'] = trades_df['sell_date'].dt.date

        # Group by sell_date
        daily_profit = trades_df.groupby('sell_date').apply(lambda x: x['sell_price'].sum() - x['buy_price'].sum()).reset_index(name='profit')
        daily_profit['cumulative_profit'] = daily_profit['profit'].cumsum()

        # Convert to lists for chart.js
        chart_labels = daily_profit['sell_date'].astype(str).tolist()
        chart_data = daily_profit['cumulative_profit'].tolist()
    else:
        chart_labels = []
        chart_data = []

    return render_template('index.html',
                           daily_return=daily_return,
                           weekly_return=weekly_return,
                           monthly_return=monthly_return,
                           total_profit=total_profit,
                           total_return_pct=total_return_pct,
                           total_trades=total_trades,
                           winning_trades=winning_trades,
                           losing_trades=losing_trades,
                           win_rate=win_rate,
                           chart_labels=json.dumps(chart_labels),
                           chart_data=json.dumps(chart_data),
                           trading_parameters=trading_parameters)

def calculate_return(trades_df, start_date, end_date):
    period_trades = trades_df[(trades_df['sell_date'].dt.date >= start_date) & (trades_df['sell_date'].dt.date <= end_date)]
    profit = period_trades['sell_price'].sum() - period_trades['buy_price'].sum()
    investment = period_trades['investment'].sum()
    return (profit / investment) * 100 if investment > 0 else 0

@app.route('/start_trading', methods=['POST'])
def start_trading():
    global trading_thread, stop_trading, trading_parameters
    if trading_thread is None or not trading_thread.is_alive():
        # Get parameters from form
        decrease_threshold = float(request.form.get('decrease_threshold', -10))
        trading_parameters['decrease_threshold'] = decrease_threshold

        stop_trading.clear()
        trading_thread = threading.Thread(target=trading_logic)
        trading_thread.start()
    return redirect(url_for('index'))

@app.route('/stop_trading', methods=['POST'])
def stop_trading_route():
    global stop_trading
    stop_trading.set()
    return redirect(url_for('index'))

@app.route('/trades')
def trades():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM trades")
    trades = c.fetchall()
    trades_df = pd.DataFrame(trades, columns=['id', 'ticker', 'buy_price', 'sell_price', 'buy_date', 'sell_date', 'status', 'investment', 'return_pct', 'buy_rule', 'target_profit'])
    trades_df = trades_df.sort_values(by='id', ascending=False)
    return render_template('trades.html', trades=trades_df.to_dict(orient='records'))

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def trading_logic():
    run_trading_bot(stop_trading, 
                    decrease_threshold=trading_parameters['decrease_threshold'])

if __name__ == '__main__':
    app.run(port=5003)