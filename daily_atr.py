import warnings

# FutureWarning 무시
warnings.filterwarnings("ignore", category=FutureWarning)
# Import Zipline functions that we need
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol, schedule_function, date_rules, time_rules, record
import random

# Import date and time zone libraries
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import talib
import pyfolio as pf

# Import visualization
import matplotlib.pyplot as plt


def print_array(array, array_name="Array"):
    print(f"--- {array_name} ---")
    print(array)
    print(f"Min: {array.min()}, Max: {array.max()}")
    print("---------------------")

def normalization(array, array_name="Array"):
    min_val = array.min()
    max_val = array.max()
    if min_val == max_val:
        print_array(array, array_name)
        raise ValueError(f"{array_name} has min and max values equal, normalization not possible.")
    array = (array - min_val) / (max_val - min_val)
    return array


def initialize(context):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    now_sp500_ticker = list(tables[0]['Symbol'])
    all_historic_sp500 = list(set(now_sp500_ticker))
    tickers = [str(ticker) for ticker in all_historic_sp500 if ticker != "nan"]
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    tickers = sorted(tickers)

    # Make symbol list from tickers
    lst = []
    for s in tickers:
        try:
            lst.append(symbol(s))
        except:
            continue

    context.universe = lst

    # Moving average window
    context.history_window = 60




def handle_data(context, data):
    print("***")
    record(leverage=context.account.leverage)
    cash_amount = context.portfolio.cash

    # 현재 포트폴리오의 총 자산
    total_assets = context.portfolio.portfolio_value

    # 총 자산의 5%를 계산
    cash_allocation = int(0.05 * total_assets)

    # 현재 보유 현금을 총 자산의 5%로 나눈 몫
    cash_allocation_share = int(cash_amount // cash_allocation)

    if cash_allocation_share > 0:
        # Request history for the stock
        hist = data.history(context.universe, ["close", "high", "low"], context.history_window, "1d")
        # date_lst = hist.index
        atr_df = pd.DataFrame()
        rsi_df = pd.DataFrame()
        # macd, macdsignal, macdhist = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

        for stock in context.universe:
            high_prices = np.array(hist.loc[(slice(None), stock), 'high'].values, dtype=np.float64)
            low_prices = np.array(hist.loc[(slice(None), stock), 'low'].values, dtype=np.float64)
            close_prices = np.array(hist.loc[(slice(None), stock), 'close'].values, dtype=np.float64)

            atr = np.array(talib.ATR(high_prices, low_prices, close_prices, timeperiod=14))
            rsi = talib.RSI(close_prices, timeperiod=14)
            # macd, macdsignal, macdhist = talib.MACD(close_prices, 12, 26, 9)

            # Normalize the series if they are not None and not empty
            atr = atr[np.logical_not(np.isnan(atr))]
            rsi = rsi[np.logical_not(np.isnan(rsi))]

            if atr is not None and len(atr) > 14:
                normalized_atr = normalization(atr)
                temp_df = pd.DataFrame({stock: normalized_atr})
                atr_df = pd.concat([atr_df, temp_df], axis=1)

            if rsi is not None and len(atr) > 14:
                temp_df = pd.DataFrame({stock: rsi})
                rsi_df = pd.concat([rsi_df, temp_df], axis=1)

            else:
                pass

        last_3_days_data = atr_df.iloc[-3:]

        # Step 2: Calculate the mean for each column
        means = last_3_days_data.mean().nlargest(cash_allocation_share).index

        monthly_buy_list = means

        for buy_stock in monthly_buy_list:
            if data.can_trade(buy_stock) and not buy_stock in list(context.portfolio.positions.keys()):
                if rsi_df[buy_stock].iloc[-1] < 30:
                    order_target_percent(buy_stock, 0.1)

    for stock in context.portfolio.positions:
        position = context.portfolio.positions[stock]
        cost_basis = position.cost_basis

        current_price = data.current(stock, 'price')
        percent_change = (current_price - cost_basis) / cost_basis * 100

        current_position = context.portfolio.positions[stock].amount

        # 손실이 2% 이상 발생하면 청산
        if percent_change <= -3:
            order_target_percent(stock, 0)

        # 수익이 4% 이상 발생하면 청산
        elif percent_change >= 10:
            order_target_percent(stock, 0)


def analyze(context, perf):
    fig = plt.figure(figsize=(12, 8))

    # First chart
    ax = fig.add_subplot(311)
    ax.set_title(f'Strategy Results{start} to {end}, momentom, equal weight')
    ax.semilogy(perf['portfolio_value'], linestyle='-',
                label='Equity Curve', linewidth=3.0)
    ax.legend()
    ax.grid(False)

    # Second chart
    ax = fig.add_subplot(312)
    ax.plot(perf['gross_leverage'],
            label='Exposure', linestyle='-', linewidth=1.0)
    ax.legend()
    ax.grid(True)

    # Third chart
    ax = fig.add_subplot(313)
    ax.plot(perf['returns'], label='Returns', linestyle='-.', linewidth=1.0)
    ax.legend()
    ax.grid(True)
    plt.show()


# Set start and end date
start = pd.Timestamp(2015, 10, 1)
end = pd.Timestamp(2017, 12, 31)

# Fire off the backtest
results = run_algorithm(
    start=start,
    end=end,
    initialize=initialize,
    analyze=analyze,
    handle_data=handle_data,
    capital_base=10000,
    data_frequency='daily', bundle='quandl'
)

# Extract daily portfolio value
returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)

# Create a PyFolio tear sheet
pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions)