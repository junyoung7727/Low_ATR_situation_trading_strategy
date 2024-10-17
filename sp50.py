import pandas as pd
import yfinance as yf
start = pd.Timestamp(2016, 10, 1)
end = pd.Timestamp(2017, 12, 31)
df = yf.download('^GSPC','2016-10-01','2017-12-31')
print(df)

print((df['Adj Close'].iloc[-1]-df['Adj Close'].iloc[0])/df['Adj Close'].iloc[0])
