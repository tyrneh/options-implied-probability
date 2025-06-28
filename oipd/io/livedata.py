import yfinance as yf
def get_ticker_data(ticker, strike_date):
    # Fetch option chain data for a given ticker and strike date
    opt = yf.Ticker(ticker).option_chain(strike_date)
    df = opt.calls[['strike', 'bid', 'ask', 'lastPrice']]
    return df