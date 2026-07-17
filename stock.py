import yfinance as yf
import pandas as pd

def download_tesla_prices():
    ticker = "TSLA"
    df = yf.download(ticker, period="2y")  # last 2 years
    df.reset_index(inplace=True)          # move Date from index → column
    df.to_csv("tesla_prices_2y.csv", index=False)
    print("Saved tesla_prices_2y.csv with", len(df), "rows")

if __name__ == "__main__":
    download_tesla_prices()
