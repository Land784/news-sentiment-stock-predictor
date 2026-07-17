import os
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_tokenizer = None
_model = None


def get_finbert():  # lazy loader for finbert
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model.eval()
    return _tokenizer, _model


def get_sentiment(text):
    tokenizer, finbert = get_finbert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = finbert(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    return scores[0][0].item() - scores[0][1].item()


TESLA_KEYWORDS = [
    "tesla", "tsla", "elon", "musk",
    "cybertruck", "model 3", "model y", "model s", "model x",
    "gigafactory", "supercharger", "autopilot", "fsd",
    "electric vehicle", "ev", "electric car",
    "spacex",
]


def filter_for_tesla(headline: str) -> bool:
    h = headline.lower()
    if any(k in h for k in TESLA_KEYWORDS):
        return True
    if ("electric" in h or "ev" in h) and ("stock" in h or "market" in h or "share" in h):
        return True
    return False


def load_news_data(path="data/news.csv"):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, path)

    newsDF = pd.read_csv(csv_path, names=["date", "headline"])
    newsDF["is_tesla"] = newsDF["headline"].apply(filter_for_tesla)
    newsDF = newsDF[newsDF["is_tesla"]].drop(columns=["is_tesla"])
    newsDF["date"] = pd.to_datetime(newsDF["date"])
    newsDF = newsDF.sort_values("date")
    return newsDF


def load_stock_data(ticker="TSLA", period="2y"):
    df = yf.download(ticker, period=period)

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index().rename(columns={"Date": "date"})
    df = df[["date", "Close"]]
    df["PrevClose"] = df["Close"].shift(1)
    df.dropna(inplace=True)
    return df


def add_daily_sentiment(news):
    news["sentiment"] = news["headline"].apply(get_sentiment)
    daily = news.groupby(news["date"].dt.date)["sentiment"].mean()
    daily = pd.DataFrame(daily)
    return daily


def build_merged_data():
    stocks = load_stock_data()
    news = load_news_data()
    sentiment = add_daily_sentiment(news)

    sentiment = sentiment.reset_index().rename(columns={"index": "date"})
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    merged = stocks.merge(sentiment, on="date", how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0)

    merged["NextClose"] = merged["Close"].shift(-1)
    merged["PercentChange"] = ((merged["NextClose"] - merged["Close"]) / merged["Close"]) * 100
    merged.dropna(inplace=True)

    news_days = len(merged[merged["sentiment"] != 0])
    print(f"Total trading days: {len(merged)}, Days with Tesla news: {news_days}")

    return merged


class StockDataset(Dataset):
    def __init__(self, normalize=True):
        df = build_merged_data()
        self.X = df[["sentiment", "Close"]].values
        self.y = df["PercentChange"].values.reshape(-1, 1)

        self.normalize = normalize
        if normalize:
            self.sentiment_mean = self.X[:, 0].mean()
            self.sentiment_std = self.X[:, 0].std() + 1e-8

            self.X[:, 1] = np.log(self.X[:, 1])
            self.price_mean = self.X[:, 1].mean()
            self.price_std = self.X[:, 1].std() + 1e-8

            self.target_mean = self.y.mean()
            self.target_std = self.y.std() + 1e-8

            self.X[:, 0] = (self.X[:, 0] - self.sentiment_mean) / self.sentiment_std
            self.X[:, 1] = (self.X[:, 1] - self.price_mean) / self.price_std
            self.y = (self.y - self.target_mean) / self.target_std

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def denormalize_prediction(self, normalized_pred):
        if self.normalize:
            return normalized_pred * self.target_std + self.target_mean
        return normalized_pred

    def normalize_input(self, sentiment, current_price):
        if self.normalize:
            norm_sentiment = (sentiment - self.sentiment_mean) / self.sentiment_std
            log_price = np.log(current_price)
            norm_price = (log_price - self.price_mean) / self.price_std
            return norm_sentiment, norm_price
        return sentiment, current_price

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    merged = build_merged_data()
    print(merged.head(20))