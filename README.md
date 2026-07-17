# News Sentiment Stock Predictor

Small side project that tries to predict next-day % change in TSLA's closing price using a mix of Tesla-related news sentiment and the current price.

Pipeline is: pull headlines, filter down to ones actually about Tesla, run them through FinBERT to get a sentiment score, average that by day, merge it with daily close prices from yfinance, and feed `[sentiment, price]` into a small feedforward net that predicts the next day's percent change.

Don't take the predictions seriously, this was mostly an excuse to build a full data -> model -> inference pipeline end to end.

## How it fits together

- `src/News Downloader.py` — pulls headlines from NewsAPI over a date range and dumps them to CSV
- `stock.py` — pulls 2 years of TSLA daily prices from yfinance
- `src/dataset.py` — filters headlines for Tesla relevance (keyword match on things like "tesla", "cybertruck", "elon", etc.), scores each one with FinBERT, averages sentiment per day, and merges it onto the price data
- `src/model.py` — the model itself, a 3-layer MLP (2 -> 64 -> 32 -> 1) with dropout
- `src/train.py` — trains it, saves the best checkpoint to `model_best.pt` whenever validation loss improves
- `src/inference.py` — loads the trained model, asks you for a headline, and prints a predicted price move
- `analysis/timing.py` — rough benchmark of dataset load / train step / inference time on synthetic data

`data/news.csv` and `tesla_prices_2y.csv` are the data this was actually trained on, and `model_best.pt` is the resulting checkpoint, so you can run inference right away without retraining anything.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to pull fresh headlines with `News Downloader.py`, you'll need a NewsAPI key (free tier works, [newsapi.org](https://newsapi.org)):

```
export NEWSAPI_KEY=your_key_here
```

Copy `.env.example` to `.env` if you'd rather keep it out of your shell history.

## Running things

```
make test    # run the test suite
make demo    # load model_best.pt and predict off a headline you type in
make train   # retrain from scratch, overwrites model_best.pt on improvement
make timing  # quick perf benchmark
```

`make demo` will also hit yfinance for the latest TSLA close, so you need a working internet connection for it.

## Notes / limitations

- The dataset is tiny (~500 trading days, only a subset of those actually have matching Tesla news), so don't expect this to generalize well.
- FinBERT sentiment is computed per-headline and just averaged per day, no weighting by source or article length.
- Model only sees two features (sentiment, price), it's intentionally simple.
