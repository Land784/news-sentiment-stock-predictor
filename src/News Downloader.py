import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time

API_KEY = os.environ.get("NEWSAPI_KEY")


def fetch_articles_for_date_range(from_date, to_date, query):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "from": from_date,
        "to": to_date,
        "apiKey": API_KEY,
        "page": 1
    }

    try:
        r = requests.get(url, params=params, timeout=10)

        if r.status_code != 200:
            return []

        data = r.json()

        if "articles" not in data:
            return []

        articles = []
        for item in data["articles"]:
            articles.append({
                "date": item.get("publishedAt", "")[:10],
                "headline": item.get("title", "")
            })

        return articles

    except Exception as e:
        print(f"  Error: {e}")
        return []


def load_news_data():

    print("Fetching Tesla/Elon Musk news from last 30 days...")
    print("(NewsAPI free tier limits to ~100 articles per query)\n")

    all_articles = []

    # Split into weekly chunks and use different search terms

    end_date = datetime.now()

    date_ranges = []
    for i in range(0, 30, 5):
        to_date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        from_date = (end_date - timedelta(days=i + 4)).strftime("%Y-%m-%d")
        date_ranges.append((from_date, to_date))

    queries = [
        "AAPL",
        "Apple",
        "iPhone"
    ]

    total_fetched = 0

    for query in queries:
        print(f"Searching for: '{query}'")

        for from_date, to_date in date_ranges:
            print(f"  {from_date} to {to_date}...", end=" ")

            articles = fetch_articles_for_date_range(from_date, to_date, query)

            if articles:
                all_articles.extend(articles)
                print(f"Found {len(articles)} articles")
                total_fetched += len(articles)
            else:
                print("No articles")

            time.sleep(0.5)

        print()

    if not all_articles:
        print("No articles found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    df = df[df['headline'].notna() & (df['headline'] != "")]
    df = df.drop_duplicates(subset=['headline'])

    df = df.sort_values('date', ascending=False)

    os.makedirs("../data", exist_ok=True)

    output_file = "../data/newsApple.csv"
    df.to_csv(output_file, index=False)

    #Summary
    print(f"Total articles fetched: {total_fetched}")
    print(f"✓ Saved {len(df)} unique articles to {output_file}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Show distribution by date
    print("\nArticles per date:")
    date_counts = df['date'].value_counts().sort_index(ascending=False)
    for date, count in date_counts.head(10).items():
        print(f"  {date}: {count} articles")

    print("\nSample headlines:")
    for _, row in df.head(5).iterrows():
        print(f"  {row['date']}: {row['headline'][:75]}...")

    return df


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Set the NEWSAPI_KEY environment variable before running this script.")
    df = load_news_data()