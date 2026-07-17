import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PricePredictor
from src.dataset import get_sentiment, load_stock_data


def run_demo():
    # Load model
    model = PricePredictor()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "model_best.pt")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Loading dataset for normalization parameters...")
    from src.dataset import StockDataset
    dataset = StockDataset(normalize=True)

    # User Input
    headline = input("\nEnter a news headline about Tesla: ")
    sentiment = get_sentiment(headline)
    print(f"Sentiment score: {sentiment:.4f}")

    # Get latest stock price
    stock_data = load_stock_data(period="5d")
    current_price = stock_data.iloc[-1]["Close"]
    print(f"Current closing price: ${current_price:.2f}")

    norm_sentiment, norm_price = dataset.normalize_input(sentiment, current_price)
    X = torch.tensor([[norm_sentiment, norm_price]], dtype=torch.float32)

    # Make prediction (this is percent change)
    with torch.no_grad():
        norm_pred = model(X)

    percent_change = dataset.denormalize_prediction(norm_pred.item())

    predicted_price = current_price * (1 + percent_change / 100)
    dollar_change = predicted_price - current_price

    print(f"\n{'=' * 50}")
    print(f"Predicted percent change: {percent_change:+.2f}%")
    print(f"Predicted next-day closing price: ${predicted_price:.2f}")
    print(f"Expected dollar change: ${dollar_change:+.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_demo()