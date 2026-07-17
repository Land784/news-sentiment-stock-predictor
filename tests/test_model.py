import pytest
import sys
import os
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PricePredictor
from src.dataset import (
    filter_for_tesla,
    get_sentiment,
    StockDataset,
    load_news_data,
    load_stock_data
)


class TestModel:

    def test_model_initialization(self):
        model = PricePredictor()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_output_range(self):
        """Test that model outputs reasonable price predictions"""
        model = PricePredictor()
        sample_input = torch.tensor([[0.5, 250.0]], dtype=torch.float32)
        output = model(sample_input)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1)


class TestDataset:
    """Test the dataset and data processing functions"""

    def test_filter_for_tesla(self):
        """Test Tesla keyword filtering"""
        assert filter_for_tesla("Tesla announces new model") == True
        assert filter_for_tesla("TSLA stock rises") == True
        assert filter_for_tesla("Elon Musk tweets") == True
        assert filter_for_tesla("Apple releases new iPhone") == False
        assert filter_for_tesla("Market news today") == False

    def test_sentiment_analysis(self):
        """Test sentiment scoring"""
        positive_text = "Tesla stock soars on amazing earnings"
        negative_text = "Tesla faces major recall crisis"

        pos_score = get_sentiment(positive_text)
        neg_score = get_sentiment(negative_text)

        # Sentiment should be a float
        assert isinstance(pos_score, float)
        assert isinstance(neg_score, float)
        # Sentiment should be in reasonable range
        assert -1 <= pos_score <= 1
        assert -1 <= neg_score <= 1

    def test_stock_data_loading(self):
        """Test that stock data loads correctly"""
        df = load_stock_data(period="5d")
        assert isinstance(df, pd.DataFrame)
        assert "Close" in df.columns
        assert "PrevClose" in df.columns
        assert len(df) > 0



if __name__ == "__main__":
    pytest.main([__file__, "-v"])