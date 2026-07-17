import sys
import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PricePredictor


def create_synthetic_dataset(size):
    X = torch.randn(size, 2, dtype=torch.float32)
    y = torch.randn(size, 1, dtype=torch.float32)
    return TensorDataset(X, y)


def time_training_epoch(model, dataloader, optimizer, criterion):
    model.train()
    start = time.time()

    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

    end = time.time()
    return end - start


def time_model_inference(model, num_samples):
    model.eval()
    data = torch.randn(num_samples, 2)

    with torch.no_grad():
        start = time.time()
        predictions = model(data)
        end = time.time()

    return end - start


def main():
    print("TIMING ANALYSIS")
    print("=" * 50)

    small_size = 100
    large_size = 10000

    print(f"\nSmall dataset: {small_size} samples")
    print(f"Large dataset: {large_size} samples\n")

    print("Dataset Loading:")
    start = time.time()
    small_dataset = create_synthetic_dataset(small_size)
    small_load_time = time.time() - start

    start = time.time()
    large_dataset = create_synthetic_dataset(large_size)
    large_load_time = time.time() - start

    print(f"  Small: {small_load_time:.4f}s")
    print(f"  Large: {large_load_time:.4f}s")

    print("\nTraining (one epoch):")
    criterion = torch.nn.MSELoss()
    batch_size = 8

    small_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)
    model = PricePredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    small_train_time = time_training_epoch(model, small_loader, optimizer, criterion)

    large_loader = DataLoader(large_dataset, batch_size=batch_size, shuffle=True)
    model = PricePredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    large_train_time = time_training_epoch(model, large_loader, optimizer, criterion)

    print(f"  Small: {small_train_time:.4f}s")
    print(f"  Large: {large_train_time:.4f}s")

    # Inference Time
    print("\nModel Inference:")
    model = PricePredictor()
    model.eval()

    small_inference_time = time_model_inference(model, small_size)
    large_inference_time = time_model_inference(model, large_size)

    print(f"  Small: {small_inference_time:.4f}s")
    print(f"  Large: {large_inference_time:.4f}s")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()