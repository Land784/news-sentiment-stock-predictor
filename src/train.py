import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StockDataset
from src.model import PricePredictor

EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in tqdm(dataloader, desc="Training"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("Loading dataset...")
    dataset = StockDataset()

    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset.X.shape}")
    print(f"Target shape: {dataset.y.shape}")

    #Debugging
    print(f"\nFeature statistics:")
    print(f"Sentiment - Mean: {dataset.X[:, 0].mean():.4f}, Std: {dataset.X[:, 0].std():.4f}")
    print(f"PrevClose - Mean: {dataset.X[:, 1].mean():.2f}, Std: {dataset.X[:, 1].std():.2f}")
    print(f"Target - Mean: {dataset.y.mean():.2f}, Std: {dataset.y.std():.2f}")

    # Split into train / val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PricePredictor().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n==== Epoch {epoch + 1}/{EPOCHS} ====")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save only if improved
        if val_loss < best_loss:
            best_loss = val_loss
            # Save to project root, not src folder
            save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}!")

    print(f"\nTraining complete. Best Val Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()