import torch
import torch.nn as nn


class PricePredictor(nn.Module):
    def __init__(self, input_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = PricePredictor()
    sample = torch.randn(1, 2)
    print(f"Input shape: {sample.shape}")
    output = model(sample)
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")