"""
train.py

Training loop for FloodUNet. Loads the prepared train/val datasets,
runs a fixed number of epochs of forward/loss/backward/update, and
prints the training loss each epoch so progress is visible.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_datasets
from model import FloodUNet

train_dataset, val_dataset, test_dataset = load_datasets()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

model = FloodUNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50

for epoch in range(n_epochs):
    train_loss_total = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()

    val_los_total = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                val_los_total += loss.item()

    avg_loss = train_loss_total / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs} - loss: {avg_loss:.4f}")
    avg_loss = val_los_total / len(val_loader)
    print(f"Epoch {epoch+1}/{n_epochs} - loss: {avg_loss:.4f}")
