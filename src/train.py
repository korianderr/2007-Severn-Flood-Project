"""
train.py

Runs training loop on each batch of scenarios. Sends through U-net, then adjusts
weights accordingly. 
"""


for epoch in range(n_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
