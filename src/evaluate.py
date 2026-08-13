"""
evaluate.py


"""

import torch
import torch.nn as nn

from model import FloodUNet
from dataset import load_datasets

model = FloodUNet()
model.load_state_dict(torch.load("data/flood_unet.pt"))
model.eval() # Tells model it is being used for predictions now and not training

criterion = nn.MSELoss()

def denormalise(array, min_val, max_val):
    """Converts back to real units"""
    return array * (max_val - min_val) + min_val

train_dataset, val_dataset, test_dataset = load_datasets()
x, y = test_dataset[0]

with torch.no_grad():
    x_batch = x.unsqueeze(0)
    prediction = model(x_batch)

pred_depth = denormalise(prediction, test_dataset.depth_min, test_dataset.depth_max)
true_depth = denormalise(y, test_dataset.depth_min, test_dataset.depth_max)

print("Prediction shape:", pred_depth.shape)
print("True depth shape:", true_depth.shape)
