"""
evaluate.py

Evaluates the trained flood U-net against the tests. Generates a
comparison plot of the depths, then MAE, RMSE and CSI to evaluate
error.

Test scenarios sit ~4.5cm apart in boundary_wse (100 scenarios over
9.5-14.0 mAOD), so these metrics measure interpolation within the
trained range rather than generalisation to unseen conditions.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

from model import FloodUNet
from dataset import load_datasets
from solver import bathtub_fill

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

pred_2d = pred_depth.squeeze().numpy()
true_2d = true_depth.squeeze().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

vmax = true_2d.max()
axes[0].imshow(true_2d, cmap='Blues', vmin=0, vmax=vmax)
axes[1].imshow(pred_2d, cmap='Blues', vmin=0, vmax=vmax)

axes[0].set_title('True depth')
axes[1].set_title('Predicted depth')

plt.show()

### CALCULATE ERRORS

abs_error_sum = 0.0
sq_error_sum = 0.0
pixel_count = 0
tp, fp, fn = 0, 0, 0 # true positive, false positive, false negative

mask = test_dataset.mask

with torch.no_grad():
    for x, y in test_dataset:
        prediction = model(x.unsqueeze(0))

        pred_depth = denormalise(prediction, test_dataset.depth_min, test_dataset.depth_max).squeeze().numpy()
        true_depth = denormalise(y, test_dataset.depth_min, test_dataset.depth_max).squeeze().numpy()

        # Accumulate abs_error_sum, sq_error_sum, pixel_count from pred_depth vs true_depth and apply mask
        error = (pred_depth - true_depth)[mask]
        abs_error_sum += np.abs(error).sum()
        sq_error_sum += (error ** 2).sum()
        pixel_count += mask.sum()

        flood_threshold = 0.05  # below this is noise
        pred_flooded = (pred_depth > flood_threshold) & mask
        true_flooded = (true_depth > flood_threshold) & mask


        # Update tp, fp, fn using pred_flooded and true_flooded
        tp += (pred_flooded & true_flooded).sum()
        fp += (pred_flooded & ~true_flooded).sum()
        fn += (~pred_flooded & true_flooded).sum()

mae = abs_error_sum / pixel_count
rmse = (sq_error_sum / pixel_count) ** 0.5
csi = tp / (tp + fp + fn)

print(f"MAE: {mae:.4f} m")
print(f"RMSE: {rmse:.4f} m")
print(f"CSI: {csi:.4f}")

### TESTING SPEED AGAINST BATHTUB SPEED

model_times = []
solver_times = []

with torch.no_grad():
    for i, (x, y) in enumerate(test_dataset):
        start = time.perf_counter()
        _ = model(x.unsqueeze(0))
        model_times.append(time.perf_counter() - start)

        boundary_wse = float(test_dataset.rows[i]['boundary_wse'])
        start = time.perf_counter()
        _ = bathtub_fill(test_dataset.z, boundary_wse=boundary_wse)
        solver_times.append(time.perf_counter() - start)

avg_model_time = sum(model_times) / len(model_times)
avg_solver_time = sum(solver_times) / len(solver_times)

print(f"Avg model time: {avg_model_time:.4f}s")
print(f"Avg solver time: {avg_solver_time:.4f}s")
print(f"Speed-up: {avg_solver_time / avg_model_time:.2f}x")
