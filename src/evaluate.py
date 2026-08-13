"""
evaluate.py


"""

from model import FloodUNet
import torch

model = FloodUNet()
model.load_state_dict(torch.load("data/flood_unet.pt"))
model.eval() # Tells model it is being used for predictions now and not training
